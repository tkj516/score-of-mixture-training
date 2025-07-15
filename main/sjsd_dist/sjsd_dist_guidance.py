import contextlib
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import dnnlib
from main.sjsd_dist.edm_network import get_edm_network
from main.sjsd_dist.utils import AlphaSampler, AlphaScheduler


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    # from https://github.com/crowsonkb/k-diffusion
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def load_edm_model(args):
    if args.dataset_name == "mnist":
        state_dict = torch.load(args.model_id, map_location="cpu")
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    else:
        with dnnlib.util.open_url(args.model_id) as f:
            temp_edm = pickle.load(f)["ema"]
        if args.dataset_name == "cifar10":
            del temp_edm.model.map_augment
            temp_edm.model.map_augment = None
        state_dict = temp_edm.state_dict()
        del temp_edm

    return state_dict


class SJSDDistGuidance(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.accelerator = accelerator

        # initialize the unets
        state_dict = load_edm_model(args)
        self.real_unet = get_edm_network(args)
        self.real_unet.load_state_dict(state_dict, strict=True)
        self.real_unet.requires_grad_(False)
        del self.real_unet.model.map_augment
        self.real_unet.model.map_augment = None

        self.fake_unet = copy.deepcopy(self.real_unet)
        self.fake_unet.requires_grad_(True)

        # some training hyper-parameters
        self.sigma_data = args.sigma_data
        self.sigma_max = args.sigma_max
        self.sigma_min = args.sigma_min
        self.rho = args.rho

        if args.dataset_name == "imagenet":
            self.cls_pred_branch = nn.Sequential(
                nn.Conv2d(
                    kernel_size=4,
                    in_channels=768,
                    out_channels=768,
                    stride=2,
                    padding=1,
                ),  # 8x8 -> 4x4
                nn.GroupNorm(num_groups=32, num_channels=768),
                nn.SiLU(),
                nn.Conv2d(
                    kernel_size=4,
                    in_channels=768,
                    out_channels=768,
                    stride=4,
                    padding=0,
                ),  # 4x4 -> 1x1
                nn.GroupNorm(num_groups=32, num_channels=768),
                nn.SiLU(),
                nn.Conv2d(
                    kernel_size=1, in_channels=768, out_channels=1, stride=1, padding=0
                ),  # 1x1 -> 1x1
            )
        else:
            self.cls_pred_branch = nn.Sequential(
                nn.Conv2d(
                    kernel_size=4,
                    in_channels=256,
                    out_channels=256,
                    stride=2,
                    padding=1,
                ),  # 8x8 -> 4x4
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.SiLU(),
                nn.Conv2d(
                    kernel_size=4,
                    in_channels=256,
                    out_channels=256,
                    stride=4,
                    padding=0,
                ),  # 4x4 -> 1x1
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.SiLU(),
                nn.Conv2d(
                    kernel_size=1, in_channels=256, out_channels=1, stride=1, padding=0
                ),  # 1x1 -> 1x1
            )
        self.cls_pred_branch.requires_grad_(True)

        self.num_train_timesteps = args.num_train_timesteps
        # small sigma first, large sigma later
        karras_sigmas = torch.flip(
            get_sigmas_karras(
                self.num_train_timesteps,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                rho=self.rho,
            ),
            dims=[0],
        )
        self.register_buffer("karras_sigmas", karras_sigmas)

        self.min_step = int(args.min_step_percent * self.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.num_train_timesteps)

        self.guidance_alpha_sampler = (
            AlphaScheduler(
                args.train_iters,
                args.adaptive_alpha_start_partitions,
                args.adaptive_alpha_end_partitions,
            )
            if args.guidance_alpha_scheduler == "adaptive"
            else AlphaSampler(
                args.guidance_alpha_scheduler,
                maximum=args.max_partitions,
                p=args.point_mass_prob,
            )
        )
        self.generator_alpha_sampler = (
            AlphaScheduler(
                args.train_iters,
                args.adaptive_alpha_start_partitions,
                args.adaptive_alpha_end_partitions,
            )
            if args.generator_alpha_scheduler == "adaptive"
            else AlphaSampler(
                args.generator_alpha_scheduler,
                maximum=args.max_partitions,
                p=args.point_mass_prob,
            )
        )

    def compute_density_ratio(self, noisy_latents, timestep_sigmas, labels):
        with torch.no_grad() if not self.joint_training else contextlib.nullcontext():
            rep = self.fake_unet(
                noisy_latents, timestep_sigmas, labels, return_bottleneck=True
            ).float()
        return self.cls_pred_branch(rep)

    def scale(
        self, noisy_latents, timestep_sigmas, alphas, labels, return_density_ratio=False
    ):
        density_ratio = self.compute_density_ratio(
            noisy_latents, timestep_sigmas, labels
        )
        scale = torch.sigmoid(density_ratio + torch.log(alphas / (1 - alphas)))
        return (scale, density_ratio) if return_density_ratio else scale

    def interpolated_model(self, noisy_latents, timestep_sigmas, alphas, labels):
        scale, _ = self.scale(
            noisy_latents, timestep_sigmas, alphas, labels, return_density_ratio=True
        )

        fake_pred = self.fake_unet(noisy_latents, timestep_sigmas, labels).detach()
        real_pred = self.real_unet(noisy_latents, timestep_sigmas, labels).detach()
        return scale * real_pred + (1 - scale) * fake_pred

    def compute_distribution_matching_loss(self, latents, labels, step=None):
        original_latents = latents
        batch_size = latents.shape[0]

        timesteps = torch.randint(
            self.min_step,
            min(self.max_step + 1, self.num_train_timesteps),
            [batch_size, 1, 1, 1],
            device=latents.device,
            dtype=torch.long,
        )
        timestep_sigma = self.karras_sigmas[timesteps]

        alphas = (
            self.generator_alpha_sampler(batch_size, step=step)
            .reshape(-1, 1, 1, 1)
            .to(latents.device)
        )

        noise = torch.randn_like(latents)
        noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise

        additional_log_dict = {}
        with torch.no_grad():
            pred_real_image = self.real_unet(noisy_latents, timestep_sigma, labels)
            pred_fake_image = self.fake_unet(noisy_latents, timestep_sigma, labels)

            p_real = latents - pred_real_image
            p_fake = latents - pred_fake_image
            weight_factor = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
            grad = (
                self.scale(noisy_latents, timestep_sigma, alphas, labels)
                * (p_real - p_fake)
                / weight_factor
                / alphas
            )
        additional_log_dict.update(
            {
                "dmtrain_pred_real_image": pred_real_image.detach(),
                "dmtrain_pred_fake_image": pred_fake_image.detach(),
            }
        )

        grad = torch.nan_to_num(grad)
        loss = 0.5 * F.mse_loss(
            original_latents, (original_latents - grad).detach(), reduction="mean"
        )

        # Add a GAN loss
        density_ratio = self.compute_density_ratio(
            noisy_latents, timestep_sigma, labels
        )

        loss += (
            3e-3
            * (F.softplus(-(density_ratio + torch.log(alphas / (1 - alphas))))).mean()
        )

        loss_dict = {"loss_dm": loss}

        dm_log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach(),
            "dmtrain_grad": grad.detach(),
            "dmtrain_gradient_norm": torch.norm(grad).item(),
            "dmtrain_timesteps": timesteps.detach(),
            "dmtrain_alphas": alphas.detach(),
            "dmtrain_scale": self.scale(
                noisy_latents, timestep_sigma, alphas, labels
            ).detach(),
            "dmtrain_weight_factor": weight_factor.detach(),
        }
        dm_log_dict.update(additional_log_dict)
        return loss_dict, dm_log_dict

    def compute_loss_fake(
        self,
        latents,
        labels,
    ):
        batch_size = latents.shape[0]

        latents = latents.detach()  # no gradient to generator

        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            [batch_size, 1, 1, 1],
            device=latents.device,
            dtype=torch.long,
        )
        timestep_sigma = self.karras_sigmas[timesteps]

        noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise
        fake_x0_pred = self.fake_unet(noisy_latents, timestep_sigma, labels)

        weights = timestep_sigma**-2 + 1.0 / self.sigma_data**2

        target = latents
        loss_fake = torch.mean(weights * (fake_x0_pred - target) ** 2)

        loss_dict = {"loss_fake_mean": loss_fake}

        fake_log_dict = {
            "faketrain_latents": latents.detach(),
            "faketrain_noisy_latents": noisy_latents.detach(),
            "faketrain_x0_pred": fake_x0_pred.detach(),
        }
        return loss_dict, fake_log_dict

    def compute_loss_interpolated(
        self, real_images, fake_images, real_labels, fake_labels, step=None
    ):
        batch_size = real_images.shape[0]

        fake_images = fake_images.detach()

        noise = torch.randn_like(fake_images)

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            [batch_size, 1, 1, 1],
            device=fake_images.device,
            dtype=torch.long,
        )
        timestep_sigma = self.karras_sigmas[timesteps]

        alphas = (
            self.guidance_alpha_sampler(batch_size, step=step)
            .reshape(-1, 1, 1, 1)
            .to(fake_images.device)
        )

        noisy_fake_images = fake_images + timestep_sigma.reshape(-1, 1, 1, 1) * noise
        noisy_real_images = real_images + timestep_sigma.reshape(-1, 1, 1, 1) * noise

        pred_fake_images = self.interpolated_model(
            noisy_fake_images, timestep_sigma, alphas, fake_labels
        )
        pred_real_images = self.interpolated_model(
            noisy_real_images, timestep_sigma, alphas, real_labels
        )

        weights = timestep_sigma**-2 + 1.0 / self.sigma_data**2

        with torch.no_grad():
            fake_targets = self.fake_unet(
                noisy_fake_images, timestep_sigma, fake_labels
            )
            real_targets = self.real_unet(
                noisy_real_images, timestep_sigma, real_labels
            )

        loss_fake = torch.mean(
            weights * (1 - alphas) * (pred_fake_images - fake_targets) ** 2
        )
        loss_real = torch.mean(
            weights * alphas * (pred_real_images - real_targets) ** 2
        )

        loss_dict = {
            "loss_interpolated_fake_mean": loss_fake,
            "loss_interpolated_real_mean": loss_real,
        }

        log_dict = {
            "interpolatedtrain_real_images": real_images.detach(),
            "interpolatedtrain_fake_images": fake_images.detach(),
            "interpolatedtrain_noisy_real_images": noisy_real_images.detach(),
            "interpolatedtrain_noisy_fake_images": noisy_fake_images.detach(),
            "interpolatedtrain_pred_real_images": pred_real_images.detach(),
            "interpolatedtrain_pred_fake_images": pred_fake_images.detach(),
            "interpolatedtrain_timesteps": timesteps.detach(),
            "interpolatedtrain_alphas": alphas.detach(),
        }
        return loss_dict, log_dict

    def generator_forward(self, image, labels, step=None):
        loss_dict, log_dict = self.compute_distribution_matching_loss(
            image, labels, step=step
        )
        return loss_dict, log_dict

    def guidance_forward(self, image, labels, real_train_dict=None, step=None):
        loss_dict = {}
        log_dict = {}

        fake_dict, fake_log_dict = self.compute_loss_fake(image, labels)
        loss_dict.update(fake_dict)
        log_dict.update(fake_log_dict)

        interpolated_dict, interpolated_log_dict = self.compute_loss_interpolated(
            real_images=real_train_dict["real_image"],
            fake_images=image,
            real_labels=real_train_dict["real_label"],
            fake_labels=labels,
            step=step,
        )
        loss_dict.update(interpolated_dict)
        log_dict.update(interpolated_log_dict)

        return loss_dict, log_dict

    def forward(
        self,
        generator_turn=False,
        guidance_turn=False,
        generator_data_dict=None,
        guidance_data_dict=None,
        step=None,
    ):
        if generator_turn:
            loss_dict, log_dict = self.generator_forward(
                image=generator_data_dict["image"],
                labels=generator_data_dict["label"],
                step=step,
            )
        elif guidance_turn:
            loss_dict, log_dict = self.guidance_forward(
                image=guidance_data_dict["image"],
                labels=guidance_data_dict["label"],
                real_train_dict=guidance_data_dict["real_train_dict"],
                step=step,
            )
        else:
            raise NotImplementedError

        return loss_dict, log_dict
