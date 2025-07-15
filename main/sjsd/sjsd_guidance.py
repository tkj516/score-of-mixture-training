import pickle
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import dnnlib
from main.sjsd.edm_network import get_edm_network, get_skewed_network
from main.sjsd.transformer_network import get_transformer_network
from main.sjsd.utils import (
    AlphaSampler,
    AlphaScheduler,
)
from third_party.k_diffusion.k_diffusion.utils import (
    rand_log_normal,
)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    # from https://github.com/crowsonkb/k-diffusion
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def get_sigma_rand_log_normal(loc=-1.2, scale=1.2, **kwargs):
    return partial(rand_log_normal, loc=loc, scale=scale)


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


def load_skewed_model(state_dict, model):
    # Only load weights of model that are in state_dict
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


class SJSDGuidance(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.accelerator = accelerator

        state_dict = load_edm_model(args)

        # Whether or not to distill a pretrained score
        # model
        self.distillation = args.distillation

        # initialize the unets
        if args.fake_score_arch == "unet":
            self.fake_unet = get_skewed_network(args)
        elif args.fake_score_arch == "transformer":
            self.fake_unet = get_transformer_network(args)
        else:
            raise NotImplementedError(
                f"Unknown fake_score_arch: {args.fake_score_arch}"
            )
        self.fake_unet.requires_grad_(True)

        # If distilling then load the pretrained model
        # for the data score and initialize the interpolated
        # score model with these weights
        if self.distillation:
            self.real_unet = get_edm_network(args)
            self.real_unet.load_state_dict(state_dict, strict=True)
            self.real_unet.requires_grad_(False)

            del self.real_unet.model.map_augment
            self.real_unet.model.map_augment = None

        if args.fake_score_arch == "unet":
            del self.fake_unet.model.map_augment
            self.fake_unet.model.map_augment = None

        # some training hyper-parameters
        self.sigma_data = args.sigma_data
        self.sigma_max = args.sigma_max
        self.sigma_min = args.sigma_min
        self.rho = args.rho

        self.gan_classifier = args.gan_classifier
        self.diffusion_gan = args.diffusion_gan
        self.diffusion_gan_max_timestep = args.diffusion_gan_max_timestep

        if self.gan_classifier:
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
                        kernel_size=1,
                        in_channels=768,
                        out_channels=1,
                        stride=1,
                        padding=0,
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
                        kernel_size=1,
                        in_channels=256,
                        out_channels=1,
                        stride=1,
                        padding=0,
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
                maximum=args.max_partitions_guidance,
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
        rep = self.fake_unet(
            noisy_latents,
            timestep_sigmas,
            torch.ones_like(timestep_sigmas) / 2,
            labels,
            return_bottleneck=True,
        ).float()
        return self.cls_pred_branch(rep)

    def compute_distribution_matching_loss(self, latents, labels, step=None):
        original_latents = latents
        batch_size = latents.shape[0]

        with torch.no_grad():
            timesteps = torch.randint(
                self.min_step,
                min(self.max_step + 1, self.num_train_timesteps),
                [batch_size, 1, 1, 1],
                device=latents.device,
                dtype=torch.long,
            )

            # Sample an alpha for each sample in the batch
            # with a low discrepancy sampler
            alphas = (
                self.generator_alpha_sampler(
                    batch_size, step=step, val=self.args.constant_alpha_val
                )
                .reshape(-1, 1, 1, 1)
                .to(latents.device)
            )

            noise = torch.randn_like(latents)
            timestep_sigma = self.karras_sigmas[timesteps]

            noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise

            fake_unet = self.fake_unet

            pred_fake_image_0 = fake_unet(
                noisy_latents, timestep_sigma, torch.zeros_like(alphas), labels
            )
            pred_fake_image_alpha = fake_unet(
                noisy_latents, timestep_sigma, alphas, labels
            )
            pred_fake_image_1 = fake_unet(
                noisy_latents, timestep_sigma, torch.ones_like(alphas), labels
            )

            p_0 = latents - pred_fake_image_0
            p_alpha = latents - pred_fake_image_alpha
            p_1 = latents - pred_fake_image_1

            weight_factor = torch.abs(p_1).mean(dim=[1, 2, 3], keepdim=True)

            norm_ratio = torch.sqrt(
                torch.mean((p_alpha - p_0) ** 2, dim=[1, 2, 3], keepdim=True)
                / torch.mean((p_1 - p_0) ** 2 + 1e-6, dim=[1, 2, 3], keepdim=True)
            )

            weight_factor = weight_factor * norm_ratio

            # alpha-JSD
            grad = (p_alpha - p_0) * weight_factor

            grad = torch.nan_to_num(grad)

        # this loss gives the grad as gradient through autodiff,
        # following https://github.com/ashawkey/stable-dreamfusion
        loss = 0.5 * F.mse_loss(
            original_latents, (original_latents - grad).detach(), reduction="mean"
        )

        loss_dict = {"loss_dm": loss}

        dm_log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach(),
            "dmtrain_pred_fake_image_0": pred_fake_image_0.detach(),
            "dmtrain_pred_fake_image_alpha": pred_fake_image_alpha.detach(),
            "dmtrain_grad": grad.detach(),
            "dmtrain_gradient_norm": torch.norm(grad).item(),
            "dmtrain_timesteps": timesteps.detach(),
            "dmtrain_alphas": alphas.detach(),
        }
        return loss_dict, dm_log_dict

    def compute_loss_fake(
        self,
        latents,
        labels,
        real_train_dict=None,
        step=None,
    ):
        batch_size = latents.shape[0]

        # Sample a timestep for each sample in the batch
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            [batch_size, 1, 1, 1],
            device=latents.device,
            dtype=torch.long,
        )
        timestep_sigma = self.karras_sigmas[timesteps]

        # Sample an alpha for each sample in the batch
        # with a low discrepancy sampler and set half
        # of the samples to 0.0
        alphas = (
            self.guidance_alpha_sampler(
                batch_size, step=step, val=self.args.constant_alpha_val
            )
            .reshape(-1, 1, 1, 1)
            .to(latents.device)
        )

        weights_alpha = torch.ones_like(alphas)
        weights_sigma = timestep_sigma**-2 + 1.0 / self.sigma_data**2
        weights = weights_alpha * weights_sigma

        # Score matching for the fake samples
        latents = latents.detach()  # no gradient to generator
        noise = torch.randn_like(latents)
        noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise
        fake_x0_pred = self.fake_unet(noisy_latents, timestep_sigma, alphas, labels)

        target = latents
        loss_fake = torch.mean(
            weights * (1 - alphas) * (fake_x0_pred - target) ** 2,
            dim=[1, 2, 3],
            keepdim=True,
        )

        # Score matching for the real samples
        latents_real = real_train_dict["real_image"]
        noise_real = torch.randn_like(latents_real)
        noisy_latents_real = (
            latents_real + timestep_sigma.reshape(-1, 1, 1, 1) * noise_real
        )
        real_x0_pred = self.fake_unet(
            noisy_latents_real, timestep_sigma, alphas, real_train_dict["real_label"]
        )

        if self.distillation:
            with torch.no_grad():
                target_real = self.real_unet(
                    noisy_latents_real, timestep_sigma, real_train_dict["real_label"]
                )
        else:
            target_real = latents_real
        loss_real = torch.mean(
            weights * alphas * (real_x0_pred - target_real) ** 2,
            dim=[1, 2, 3],
            keepdim=True,
        )

        loss = loss_fake + loss_real

        loss_dict = {
            "loss_score": loss,
            "loss_score_mean": loss.mean(),
            "loss_score_real_mean": loss_real.mean(),
            "loss_score_fake_mean": loss_fake.mean(),
        }

        fake_log_dict = {
            "faketrain_latents": latents.detach(),
            "faketrain_noisy_latents": noisy_latents.detach(),
            "faketrain_latents_real": latents_real.detach(),
            "faketrain_noisy_latents_real": noisy_latents_real.detach(),
            "faketrain_fake_x0_pred": fake_x0_pred.detach(),
            "faketrain_real_x0_pred": real_x0_pred.detach(),
            "faketrain_timesteps": timestep_sigma.detach(),
            "faketrain_alphas": alphas.detach(),
        }
        return loss_dict, fake_log_dict

    def compute_cls_logits(self, image, alpha, label):
        t = torch.randint(
            0,
            self.num_train_timesteps,
            [image.shape[0]],
            device=image.device,
            dtype=torch.long,
        )
        t_sigma = self.karras_sigmas[t]
        x_t = image + t_sigma.reshape(-1, 1, 1, 1) * torch.randn_like(image)

        return self.compute_density_ratio(x_t, t_sigma, alpha, label).squeeze(
            dim=[2, 3]
        )

    def compute_generator_clean_cls_loss(self, image, label, step=None):
        loss_dict = {}

        # Sample an alpha for each sample in the batch
        # with a low discrepancy sampler
        alpha = (
            self.generator_alpha_sampler(
                image.shape[0], step=step, val=self.args.constant_alpha_val
            )
            .reshape(-1, 1)
            .to(image.device)
        )

        pred_realism_on_fake_with_grad = self.compute_cls_logits(image, alpha, label)
        loss_dict["gen_cls_loss"] = F.softplus(-pred_realism_on_fake_with_grad).mean()
        return loss_dict

    def compute_guidance_clean_cls_loss(self, x_real, x_fake, label, step=None):
        # Sample an alpha for each sample in the batch
        # with a low discrepancy sampler
        alpha = (
            self.generator_alpha_sampler(
                x_fake.shape[0], step=step, val=self.args.constant_alpha_val
            )
            .reshape(-1, 1)
            .to(x_fake.device)
        )

        pred_realism_on_real = self.compute_cls_logits(
            x_real.detach(),
            alpha,
            label,
        )
        pred_realism_on_fake = self.compute_cls_logits(
            x_fake.detach(),
            alpha,
            label,
        )

        classification_loss = torch.mean(
            (1 - alpha) * F.softplus(pred_realism_on_fake)
        ) + torch.mean(alpha * F.softplus(-pred_realism_on_real))

        log_dict = {
            "pred_realism_on_real": torch.sigmoid(pred_realism_on_real)
            .squeeze(dim=1)
            .detach(),
            "pred_realism_on_fake": torch.sigmoid(pred_realism_on_fake)
            .squeeze(dim=1)
            .detach(),
        }

        loss_dict = {"guidance_cls_loss": classification_loss}
        return loss_dict, log_dict

    def generator_forward(self, image, labels, step=None):
        loss_dict = {}
        log_dict = {}

        dm_dict, dm_log_dict = self.compute_distribution_matching_loss(
            image, labels, step=step
        )

        loss_dict.update(dm_dict)
        log_dict.update(dm_log_dict)

        if self.gan_classifier:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(image, labels)
            loss_dict.update(clean_cls_loss_dict)

        return loss_dict, log_dict

    def guidance_forward(self, image, labels, real_train_dict=None, step=None):
        fake_dict, fake_log_dict = self.compute_loss_fake(
            image,
            labels,
            real_train_dict,
            step=step,
        )

        loss_dict = fake_dict
        log_dict = fake_log_dict

        if self.gan_classifier or self.instant:
            clean_cls_loss_dict, clean_cls_log_dict = (
                self.compute_guidance_clean_cls_loss(
                    real_image=real_train_dict["real_image"],
                    fake_image=image,
                    real_label=real_train_dict["real_label"],
                    fake_label=labels,
                )
            )
            loss_dict.update(clean_cls_loss_dict)
            log_dict.update(clean_cls_log_dict)
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
