# A single unified model that wraps both the generator and score model
from main.sjsd.edm_network import get_edm_network
from main.sjsd.sjsd_guidance import SJSDGuidance, load_edm_model
from torch import nn
import torch
import copy



class SJSDUniModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.args = args

        self.conditioning_sigma = args.conditioning_sigma
        self.batch_size = args.batch_size
        self.resolution = args.resolution
        self.channels = args.channels
        self.multistep_generator = args.multistep_generator

        self.guidance_model = SJSDGuidance(args, accelerator)

        self.guidance_min_step = self.guidance_model.min_step
        self.guidance_max_step = self.guidance_model.max_step

        if args.initialie_generator:
            self.feedforward_model = get_edm_network(args)
            self.feedforward_model.load_state_dict(load_edm_model(args), strict=True)
            del self.feedforward_model.model.map_augment
            self.feedforward_model.model.map_augment = None
        else:
            self.feedforward_model = get_edm_network(args)

        self.feedforward_model.requires_grad_(True)

        self.accelerator = accelerator
        self.num_train_timesteps = args.num_train_timesteps

        self.label_dim = args.label_dim
        self.eye_matrix = (
            torch.eye(self.label_dim, device=accelerator.device)
            if self.label_dim > 0
            else None
        )

    @property
    def karras_sigmas(self):
        return self.guidance_model.module.karras_sigmas

    def sample_scaled_noise(self):
        # Generate scaled noise based on the maximum sigma value.
        scaled_noise = (
            torch.randn(
                self.batch_size,
                self.channels,
                self.resolution,
                self.resolution,
                device=self.accelerator.device,
            )
            * self.conditioning_sigma
        )

        return scaled_noise

    def sample_label(self):
        # For conditional generation, randomly generate labels.
        if self.label_dim > 0:
            labels = torch.randint(
                low=0,
                high=self.label_dim,
                size=(self.batch_size,),
                device=self.accelerator.device,
                dtype=torch.long,
            )
            # Convert these labels to one-hot encoding.
            labels = self.eye_matrix[labels]
        else:
            labels = None

        return labels

    def generate(self, latents, labels):
        generator_log_dict = {}

        if not self.multistep_generator:
            # Set timestep sigma to a preset value for all images in the batch.
            timestep_sigma = (
                torch.ones(self.batch_size, device=self.accelerator.device)
                * self.conditioning_sigma
            )
            noisy_latents = self.sample_scaled_noise()
            labels = self.sample_label()
        else:
            with torch.no_grad():
                # Sample a target noise level
                timesteps = (
                    torch.randint(
                        1,
                        self.args.num_generator_steps + 1,
                        [self.batch_size, 1, 1, 1],
                        device=self.accelerator.device,
                        dtype=torch.long,
                    )
                    * (self.num_train_timesteps // self.args.num_generator_steps)
                    - 1
                )
                timestep_sigma = self.karras_sigmas[timesteps]

                # Compute x_t
                noisy_latents = latents + timestep_sigma.reshape(
                    -1, 1, 1, 1
                ) * torch.randn_like(latents)

                noise = self.sample_scaled_noise()
                pure_noise_mask = (timesteps.squeeze() == (self.num_train_timesteps-1))
                noisy_latents[pure_noise_mask] = noise[pure_noise_mask]

            generator_log_dict = {
                "generator_timesteps": timestep_sigma.detach(),
                "generator_noisy_latents": noisy_latents.detach(),
            }

        return (
            self.feedforward_model(noisy_latents, timestep_sigma, labels),
            labels,
            generator_log_dict,
        )

    def forward(
        self,
        real_train_dict=None,
        compute_generator_gradient=False,
        generator_turn=False,
        guidance_turn=False,
        guidance_data_dict=None,
        step=None,
    ):
        assert (generator_turn and not guidance_turn) or (
            guidance_turn and not generator_turn
        )

        if generator_turn:
            if not compute_generator_gradient:
                with torch.no_grad():
                    generated_image, labels, generator_log_dict = self.generate(
                        real_train_dict["real_image"], real_train_dict["real_label"]
                    )
            else:
                generated_image, labels, generator_log_dict = self.generate(
                    real_train_dict["real_image"], real_train_dict["real_label"]
                )

            if compute_generator_gradient:
                generator_data_dict = {
                    "image": generated_image,
                    "label": labels,
                    "real_train_dict": real_train_dict,
                }

                # as we don't need to compute gradient for guidance model
                # we disable gradient to avoid side effects (in GAN Loss computation)
                self.guidance_model.requires_grad_(False)
                loss_dict, log_dict = self.guidance_model(
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict,
                    step=step,
                )
                self.guidance_model.requires_grad_(True)
            else:
                loss_dict = {}
                log_dict = {}

            log_dict["generated_image"] = generated_image.detach()

            log_dict["guidance_data_dict"] = {
                "image": generated_image.detach(),
                "label": labels.detach() if labels is not None else None,
                "real_train_dict": real_train_dict,
            }

            log_dict = log_dict | generator_log_dict

        elif guidance_turn:
            assert guidance_data_dict is not None
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict,
                step=step,
            )

        return loss_dict, log_dict
