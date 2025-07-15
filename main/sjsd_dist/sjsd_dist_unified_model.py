# A single unified model that wraps both the generator and discriminator
import torch
from torch import nn

from main.sjsd_dist.edm_network import get_edm_network
from main.sjsd_dist.sjsd_dist_guidance import SJSDDistGuidance, load_edm_model


class SJSDDistUniModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.guidance_model = SJSDDistGuidance(args, accelerator)

        self.guidance_min_step = self.guidance_model.min_step
        self.guidance_max_step = self.guidance_model.max_step

        self.feedforward_model = get_edm_network(args)
        self.feedforward_model.load_state_dict(load_edm_model(args), strict=True)
        del self.feedforward_model.model.map_augment
        self.feedforward_model.model.map_augment = None

        self.feedforward_model.requires_grad_(True)

        self.accelerator = accelerator
        self.num_train_timesteps = args.num_train_timesteps

    def forward(
        self,
        scaled_noisy_image,
        timestep_sigma,
        labels,
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
                    generated_image = self.feedforward_model(
                        scaled_noisy_image, timestep_sigma, labels
                    )
            else:
                generated_image = self.feedforward_model(
                    scaled_noisy_image, timestep_sigma, labels
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
                "label": labels,
                "real_train_dict": real_train_dict,
            }

        elif guidance_turn:
            assert guidance_data_dict is not None
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict,
                step=step,
            )

        return loss_dict, log_dict
