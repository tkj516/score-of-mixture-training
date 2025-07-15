from main.sjsd.edm_network import get_cifar10_edm_config
from third_party.edm.training.networks import EDMPrecond
from main.edm.edm_network import get_imagenet_edm_config
from accelerate.utils import ProjectConfiguration
from accelerate.utils import set_seed
from accelerate import Accelerator
import numpy as np
import argparse
import wandb
import torch
import time
import os


def get_imagenet_config():
    base_config = {
        "img_resolution": 64,
        "img_channels": 3,
        "label_dim": 1000,
        "use_fp16": False,
        "sigma_min": 0,
        "sigma_max": float("inf"),
        "sigma_data": 0.5,
        "model_type": "DhariwalUNet",
    }
    base_config.update(get_imagenet_edm_config())
    return base_config


def get_cifar10_config():
    base_config = {
        "img_resolution": 32,
        "img_channels": 3,
        "label_dim": 0,
        "use_fp16": False,
        "sigma_min": 0,
        "sigma_max": float("inf"),
        "sigma_data": 0.5,
        "model_type": "SongUNet",
    }
    base_config.update(get_cifar10_edm_config())
    return base_config


def create_generator(checkpoint_path, base_model=None, dataset_name="imagenet"):
    if base_model is None:
        if dataset_name == "cifar10":
            base_config = get_cifar10_config()
        elif dataset_name == "imagenet":
            base_config = get_imagenet_config()
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")
        generator = EDMPrecond(**base_config)
        del generator.model.map_augment
        generator.model.map_augment = None
    else:
        generator = base_model

    while True:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            break
        except:
            print(f"fail to load checkpoint {checkpoint_path}")
            time.sleep(1)

    print(generator.load_state_dict(state_dict, strict=True))

    return generator


@torch.no_grad()
def sample(accelerator, current_model, args, model_index):
    timesteps = torch.ones(
        args.eval_batch_size, device=accelerator.device, dtype=torch.long
    )
    current_model.eval()
    all_images = []
    all_images_tensor = []

    current_index = 0

    all_labels = (
        torch.arange(
            0, args.total_eval_samples * 2, device=accelerator.device, dtype=torch.long
        )
        % args.label_dim
    )

    set_seed(args.seed + accelerator.process_index)

    while (
        len(all_images_tensor) * args.eval_batch_size * accelerator.num_processes
        < args.total_eval_samples
    ):
        noise = torch.randn(
            args.eval_batch_size,
            3,
            args.resolution,
            args.resolution,
            device=accelerator.device,
        )

        imagenet_classes = [
            ("golden retriever", 207),
            ("clownfish", 397),  # Added fish
            ("grand piano", 579),
            ("goblet", 812),
            ("snake", 60),
            ("chicken", 8),  # Added chicken
            ("owl", 24),
            ("African elephant", 385),
            ("kingfisher", 22),
            ("rooster", 7),  # Added rooster
            ("koala", 105),
            ("harp", 579),
            ("parrot", 87),  # Added another bird
            ("cello", 486),
            ("goldfish", 2),  # Added another fish
            ("espresso", 967),
            ("tarantula", 72),
            # ("hotdog", 934),
            ("stingray", 4),  # Added another aquatic animal
            # ("binoculars", 487)
        ]

        imagenet_classes = np.array([x[1] for x in imagenet_classes])

        if args.label_dim > 0:
            # random_labels = all_labels[current_index:current_index+args.eval_batch_size]
            # one_hot_labels = torch.eye(args.label_dim, device=accelerator.device)[
            #     random_labels
            # ]
            # Use the same label for all images
            # one_hot_labels = torch.eye(args.label_dim, device=accelerator.device)[
            #     all_labels[current_index * 2]
            # ].unsqueeze(0).repeat(args.eval_batch_size, 1)
            # Choose a random label from imagenet_classes and apply it to all images
            random_label = np.random.choice(imagenet_classes)
            one_hot_labels = (
                torch.eye(args.label_dim, device=accelerator.device)[random_label]
                .unsqueeze(0)
                .repeat(args.eval_batch_size, 1)
            )
        else:
            one_hot_labels = None

        current_index += args.eval_batch_size

        eval_images = current_model(
            noise * args.conditioning_sigma,
            timesteps * args.conditioning_sigma,
            one_hot_labels,
        )
        eval_images = (
            ((eval_images + 1.0) * 127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
        )
        eval_images = eval_images.contiguous()

        gathered_images = accelerator.gather(eval_images)

        all_images.append(gathered_images.cpu().numpy())
        all_images_tensor.append(gathered_images.cpu())

    if accelerator.is_main_process:
        print("all_images len ", len(torch.cat(all_images_tensor, dim=0)))

    all_images = np.concatenate(all_images, axis=0)[: args.total_eval_samples]
    all_images_tensor = torch.cat(all_images_tensor, dim=0)[: args.total_eval_samples]

    if accelerator.is_main_process:
        # Uncomment if you need to save the images
        # np.savez(os.path.join(args.folder, f"eval_image_model_{model_index:06d}.npz"), all_images)
        # raise
        grid_size = int(args.test_visual_batch_size ** (1 / 2))
        eval_images_grid = all_images[: grid_size * grid_size].reshape(
            grid_size, grid_size, args.resolution, args.resolution, 3
        )
        eval_images_grid = np.swapaxes(eval_images_grid, 1, 2).reshape(
            grid_size * args.resolution, grid_size * args.resolution, 3
        )

        data_dict = {"generated_image_grid": wandb.Image(eval_images_grid)}

        data_dict["image_mean"] = all_images_tensor.float().mean().item()
        data_dict["image_std"] = all_images_tensor.float().std().item()

        wandb.log(data_dict, step=model_index)

    accelerator.wait_for_everyone()
    return all_images_tensor


@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--total_eval_samples", type=int, default=400)
    parser.add_argument("--label_dim", type=int, default=10)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--test_visual_batch_size", type=int, default=400)
    parser.add_argument("--max_batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    # initialize accelerator
    accelerator_project_config = ProjectConfiguration(
        logging_dir="/home/tejasj/data/alpha-skew-jsd/runs/samples"
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    print(accelerator.state)

    # assert accelerator.num_processes == 1, "currently multi-gpu inference generates images with biased class distribution and leads to much worse FID"

    # initialize wandb
    if accelerator.is_main_process:
        wandb.login(
            host="https://api.wandb.ai/",
            key=os.environ.get("WANDB_API_KEY"),
        )
        run = wandb.init(
            config=args,
            dir="/home/tejasj/data/alpha-skew-jsd/runs/samples",
            **{
                "mode": "online",
                "entity": args.wandb_entity,
                "project": args.wandb_project,
            },
        )
        wandb.run.name = args.wandb_name
        print(f"wandb run dir: {run.dir}")

    generator = None

    checkpoint = args.checkpoint

    model_index = int(checkpoint.replace("/", "").split("_")[-1])

    generator = create_generator(
        os.path.join(checkpoint, "pytorch_model.bin"),
        base_model=generator,
        dataset_name=args.dataset_name,
    )
    generator = generator.to(accelerator.device)

    _ = sample(accelerator, generator, args, model_index)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    evaluate()
