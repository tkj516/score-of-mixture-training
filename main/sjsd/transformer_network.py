from einops import rearrange
import torch
import torch.nn as nn

from third_party.k_diffusion.k_diffusion import layers
from third_party.k_diffusion.k_diffusion.models.axial_rope import make_axial_pos
from third_party.k_diffusion.k_diffusion.models.image_transformer_v2 import (
    GlobalAttentionSpec,
    GlobalTransformerLayer,
    Level,
    LevelSpec,
    Linear,
    MappingNetwork,
    MappingSpec,
    NeighborhoodAttentionSpec,
    NeighborhoodTransformerLayer,
    NoAttentionSpec,
    NoAttentionTransformerLayer,
    RMSNorm,
    ShiftedWindowAttentionSpec,
    ShiftedWindowTransformerLayer,
    TokenMerge,
    TokenSplit,
    TokenSplitWithoutSkip,
    downscale_pos,
    filter_params,
    tag_module,
)


class SkewedImageTransformerDenoiserModelV2(nn.Module):
    def __init__(
        self,
        levels,
        mapping,
        in_channels,
        out_channels,
        patch_size,
        num_classes=0,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)

        self.input_alpha_emb = layers.FourierFeatures(1, levels[0].width)
        self.input_alpha_in_proj = Linear(levels[0].width, levels[0].width, bias=False)

        self.time_emb = layers.FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.alpha_emb = layers.FourierFeatures(1, mapping.width)
        self.alpha_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.class_emb = (
            nn.Embedding(num_classes, mapping.width) if num_classes else None
        )
        self.mapping = tag_module(
            MappingNetwork(
                mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout
            ),
            "mapping",
        )

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):

                def layer_factory(_):
                    return GlobalTransformerLayer(
                        spec.width,
                        spec.d_ff,
                        spec.self_attn.d_head,
                        mapping.width,
                        dropout=spec.dropout,
                    )
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):

                def layer_factory(_):
                    return NeighborhoodTransformerLayer(
                        spec.width,
                        spec.d_ff,
                        spec.self_attn.d_head,
                        mapping.width,
                        spec.self_attn.kernel_size,
                        dropout=spec.dropout,
                    )
            elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):

                def layer_factory(i):
                    return ShiftedWindowTransformerLayer(
                        spec.width,
                        spec.d_ff,
                        spec.self_attn.d_head,
                        mapping.width,
                        spec.self_attn.window_size,
                        i,
                        dropout=spec.dropout,
                    )
            elif isinstance(spec.self_attn, NoAttentionSpec):

                def layer_factory(_):
                    return NoAttentionTransformerLayer(
                        spec.width, spec.d_ff, mapping.width, dropout=spec.dropout
                    )
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.down_levels.append(
                    Level([layer_factory(i) for i in range(spec.depth)])
                )
                self.up_levels.append(
                    Level([layer_factory(i + spec.depth) for i in range(spec.depth)])
                )
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

        self.merges = nn.ModuleList(
            [
                TokenMerge(spec_1.width, spec_2.width)
                for spec_1, spec_2 in zip(levels[:-1], levels[1:])
            ]
        )
        self.splits = nn.ModuleList(
            [
                TokenSplit(spec_2.width, spec_1.width)
                for spec_1, spec_2 in zip(levels[:-1], levels[1:])
            ]
        )

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(
            levels[0].width, out_channels, patch_size
        )
        nn.init.zeros_(self.patch_out.proj.weight)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(
            lambda tags: "wd" not in tags and "mapping" not in tags, self
        )
        mapping_wd = filter_params(
            lambda tags: "wd" in tags and "mapping" in tags, self
        )
        mapping_no_wd = filter_params(
            lambda tags: "wd" not in tags and "mapping" in tags, self
        )
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {
                "params": list(mapping_no_wd),
                "lr": base_lr * mapping_lr_scale,
                "weight_decay": 0.0,
            },
        ]
        return groups

    def forward(self, x, sigma, alpha, class_cond=None, return_bottleneck=False):
        # Patching
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        # TODO: pixel aspect ratio for nonsquare patches
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(
            x.shape[-3], x.shape[-2], 2
        )

        # Mapping network
        if class_cond is None and self.class_emb is not None:
            raise ValueError("class_cond must be specified if num_classes > 0")

        c_noise = torch.log(sigma) / 4
        c_alpha = alpha - 0.5

        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))

        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0

        alpha_emb = self.alpha_in_proj(self.alpha_emb(c_alpha[..., None]))

        cond = self.mapping(time_emb + class_emb + alpha_emb)

        # Add alpha embedding to the input
        input_alpha_emb = self.input_alpha_in_proj(
            self.input_alpha_emb(c_alpha[..., None])
        )
        x = x + rearrange(input_alpha_emb, "n d -> n 1 1 d")

        # Hourglass transformer
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)
        
        if return_bottleneck:
            return x

        for up_level, split, skip, pos in reversed(
            list(zip(self.up_levels, self.splits, skips, poses))
        ):
            x = split(x, skip)
            x = up_level(x, pos, cond)

        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)

        return x


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = SkewedImageTransformerDenoiserModelV2(
            in_channels=img_channels,
            out_channels=img_channels,
            num_classes=label_dim,
            **model_kwargs,
        )

    def param_groups(self, *args, **kwargs):
        return self.model.param_groups(*args, **kwargs)

    def forward(self, x, sigma, alpha, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros((1,), device=x.device, dtype=torch.long)
            if class_labels is None
            else torch.argmax(class_labels, dim=-1)
        )
        dtype = (
            torch.bfloat16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.use_fp16 and not force_fp32,
        ):
            F_x = self.model(
                (c_in * x).to(dtype),
                sigma.flatten(),
                alpha.flatten(),
                class_cond=class_labels,
                **model_kwargs,
            )
        assert F_x.dtype == dtype

        if model_kwargs.get("return_bottleneck", False):
            return F_x.permute(0, 3, 1, 2).contiguous()

        # import pdb; pdb.set_trace()
        D_x = c_skip * x[:, :3] + c_out * F_x.to(torch.float32)

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def get_cifar10_transformer_config():
    return dict(
        patch_size=[2, 2],
        levels=[
            LevelSpec(
                depth=4,
                width=512,
                d_ff=1536,
                self_attn=GlobalAttentionSpec(d_head=64),
                dropout=0.0,
            ),
            LevelSpec(
                depth=6,
                width=768,
                d_ff=2304,
                self_attn=GlobalAttentionSpec(d_head=64),
                dropout=0.0,
            ),
        ],
        mapping=MappingSpec(depth=2, width=512, d_ff=1536, dropout=0.0),
    )


def get_imagenet_transformer_config():
    return dict(
        patch_size=[2, 2],
        levels=[
            LevelSpec(
                depth=2,
                width=768,
                d_ff=2304,
                self_attn=NeighborhoodAttentionSpec(d_head=64, kernel_size=7),
                dropout=0.0,
            ),
            LevelSpec(
                depth=11,
                width=1536,
                d_ff=4608,
                self_attn=GlobalAttentionSpec(d_head=64),
                dropout=0.0,
            ),
        ],
        mapping=MappingSpec(depth=2, width=768, d_ff=2304, dropout=0.0),
    )


def get_transformer_network(args):
    if args.dataset_name == "imagenet":
        transformer = EDMPrecond(
            img_channels=args.channels,
            label_dim=args.label_dim,
            use_fp16=args.use_fp16,
            sigma_min=0,
            sigma_max=float("inf"),
            sigma_data=args.sigma_data,
            **get_imagenet_transformer_config(),
        )
    elif args.dataset_name == "cifar10":
        transformer = EDMPrecond(
            img_channels=args.channels,
            label_dim=args.label_dim,
            use_fp16=args.use_fp16,
            sigma_min=0,
            sigma_max=float("inf"),
            sigma_data=args.sigma_data,
            **get_cifar10_transformer_config(),
        )
    else:
        raise NotImplementedError

    return transformer
