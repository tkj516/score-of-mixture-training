from third_party.edm.training.networks import EDMPrecond
from main.sjsd.networks import SkewedEDMPrecond


def get_imagenet_edm_config():
    return dict(
        augment_dim=0,
        model_channels=192,
        channel_mult=[1, 2, 3, 4],
        channel_mult_emb=4,
        attn_resolutions=[32, 16, 8],
        dropout=0.0,
        label_dropout=0,
    )


def get_cifar10_edm_config():
    return dict(
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=128,
        channel_mult=[2, 2, 2],
        dropout=0.0,
        label_dropout=0,
    )


def get_cifar10_skewed_edm_config():
    return dict(
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=128,
        channel_mult=[2, 2, 2],
        dropout=0.13,
        label_dropout=0,
    )
    
    
def get_mnist_edm_config():
    return dict(
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=16,
        channel_mult=[1, 2, 3, 4],
        attn_resolutions=[0],
        dropout=0.0,
        label_dropout=0,
        augment_dim=9,
        num_blocks=1,
    )


def get_edm_network(args):
    if args.dataset_name == "imagenet":
        unet = EDMPrecond(
            img_resolution=args.resolution,
            img_channels=3,
            label_dim=args.label_dim,
            use_fp16=args.use_fp16,
            sigma_min=0,
            sigma_max=float("inf"),
            sigma_data=args.sigma_data,
            model_type="DhariwalUNet",
            **get_imagenet_edm_config(),
        )
    elif args.dataset_name == "cifar10":
        unet = EDMPrecond(
            img_resolution=args.resolution,
            img_channels=3,
            label_dim=args.label_dim,
            use_fp16=args.use_fp16,
            sigma_min=0,
            sigma_max=float("inf"),
            sigma_data=args.sigma_data,
            model_type="SongUNet",
            **get_cifar10_edm_config(),
        )
    elif args.dataset_name == "mnist":
        unet = EDMPrecond(
            img_resolution=args.resolution,
            img_channels=1,
            label_dim=args.label_dim,
            use_fp16=args.use_fp16,
            sigma_min=0,
            sigma_max=float("inf"),
            sigma_data=args.sigma_data,
            model_type="SongUNet",
            **get_mnist_edm_config(),
        )
    else:
        raise NotImplementedError

    return unet

def get_skewed_network(args):
    if args.dataset_name == "imagenet":
        unet = SkewedEDMPrecond(
            img_resolution=args.resolution,
            img_channels=3,
            label_dim=args.label_dim,
            use_fp16=args.use_fp16,
            sigma_min=0,
            sigma_max=float("inf"),
            sigma_data=args.sigma_data,
            model_type="SkewedDhariwalUNet",
            **get_imagenet_edm_config(),
        )
    elif args.dataset_name == "cifar10":
        unet = SkewedEDMPrecond(
            img_resolution=args.resolution,
            img_channels=3,
            label_dim=args.label_dim,
            use_fp16=args.use_fp16,
            sigma_min=0,
            sigma_max=float("inf"),
            sigma_data=args.sigma_data,
            model_type="SkewedSongUNet",
            **get_cifar10_skewed_edm_config(),
        )
    elif args.dataset_name == "mnist":
        unet = SkewedEDMPrecond(
            img_resolution=args.resolution,
            img_channels=1,
            label_dim=args.label_dim,
            use_fp16=args.use_fp16,
            sigma_min=0,
            sigma_max=float("inf"),
            sigma_data=args.sigma_data,
            model_type="SkewedSongUNet",
            **get_mnist_edm_config(),
        )
    else:
        raise NotImplementedError

    return unet