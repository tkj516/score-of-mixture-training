import torchvision


class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, dataset_path):
        super().__init__(
            dataset_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return {"images": image, "class_labels": label}


class MNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, dataset_path):
        super().__init__(
            dataset_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(32),
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return {"images": image, "class_labels": label}
