"""Deterministic dataset loading for paired phase-logic experiments."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import ExperimentConfig


def binarize_nonzero(image: torch.Tensor) -> torch.Tensor:
    """Match the notebook preprocessing: zero stays 0, any ink becomes 1."""

    return (image > 0.0).to(torch.float32)


def _build_mnist(
    config: ExperimentConfig,
    *,
    train: bool,
    augment: bool = False,
):
    if augment:
        raise ValueError("CIFAR augmentation cannot be applied to MNIST.")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(binarize_nonzero)]
    )
    return datasets.MNIST(
        root=config.data_dir,
        train=train,
        download=True,
        transform=transform,
    )


def _build_cifar10(
    config: ExperimentConfig,
    *,
    train: bool,
    augment: bool = False,
):
    # Keep RGB values continuous here. CIFAR encoding is performed inside the
    # model so its fixed thresholds are registered in the state dict.
    transform_steps = [transforms.ToTensor()]
    if augment:
        if not train:
            raise ValueError("CIFAR test data must not be augmented.")
        transform_steps.extend(
            [
                transforms.RandomCrop(
                    32,
                    padding=4,
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
            ]
        )
        if config.cifar_random_erasing:
            transform_steps.append(
                transforms.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.12),
                    ratio=(0.5, 2.0),
                    value=0.0,
                )
            )
    transform = transforms.Compose(transform_steps)
    return datasets.CIFAR10(
        root=config.data_dir,
        train=train,
        download=True,
        transform=transform,
    )


def _build_cifar100(
    config: ExperimentConfig,
    *,
    train: bool,
    augment: bool = False,
):
    # CIFAR-100 has the same RGB image shape and uses the same preprocessing
    # and in-model encoder as CIFAR-10.
    transform_steps = [transforms.ToTensor()]
    if augment:
        if not train:
            raise ValueError("CIFAR test data must not be augmented.")
        transform_steps.extend(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
            ]
        )
        if config.cifar_random_erasing:
            transform_steps.append(
                transforms.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.12),
                    ratio=(0.5, 2.0),
                    value=0.0,
                )
            )
    return datasets.CIFAR100(
        root=config.data_dir,
        train=train,
        download=True,
        transform=transforms.Compose(transform_steps),
    )


def compute_cifar10_quantile_thresholds(
    train_set,
    *,
    num_bits: int,
) -> torch.Tensor:
    """Compute per-channel order-statistic thresholds from training pixels.

    CIFAR-10 stores its untransformed training images as uint8 NHWC data. A
    256-bin histogram gives the exact distributive/quantile thresholds without
    materializing or sorting hundreds of millions of floating-point values.
    """

    if num_bits not in (2, 4):
        raise ValueError("CIFAR quantile encoding supports 2 or 4 bits.")
    if getattr(train_set, "train", None) is False:
        raise ValueError("Thresholds must be computed from the training set only.")

    raw = np.asarray(train_set.data)
    if raw.ndim != 4 or raw.shape[-1] != 3:
        raise ValueError(
            "Expected CIFAR images with shape [N,H,W,3], "
            f"got {raw.shape}."
        )
    if raw.dtype != np.uint8:
        raise ValueError(f"Expected uint8 CIFAR storage, got {raw.dtype}.")

    thresholds = np.empty((3, num_bits), dtype=np.float32)
    for channel in range(3):
        counts = np.bincount(raw[..., channel].reshape(-1), minlength=256)
        cumulative = np.cumsum(counts)
        pixel_count = int(cumulative[-1])
        for bit in range(1, num_bits + 1):
            # Match distributive thresholding: select the sorted value at
            # floor(N * bit / (num_bits + 1)).
            rank = pixel_count * bit // (num_bits + 1)
            value = np.searchsorted(cumulative, rank, side="right")
            thresholds[channel, bit - 1] = value / 255.0

    return torch.from_numpy(thresholds)


def build_cifar10_input_thresholds(
    config: ExperimentConfig,
    train_set,
) -> torch.Tensor | None:
    """Create the fixed thresholds used by the model's CIFAR encoder."""

    if config.dataset not in {"cifar10", "cifar100"}:
        return None
    if config.input_encoding == "fixed":
        return torch.full((3, 1), 0.5, dtype=torch.float32)
    if config.input_encoding == "sinusoidal":
        return None
    return compute_cifar10_quantile_thresholds(
        train_set,
        num_bits=config.quantile_bits,
    )


DATASET_BUILDERS = {
    "mnist": _build_mnist,
    "cifar10": _build_cifar10,
    "cifar100": _build_cifar100,
}


def build_data_loaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, train-evaluation, and test loaders for a dataset."""

    try:
        dataset_builder = DATASET_BUILDERS[config.dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {config.dataset!r}") from error

    train_set = dataset_builder(
        config,
        train=True,
        augment=config.cifar_augmentation,
    )
    # Do not reuse the augmented dataset for reported training metrics.
    # Otherwise every evaluation would measure a new random crop/flip.
    train_eval_set = (
        dataset_builder(config, train=True, augment=False)
        if config.cifar_augmentation
        else train_set
    )
    test_set = dataset_builder(config, train=False, augment=False)

    # This generator is separate from the model RNG. Recreating it with the
    # same seed gives every treatment the same shuffled batch sequence.
    train_generator = torch.Generator().manual_seed(config.seed)
    train_eval_generator = torch.Generator().manual_seed(config.seed)
    pin_memory = config.device.startswith("cuda")

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    train_eval_loader = DataLoader(
        train_eval_set,
        batch_size=config.batch_size,
        shuffle=False,
        generator=train_eval_generator,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, train_eval_loader, test_loader


def build_mnist_loaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Backward-compatible MNIST loader entry point."""

    if config.dataset != "mnist":
        raise ValueError("build_mnist_loaders requires dataset='mnist'")
    return build_data_loaders(config)


def repeat_loader(loader: DataLoader):
    """Yield batches indefinitely while preserving epoch-wise reshuffling."""

    while True:
        yield from loader
