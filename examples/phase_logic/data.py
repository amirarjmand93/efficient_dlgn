"""Deterministic dataset loading for paired phase-logic experiments."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
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
        if config.uses_compact_logic:
            transform_steps = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=2, padding_mode="edge"),
                transforms.ToTensor(),
            ]
        else:
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


class HeartbeatDataset(Dataset):
    """Five-class MIT-BIH beat split; 187 waveform samples plus one label."""

    def __init__(self, path: Path, *, train: bool):
        if not path.is_file():
            raise FileNotFoundError(f"Heartbeat CSV not found: {path}")
        raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 188:
            raise ValueError(f"Expected heartbeat CSV with 188 columns, got {raw.shape}.")
        if not np.isfinite(raw).all():
            raise ValueError(f"Heartbeat CSV contains non-finite values: {path}")
        labels = raw[:, -1]
        if np.any(labels != np.floor(labels)) or np.any((labels < 0) | (labels > 4)):
            raise ValueError(f"Expected heartbeat class labels 0..4 in {path}.")
        self.signals = torch.from_numpy(np.ascontiguousarray(raw[:, :187])).unsqueeze(1)
        self.targets = torch.from_numpy(labels.astype(np.int64))
        self.train = train

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.signals[index], self.targets[index]


def _build_heartbeat(config: ExperimentConfig, *, train: bool, augment: bool = False):
    if augment:
        raise ValueError("CIFAR augmentation cannot be applied to heartbeat data.")
    filename = "mitbih_train.csv" if train else "mitbih_test.csv"
    return HeartbeatDataset(config.data_dir / "heartbeat" / filename, train=train)


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

    if config.dataset == "heartbeat":
        if not getattr(train_set, "train", False):
            raise ValueError("Heartbeat quantile thresholds require the training split.")
        signal = train_set.signals.numpy().reshape(-1)
        # Use only the provided training split; retain thresholds in model state.
        ranks = [len(signal) * bit // 3 for bit in (1, 2)]
        selected = np.partition(signal, ranks)[ranks]
        return torch.from_numpy(selected.copy())
    if config.dataset not in {"cifar10", "cifar100"}:
        return None
    if config.input_encoding == "fixed":
        return torch.full((3, 1), 0.5, dtype=torch.float32)
    if config.input_encoding == "thermometer":
        thresholds = torch.arange(1, config.num_input_bits + 1, dtype=torch.float32)
        return (thresholds / (config.num_input_bits + 1)).expand(3, -1).clone()
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
    "heartbeat": _build_heartbeat,
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
        drop_last=config.uses_compact_logic,
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


def build_progressive_infusion_loaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, object]:
    """Build train/train-eval/validation/test loaders with a fixed 10% holdout.

    This opt-in split exists only for adaptive infusion decisions, avoiding test
    data and augmented samples. The final item is the full training dataset used
    to calculate fixed encoder thresholds exactly as in ordinary training.
    """
    try:
        dataset_builder = DATASET_BUILDERS[config.dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {config.dataset!r}") from error
    augmented_train = dataset_builder(
        config, train=True, augment=config.cifar_augmentation,
    )
    evaluation_train = (
        dataset_builder(config, train=True, augment=False)
        if config.cifar_augmentation else augmented_train
    )
    test_set = dataset_builder(config, train=False, augment=False)
    count = len(augmented_train)
    validation_count = max(1, int(round(0.10 * count)))
    if validation_count >= count:
        raise ValueError("Progressive infusion validation split leaves no training examples.")
    split_generator = torch.Generator().manual_seed(config.seed)
    permutation = torch.randperm(count, generator=split_generator).tolist()
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    train_set = Subset(augmented_train, training_indices)
    train_eval_set = Subset(evaluation_train, training_indices)
    validation_set = Subset(evaluation_train, validation_indices)
    pin_memory = config.device.startswith("cuda")
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=config.num_workers, pin_memory=pin_memory,
        drop_last=config.uses_compact_logic,
    )
    train_eval_loader = DataLoader(
        train_eval_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=pin_memory,
    )
    return train_loader, train_eval_loader, validation_loader, test_loader, evaluation_train


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
