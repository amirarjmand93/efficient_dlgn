"""Dataset-specific CAI model and encoder defaults; training logic stays shared."""

MODEL_BY_DATASET = {
    "cifar10": "build_model_cifar10_small_compact",
    "cifar100": "build_model_cifar100_small_compact",
    "mnist": "build_model_mnist_small_compact",
    "heartbeat": "build_model_heartbeat_small_compact",
}


def adapt_config(values, dataset):
    """Apply the matching model and preprocessing when switching datasets."""
    if dataset not in MODEL_BY_DATASET:
        raise ValueError(f"Unsupported CAI dataset: {dataset}")
    values["dataset"] = dataset
    values["model_variant"] = MODEL_BY_DATASET[dataset]
    if dataset in {"cifar10", "cifar100", "heartbeat"}:
        values.update(input_encoding="hybrid", hybrid_features="lbp",
                      num_quantile_bits=2, num_sine_bits=None)
    else:
        values.update(input_encoding="fixed", hybrid_features="sine",
                      num_input_bits=1, num_quantile_bits=None,
                      num_sine_bits=None)
    values["cifar_augmentation"] = dataset in {"cifar10", "cifar100"}
    values["cifar_random_erasing"] = False
