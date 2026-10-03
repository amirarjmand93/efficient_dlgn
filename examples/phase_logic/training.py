"""Training and evaluation loop for controlled phase-logic experiments."""

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig
from .data import (
    build_cifar10_input_thresholds,
    build_data_loaders,
    repeat_loader,
)
from .losses import PhaseLogicLoss, margin_phase_statistics
from .model import (
    HybridFeatureEncoder,
    build_model_for_dataset,
    logic_layer_width_report,
)
from .schedules import (
    cosine_decay,
    linear_after_warmup,
    set_parametrization_temperature,
)
from .wire_erasing import set_wire_erasure_enabled


METRIC_FIELDS = [
    "step",
    "learning_rate",
    "temperature",
    "phase_weight",
    "train_loss",
    "train_ce",
    "train_phase",
    "train_loss_relaxed",
    "train_acc_relaxed",
    "test_loss_discrete",
    "test_acc_discrete",
    "test_loss_relaxed",
    "test_acc_relaxed",
    "accuracy_gap",
    "current_phase_energy",
    "fraction_outside_margin",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def temperature_at(config: ExperimentConfig, step: int) -> float:
    if not config.uses_temperature_annealing:
        return config.start_temperature
    return linear_after_warmup(
        step=step,
        total_steps=config.iterations,
        warmup_steps=config.temperature_warmup,
        start=config.start_temperature,
        end=config.end_temperature,
    )


def learning_rate_at(config: ExperimentConfig, step: int) -> float:
    """Return the configured learning rate for one-based training ``step``."""

    if config.lr_schedule == "constant":
        return config.learning_rate
    return cosine_decay(
        step=step,
        total_steps=config.iterations,
        start=config.learning_rate,
        end=config.end_learning_rate,
    )


def phase_weight_at(config: ExperimentConfig, step: int) -> float:
    if not config.uses_phase_loss:
        return 0.0
    return linear_after_warmup(
        step=step,
        total_steps=config.iterations,
        warmup_steps=config.phase_warmup,
        start=0.0,
        end=config.phase_lambda_max,
    )


def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    relaxed: bool,
) -> dict[str, float]:
    """Evaluate CE and accuracy, weighted by the exact number of examples."""

    set_wire_erasure_enabled(model, False)
    original_training = model.training
    model.train(relaxed)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    try:
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                predictions = model(inputs)
                total_loss += criterion(predictions, targets).item()
                total_correct += (
                    predictions.argmax(dim=-1) == targets
                ).sum().item()
                total_examples += targets.numel()
    finally:
        set_wire_erasure_enabled(model, False)
        model.train(original_training)

    return {
        "loss": total_loss / total_examples,
        "acc": total_correct / total_examples,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Train one arm and return its final reproducible summary."""

    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "config.json",
        {**config.to_dict(), "resolved_device": str(device)},
    )

    train_loader, train_eval_loader, test_loader = build_data_loaders(config)
    input_thresholds = build_cifar10_input_thresholds(
        config,
        train_loader.dataset,
    )
    model = build_model_for_dataset(
        device,
        dataset=config.dataset,
        model_variant=config.model_variant,
        parametrization=config.parametrization,
        input_thresholds=input_thresholds,
        input_encoding=config.input_encoding,
        num_quantile_bits=config.quantile_bits,
        num_sine_bits=config.sine_bits,
        sine_sigma=config.sine_sigma,
        sine_seed=config.sine_seed,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    layer_widths = logic_layer_width_report(model)
    encoding_report = None
    if config.dataset in {"cifar10", "cifar100"}:
        encoder = next(
            module
            for module in model.modules()
            if isinstance(module, HybridFeatureEncoder)
        )
        sample_input = train_loader.dataset[0][0].unsqueeze(0).to(device)
        with torch.no_grad():
            sample_encoded = encoder(sample_input)
        encoding_report = {
            "method": config.input_encoding,
            "num_bits": encoder.num_bits,
            "num_quantile_bits": encoder.num_quantile_bits,
            "num_sine_bits": encoder.num_sine_bits,
            "thresholds": (
                encoder.thresholds.detach().cpu().tolist()
                if encoder.num_quantile_bits
                else None
            ),
            "sine_weights": (
                encoder.sine_weights.detach().cpu().tolist()
                if encoder.num_sine_bits
                else None
            ),
            "sine_phases": (
                encoder.sine_phases.detach().cpu().tolist()
                if encoder.num_sine_bits
                else None
            ),
            "sine_sigma": encoder.sigma if encoder.num_sine_bits else None,
            "sine_seed": encoder.sine_seed if encoder.num_sine_bits else None,
            "sample_input_shape": list(sample_input.shape),
            "sample_encoded_shape": list(sample_encoded.shape),
            "channel_order": encoder.channel_order,
        }
    model_report = {
        "dataset": config.dataset,
        "model_variant": config.model_variant,
        "parameter_count": parameter_count,
        "logic_layer_widths": layer_widths,
        "input_encoding": encoding_report,
    }
    _write_json(run_dir / "model_report.json", model_report)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = PhaseLogicLoss(
        config.loss_option,
        margin=config.phase_margin,
    )

    print(
        f"[{config.dataset}/{config.experiment}/{config.parametrization}] "
        f"model={config.model_variant} seed={config.seed} device={device} "
        f"parameters={parameter_count} "
        f"cifar_augmentation={config.cifar_augmentation} "
        f"lr_schedule={config.lr_schedule} "
        f"lr={config.learning_rate:g}->{config.end_learning_rate:g}"
    )
    if encoding_report is not None:
        print(
            f"  input_encoding={config.input_encoding} "
            f"quantile_bits={encoding_report['num_quantile_bits']} "
            f"sine_bits={encoding_report['num_sine_bits']} "
            f"total_bits={encoding_report['num_bits']}"
        )
        channel_names = ("R", "G", "B")
        if encoding_report["thresholds"] is not None:
            formatted_thresholds = " ".join(
                f"{channel}=["
                + ", ".join(f"{value:.6f}" for value in values)
                + "]"
                for channel, values in zip(
                    channel_names,
                    encoding_report["thresholds"],
                )
            )
            print(f"  quantile_thresholds {formatted_thresholds}")
        if encoding_report["sine_weights"] is not None:
            formatted_weights = " ".join(
                f"{channel}=["
                + ", ".join(f"{value:.6f}" for value in values)
                + "]"
                for channel, values in zip(
                    channel_names,
                    encoding_report["sine_weights"],
                )
            )
            formatted_phases = " ".join(
                f"{channel}=["
                + ", ".join(f"{value:.6f}" for value in values)
                + "]"
                for channel, values in zip(
                    channel_names,
                    encoding_report["sine_phases"],
                )
            )
            print(
                f"  sine_features sigma={encoding_report['sine_sigma']:g} "
                f"seed={encoding_report['sine_seed']} weights {formatted_weights}"
            )
            print(f"  sine_phases {formatted_phases}")
        print(
            "  encoder_shapes="
            f"{tuple(encoding_report['sample_input_shape'])} -> "
            f"{tuple(encoding_report['sample_encoded_shape'])}"
        )
    print(
        "  logic_layer_widths="
        + ", ".join(
            f"{layer['layer']}:{layer['width']}" for layer in layer_widths
        )
    )

    metrics_path = run_dir / "metrics.csv"
    best_discrete_accuracy = 0.0
    final_metrics: dict[str, Any] = {}
    running_examples = 0
    running_total = 0.0
    running_ce = 0.0
    running_phase = 0.0

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
        writer.writeheader()

        for step, (inputs, targets) in enumerate(repeat_loader(train_loader), start=1):
            if step > config.iterations:
                break

            temperature = temperature_at(config, step)
            learning_rate = learning_rate_at(config, step)
            phase_weight = phase_weight_at(config, step)
            set_parametrization_temperature(model, temperature)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            model.train()
            set_wire_erasure_enabled(model, True)
            try:
                predictions = model(inputs)
                terms = criterion(model, predictions, targets, phase_weight)

                optimizer.zero_grad(set_to_none=True)
                terms.total.backward()
                optimizer.step()
            finally:
                set_wire_erasure_enabled(model, False)

            batch_size = targets.numel()
            running_examples += batch_size
            running_total += terms.total.detach().item() * batch_size
            running_ce += terms.cross_entropy.detach().item() * batch_size
            running_phase += terms.phase.detach().item() * batch_size

            should_evaluate = (
                step % config.eval_frequency == 0 or step == config.iterations
            )
            if not should_evaluate:
                continue

            discrete = evaluate(model, test_loader, device, relaxed=False)
            relaxed = evaluate(model, test_loader, device, relaxed=True)
            train_relaxed = evaluate(
                model, train_eval_loader, device, relaxed=True
            )
            if config.parametrization == "light":
                with torch.no_grad():
                    current_phase, fraction_outside = margin_phase_statistics(
                        model, config.phase_margin
                    )
                    current_phase = current_phase.item()
                    fraction_outside = fraction_outside.item()
            else:
                current_phase = None
                fraction_outside = None

            row = {
                "step": step,
                "learning_rate": learning_rate,
                "temperature": temperature,
                "phase_weight": phase_weight,
                "train_loss": running_total / running_examples,
                "train_ce": running_ce / running_examples,
                "train_phase": running_phase / running_examples,
                "train_loss_relaxed": train_relaxed["loss"],
                "train_acc_relaxed": train_relaxed["acc"],
                "test_loss_discrete": discrete["loss"],
                "test_acc_discrete": discrete["acc"],
                "test_loss_relaxed": relaxed["loss"],
                "test_acc_relaxed": relaxed["acc"],
                "accuracy_gap": relaxed["acc"] - discrete["acc"],
                "current_phase_energy": current_phase,
                "fraction_outside_margin": fraction_outside,
            }
            writer.writerow(row)
            metrics_file.flush()
            final_metrics = row
            best_discrete_accuracy = max(
                best_discrete_accuracy, discrete["acc"]
            )

            print(
                f"  step={step:5d} lr={learning_rate:.6f} "
                f"temp={temperature:.3f} "
                f"phase_weight={phase_weight:.4f} "
                f"train_loss_relaxed={train_relaxed['loss']:.4f} "
                f"train_acc_relaxed={train_relaxed['acc']:.4f} "
                f"test_acc_discrete={discrete['acc']:.4f} "
                f"test_acc_relaxed={relaxed['acc']:.4f} "
                f"gap={row['accuracy_gap']:+.4f}"
            )

            running_examples = 0
            running_total = 0.0
            running_ce = 0.0
            running_phase = 0.0

    summary = {
        "experiment": config.experiment,
        "dataset": config.dataset,
        "model_variant": config.model_variant,
        "cifar_augmentation": config.cifar_augmentation,
        "seed": config.seed,
        "resolved_device": str(device),
        "parameter_count": parameter_count,
        "logic_layer_widths": layer_widths,
        "input_encoding": encoding_report,
        "best_test_acc_discrete": best_discrete_accuracy,
        "final": final_metrics,
    }
    _write_json(run_dir / "summary.json", summary)
    if config.save_model:
        torch.save(model.state_dict(), run_dir / "final_model.pt")
    return summary
