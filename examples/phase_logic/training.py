"""Training and evaluation loop for controlled phase-logic experiments."""

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig
from .compact import CandidateLogicLayer, CompactLogicController, UniformThermometerEncoder
from .hybrid_freezing import FREEZE_PROBE_FIELDS, HybridFreezeController
from .dense_freezing import DENSE_FREEZE_FIELDS, DenseFreezeController, enable_dense_group_freezing
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
from .rewiring import LossProbedRewirer, REWIRING_FIELDS
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
        compact_backend=config.compact_backend,
        hybrid_features=config.hybrid_features,
        mrlb_conv3_relational_channels=config.mrlb_conv3_relational_channels,
    )
    if config.compact_dense_freeze:
        enable_dense_group_freezing(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    layer_widths = logic_layer_width_report(model)
    encoding_report = None
    if config.dataset in {"cifar10", "cifar100"}:
        encoder = next(
            module
            for module in model.modules()
            if isinstance(module, (HybridFeatureEncoder, UniformThermometerEncoder))
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
                if encoder.num_quantile_bits or config.input_encoding == "thermometer"
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
        if config.lbp_bits:
            encoding_report.update({
                "hybrid_features": "lbp",
                "num_lbp_bits": encoder.num_lbp_bits,
                "lbp_comparisons": ["right >= left", "down >= up"],
                "lbp_radius": 1,
                "lbp_padding": "replicate",
            })
    model_report = {
        "dataset": config.dataset,
        "model_variant": config.model_variant,
        "parameter_count": parameter_count,
        "logic_layer_widths": layer_widths,
        "input_encoding": encoding_report,
    }
    if config.uses_compact_logic:
        model_report["compact"] = {
            "backend": model[1].backend,
            "candidate_count": 16,
            "base_channels": 128,
            "group_sum_tau": 40,
            "inference_boolean_nodes": sum(layer["output_wires"] for layer in layer_widths),
            "exploration_steps": config.compact_exploration_steps,
            "upstream_commit": "019677a79a1a3039e3c2c7f26689f3bcec8e22f9",
        }
    elif config.uses_classrouted_compact_head:
        heads = [layer for layer in model if isinstance(layer, CandidateLogicLayer) and not layer.is_convolution]
        model_report["compact"] = {
            "backend": heads[0].backend,
            "scope": ("CompactLogic plain six-convolution backbone and fully shared dense head"
                      if config.model_variant == "build_model_cifar10_compact_sixconv"
                      else "CompactLogic channel-concatenation skips and fully shared dense head"),
            "dense_connectivity": ["full shared input"] * len(heads),
            "concatenation_skips": list(getattr(model, "concatenation_skips", ())),
            "candidate_count": 16,
            "base_channels": 128,
            "conv_channels": [layer.out_dim for layer in model
                              if isinstance(layer, CandidateLogicLayer) and layer.is_convolution],
            "flatten_width": heads[0].impl.in_dim,
            "dense_widths": [head.out_dim for head in heads],
            "group_sum_tau": model[-1].tau,
            "inference_boolean_nodes": sum(layer["lut_node_applications"] for layer in layer_widths),
            "exploration_steps": config.compact_exploration_steps,
            "progressive_conv_freezing": config.compact_freeze_mode != "none",
            "upstream_commit": "019677a79a1a3039e3c2c7f26689f3bcec8e22f9",
        }
        if config.model_variant == "build_model_cifar10_small_mrlb_compact_head":
            model_report["compact"].update({
                "scope": "CompactLogic MRLB-v2 relational bridge and fully shared dense head",
                "conv_channels": [model[i].out_dim for i in model.backbone_conv_indices],
                "relational_bridge": model.relational_bridge_report(),
            })
    if config.compact_layer_aware_resampling:
        model_report["compact"]["candidate_resampling"] = {
            "mode": "layer_aware_v1", "preferred_fraction": 0.25,
            "uniform_fraction": 0.75, "conv1_to_3_preferred_ops": [1, 7],
            "dense2_preferred_ops": [6, 9], "other_layers": "uniform",
        }
    if config.compact_efficient_layer_aware_resampling:
        model_report["compact"]["candidate_resampling"] = {
            "mode": "efficient_layer_aware_v2",
            "conv1_to_3": {"uniform_fraction": 0.75, "preferred_fraction": 0.25,
                           "preferred_ops": [1, 7]},
            "conv4": {"uniform_fraction": 1.0},
            "dense1": {"uniform_fraction": 0.70, "identity_fraction": 0.18,
                       "identity_ops": [3, 5], "not_fraction": 0.08, "not_ops": [12, 10],
                       "constant_fraction": 0.04, "constant_ops": [15, 0]},
            "dense2": {"uniform_fraction": 0.80, "identity_fraction": 0.12,
                       "identity_ops": [3, 5], "not_fraction": 0.06, "not_ops": [12, 10],
                       "constant_fraction": 0.02, "constant_ops": [15, 0]},
        }
    if config.compact_aggressive_layer_aware_resampling:
        model_report["compact"]["candidate_resampling"] = {
            "mode": "aggressive_layer_aware_v1",
            "conv1_to_3": {"uniform_fraction": 0.75, "preferred_fraction": 0.25,
                           "preferred_ops": [1, 7]},
            "conv4": {"uniform_fraction": 1.0},
            "dense1": {"uniform_fraction": 0.50, "identity_fraction": 0.30,
                       "identity_ops": [3, 5], "not_fraction": 0.05, "not_ops": [12, 10],
                       "constant_fraction": 0.15, "constant_ops": [15, 0]},
            "dense2": {"uniform_fraction": 0.60, "identity_fraction": 0.25,
                       "identity_ops": [3, 5], "not_fraction": 0.05, "not_ops": [12, 10],
                       "constant_fraction": 0.10, "constant_ops": [15, 0]},
        }
    _write_json(run_dir / "model_report.json", model_report)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    dense_freezer = DenseFreezeController(model, config, optimizer) if config.compact_dense_freeze else None
    dense_probe_batches = repeat_loader(train_eval_loader) if dense_freezer is not None else None
    if dense_freezer is not None:
        model_report["compact"]["dense_freezing"] = {
            "mode": "dense_groups_v1", "group_fraction": config.dense_freeze_fraction,
            "after_all_convs": True, "taylor_required": False,
        }
        _write_json(run_dir / "model_report.json", model_report)
        with (run_dir / "dense_freeze_probes.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=DENSE_FREEZE_FIELDS).writeheader()
    controller_type = HybridFreezeController if config.compact_freeze_mode == "hybrid" else CompactLogicController
    compact_controller = (
        controller_type(model, config)
        if config.uses_compact_logic or config.uses_classrouted_compact_head else None
    )
    freeze_probe_batches = None
    if isinstance(compact_controller, HybridFreezeController):
        freeze_probe_batches = repeat_loader(train_eval_loader)
        with (run_dir / "freeze_probes.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FREEZE_PROBE_FIELDS).writeheader()
    if compact_controller is not None:
        with (run_dir / "compact.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                "step", "weak_rows_resampled", "strong_rows_refreshed", "frozen_conv_layers",
            ])
    rewirer = LossProbedRewirer(model, config) if config.rewire_loss_probe else None
    probe_batches = repeat_loader(train_eval_loader) if rewirer is not None else None
    if rewirer is not None:
        with (run_dir / "rewiring.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=REWIRING_FIELDS).writeheader()
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
            + (f"lbp_bits={config.lbp_bits} " if config.lbp_bits else "")
            + f"total_bits={encoding_report['num_bits']}"
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
            label = "uniform_thresholds" if config.input_encoding == "thermometer" else "quantile_thresholds"
            print(f"  {label} {formatted_thresholds}")
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
    if config.uses_compact_logic:
        print(
            f"  compact backend={model[1].backend} candidates=16 "
            f"exploration_steps={config.compact_exploration_steps} "
            f"inference_boolean_nodes={model_report['compact']['inference_boolean_nodes']} "
            "conv_channels=128,128,512,512 dense_widths=80000,80000"
        )
    elif config.uses_classrouted_compact_head:
        print(
            f"  compact_head backend={model_report['compact']['backend']} candidates=16 "
            f"exploration_steps={config.compact_exploration_steps} "
            f"inference_boolean_nodes={model_report['compact']['inference_boolean_nodes']} "
            f"conv_channels={','.join(map(str, model_report['compact']['conv_channels']))} "
            f"flatten_width={model_report['compact']['flatten_width']} "
            f"dense_widths={','.join(map(str, model_report['compact']['dense_widths']))} "
            f"dense_connectivity=fully_shared dense_engines={len(model_report['compact']['dense_widths'])} "
            f"group_sum_tau={model_report['compact']['group_sum_tau']:g} "
            f"concat_skips={';'.join(model_report['compact']['concatenation_skips']) or 'none'} "
            f"candidate_temperature=1 progressive_conv_freezing={'off' if config.compact_freeze_mode == 'none' else 'on'}"
        )
        if "relational_bridge" in model_report["compact"]:
            bridge = model_report["compact"]["relational_bridge"]
            print(f"  mrlb local=64@3x3 cross_depth=64@5x5 conv3_inputs=384 "
                  f"conv3_output_routes={bridge['conv3_output_routes']}")
    if compact_controller is not None:
        print(f"  compact_freeze_mode={config.compact_freeze_mode}")
        if config.compact_layer_aware_resampling:
            print("  candidate_resampling=layer_aware preferred=25% uniform=75% Conv1-3=AND/OR Dense2=XOR/XNOR")
        if config.compact_efficient_layer_aware_resampling:
            print("  candidate_resampling=ELAR_v2 Conv1-3=75%uniform+25%AND/OR Conv4=uniform Dense1=70%uniform+18%identity+8%NOT+4%constants Dense2=80%uniform+12%identity+6%NOT+2%constants")
        if config.compact_aggressive_layer_aware_resampling:
            print("  candidate_resampling=aggressive_layer_aware_v1 Conv1-3=75%uniform+25%AND/OR Conv4=uniform Dense1=50%uniform+30%identity+5%NOT+15%constants Dense2=60%uniform+25%identity+5%NOT+10%constants")
    if dense_freezer is not None:
        print(f"  dense_freezing=groups fraction={config.dense_freeze_fraction:g} after_all_convs=True "
              f"recovery_steps={config.dense_freeze_min_interval} "
              f"probe_batches={config.dense_freeze_probe_batches} "
              f"max_loss_increase={config.dense_freeze_max_loss_increase:g}")

    metrics_path = run_dir / "metrics.csv"
    best_discrete_accuracy = 0.0
    best_compact_step = None
    final_metrics: dict[str, Any] = {}
    running_examples = 0
    running_total = 0.0
    running_ce = 0.0
    running_phase = 0.0

    training_started = time.perf_counter()
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
        writer.writeheader()

        for step, (inputs, targets) in enumerate(repeat_loader(train_loader), start=1):
            if step > config.iterations:
                break

            temperature = temperature_at(config, step)
            learning_rate = learning_rate_at(config, step)
            phase_weight = phase_weight_at(config, step)
            if compact_controller is None:
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
                if dense_freezer is None or not dense_freezer.fully_frozen:
                    terms.total.backward()
                    if compact_controller is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
            finally:
                set_wire_erasure_enabled(model, False)

            if compact_controller is not None:
                compact_controller.step(step, optimizer)
                if (isinstance(compact_controller, HybridFreezeController)
                        and compact_controller.pending_probe is not None):
                    freeze_inputs, freeze_targets = next(freeze_probe_batches)
                    additional_batches = []
                    for _ in range(config.compact_freeze_probe_batches - 1):
                        batch_inputs, batch_targets = next(freeze_probe_batches)
                        additional_batches.append((batch_inputs.to(device, non_blocking=True),
                                                   batch_targets.to(device, non_blocking=True)))
                    event = compact_controller.probe_and_freeze(
                        model, freeze_inputs.to(device, non_blocking=True),
                        freeze_targets.to(device, non_blocking=True), step,
                        additional_batches=additional_batches,
                    )
                    with (run_dir / "freeze_probes.csv").open("a", newline="", encoding="utf-8") as handle:
                        csv.DictWriter(handle, fieldnames=FREEZE_PROBE_FIELDS).writerow(event)
                    print(f"  freeze_probe step={step} layer={event['layer']} "
                          f"taylor={event['taylor_score']} loss_increase={event['loss_increase']} "
                          f"loss_upper_bound={event['loss_upper_bound']} batches={event['probe_batches']} "
                          f"passed={event['passed']} confirmations={event['consecutive_passes']}")

            if dense_freezer is not None:
                dense_freezer.step(step)
                if dense_freezer.pending_probe:
                    batches = []
                    for _ in range(config.dense_freeze_probe_batches):
                        batch_inputs, batch_targets = next(dense_probe_batches)
                        batches.append((batch_inputs.to(device, non_blocking=True),
                                        batch_targets.to(device, non_blocking=True)))
                    event = dense_freezer.probe_and_freeze(model, batches, step, optimizer)
                    with (run_dir / "dense_freeze_probes.csv").open("a", newline="", encoding="utf-8") as handle:
                        csv.DictWriter(handle, fieldnames=DENSE_FREEZE_FIELDS).writerow(event)
                    print(f"  dense_freeze_probe step={step} layer={event['layer']} "
                          f"group={event['group_size']} loss_upper_bound={event['loss_upper_bound']} "
                          f"passed={event['passed']} confirmations={event['consecutive_passes']} "
                          f"committed={event['committed']} frozen_rows={event['frozen_rows']}/{event['total_rows']}")

            if rewirer is not None and step >= config.rewire_warmup and step % config.rewire_interval == 0:
                probe_inputs, probe_targets = next(probe_batches)
                probe_inputs = probe_inputs.to(device, non_blocking=True)
                probe_targets = probe_targets.to(device, non_blocking=True)
                layer_probe_batches = None
                if rewirer.num_layers > 1:
                    layer_probe_batches = [(probe_inputs, probe_targets)]
                    for _ in range(rewirer.num_layers - 1):
                        batch_inputs, batch_targets = next(probe_batches)
                        layer_probe_batches.append((
                            batch_inputs.to(device, non_blocking=True),
                            batch_targets.to(device, non_blocking=True),
                        ))
                event = rewirer.maybe_rewire(
                    model, inputs, targets, step, optimizer,
                    probe_inputs=probe_inputs,
                    probe_targets=probe_targets,
                    layer_probe_batches=layer_probe_batches,
                )
                if event is not None:
                    with (run_dir / "rewiring.csv").open(
                        "a", newline="", encoding="utf-8"
                    ) as handle:
                        csv.DictWriter(handle, fieldnames=REWIRING_FIELDS).writerow(event)
                    print(
                        f"  rewire step={step} eligible={event['eligible_rows']} "
                        f"attempted={event['attempted_rows']} "
                        f"accepted={event['accepted_rows']} "
                        f"probe_seconds={event['elapsed_seconds']:.3f}"
                    )

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
            if compact_controller is not None:
                compact_status = compact_controller.summary()
                with (run_dir / "compact.csv").open("a", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow([
                        step, compact_status["weak_rows_resampled"],
                        compact_status["strong_rows_refreshed"],
                        len(compact_status["frozen_conv_layers"]),
                    ])
                print(
                    f"  compact step={step} weak_resampled={compact_status['weak_rows_resampled']} "
                    f"strong_refreshed={compact_status['strong_rows_refreshed']} "
                    f"frozen_convs={len(compact_status['frozen_conv_layers'])}/{len(compact_controller.convs)}"
                )
                if isinstance(compact_controller, HybridFreezeController):
                    print(f"  hybrid next_layer={compact_status['next_freeze_layer']} "
                          f"diagnostic_stable_steps={compact_status['stable_steps']} "
                          f"winner_change={compact_status['winner_change']} "
                          f"winner_change_window={compact_status['winner_change_window']} "
                          f"entropy_drift_ema={compact_status['entropy_drift_ema']}")
                if dense_freezer is not None:
                    dense_status = dense_freezer.summary()
                    print(f"  dense_freeze next_layer={dense_status['next_layer']} "
                          f"frozen_rows={dense_status['layers']}")
                if config.save_model and (best_compact_step is None or discrete["acc"] > best_discrete_accuracy):
                    best_compact_step = step
                    torch.save({
                        "step": step, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "controller_state": compact_controller.state_dict(),
                        **({"dense_freezer_state": dense_freezer.state_dict()} if dense_freezer is not None else {}),
                        "config": config.to_dict(), "metrics": row,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    }, run_dir / "best_model.pt")
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
        "training_elapsed_seconds": time.perf_counter() - training_started,
    }
    if rewirer is not None:
        summary["rewiring"] = rewirer.summary()
    if compact_controller is not None:
        summary["compact"] = compact_controller.summary()
    if dense_freezer is not None:
        summary["dense_freezing"] = dense_freezer.summary()
    _write_json(run_dir / "summary.json", summary)
    if config.save_model:
        torch.save(model.state_dict(), run_dir / "final_model.pt")
    return summary
