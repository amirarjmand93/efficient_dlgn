"""Train a CompactLogic candidate model with continuous adaptive infusion.

This is a separate experiment. It starts from a reference run's configuration,
with fresh parameters; it does not fine-tune the reference checkpoint.
"""

import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from ...config import ExperimentConfig
from ...compact import CompactLogicController
from ...data import build_cifar10_input_thresholds, build_progressive_infusion_loaders, repeat_loader
from ...hybrid_freezing import FREEZE_PROBE_FIELDS, HybridFreezeController
from ...model import build_model_for_dataset, logic_layer_width_report
from ...training import evaluate, resolve_device, seed_everything
from .checkpoint import load_safe_checkpoint, save_safe_checkpoint
from .controller import ContinuousAdaptiveInfusion
from .gate_metrics import GateStabilityTracker, plot_gate_dynamics
from .policy import ContinuousCandidatePolicy
from .dataset_adapter import adapt_config


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
REFERENCE_CONFIG = HERE / "reference_config.json"
BASE_METRIC_FIELDS = (
    "step", "train_loss", "train_acc", "validation_loss_discrete",
    "validation_acc_discrete", "test_loss_discrete", "test_acc_discrete",
    "test_acc_relaxed", "alpha", "target_alpha", "accuracy_ema",
    "feedback_action", "weak_rows_resampled", "strong_rows_refreshed",
    "frozen_convs", "elapsed_seconds",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", type=Path, default=REFERENCE_CONFIG,
                        help="Existing best-run config.json or its containing directory.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New CAI run directory; must not already exist.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100", "mnist", "heartbeat"))
    parser.add_argument("--model-variant")
    parser.add_argument("--input-encoding")
    parser.add_argument("--hybrid-features", choices=("sine", "lbp"))
    parser.add_argument("--num-input-bits", type=int)
    parser.add_argument("--num-quantile-bits", type=int)
    parser.add_argument("--num-sine-bits", type=int)
    parser.add_argument("--sine-sigma", type=float)
    parser.add_argument("--sine-seed", type=int)
    parser.add_argument("--cifar-augmentation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--mrlb-conv3-relational-channels", type=int)
    parser.add_argument("--compact-freeze-mode", choices=("none", "entropy", "hybrid"))
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default=None)
    parser.add_argument("--compact-backend", choices=("cuda", "torch", "auto"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--exploration-steps", type=int, default=None)
    parser.add_argument("--eval-frequency", type=int, default=None)
    parser.add_argument("--initial-alpha", type=float, default=0.95)
    parser.add_argument("--step-up", type=float, default=0.08,
                        help="Fraction of remaining distance toward alpha=1 per 5%% of exploration.")
    parser.add_argument("--step-down", type=float, default=0.05)
    parser.add_argument("--step-down-mode", choices=("fractional", "absolute"),
                        default="absolute",
                        help="Use the legacy fractional response or subtract step-down directly from target alpha.")
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--ema-beta", type=float, default=0.7)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.003)
    parser.add_argument("--rollback-drop", type=float, default=0.015)
    parser.add_argument("--rollback-confirmations", type=int, default=3)
    parser.add_argument("--infusion-profile", choices=("default", "strong_unary", "ultra_unary"),
                        default="strong_unary")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration and show the planned run without loading data or a model.")
    return parser


def prepare_config(args):
    reference = args.reference_config.expanduser().resolve()
    if reference.is_dir():
        reference /= "config.json"
    if not reference.is_file():
        raise FileNotFoundError(f"Reference config not found: {reference}")
    source = json.loads(reference.read_text(encoding="utf-8"))
    # CAI runs save a nested base_config; original experiment configs are flat.
    source = source.get("base_config", source)
    allowed = {field.name for field in fields(ExperimentConfig)}
    values = {key: value for key, value in source.items() if key in allowed}
    if args.dataset is not None and args.dataset != values.get("dataset"):
        adapt_config(values, args.dataset)
    values["compact_aggressive_layer_aware_resampling"] = False
    values["compact_efficient_layer_aware_resampling"] = False
    values["compact_layer_aware_resampling"] = False
    values["compact_progressive_aggressive_infusion"] = False
    values["compact_infusion_scope"] = "all"
    values["compact_infusion_schedule"] = "aggressive"
    values["compact_infusion_level_1_only"] = False
    values["compact_infusion_start_level"] = 1
    values["compact_control_holdout"] = True
    values["compact_infusion_resume_checkpoint"] = None
    values["compact_dense_freeze"] = False
    values["experiment"] = "pure_ce"
    values["parametrization"] = "compact"
    for option, field_name in (
        (args.dataset, "dataset"), (args.model_variant, "model_variant"),
        (args.input_encoding, "input_encoding"), (args.hybrid_features, "hybrid_features"),
        (args.num_input_bits, "num_input_bits"),
        (args.num_quantile_bits, "num_quantile_bits"), (args.num_sine_bits, "num_sine_bits"),
        (args.sine_sigma, "sine_sigma"), (args.sine_seed, "sine_seed"),
        (args.cifar_augmentation, "cifar_augmentation"),
        (args.batch_size, "batch_size"), (args.num_workers, "num_workers"),
        (args.learning_rate, "learning_rate"),
        (args.mrlb_conv3_relational_channels, "mrlb_conv3_relational_channels"),
        (args.compact_freeze_mode, "compact_freeze_mode"),
        (args.device, "device"), (args.compact_backend, "compact_backend"),
        (args.seed, "seed"), (args.iterations, "iterations"),
        (args.exploration_steps, "compact_exploration_steps"),
        (args.eval_frequency, "eval_frequency"),
    ):
        if option is not None:
            values[field_name] = option
    data_dir = (args.data_dir if args.data_dir is not None else
                Path("data") if reference == REFERENCE_CONFIG else
                Path(values.get("data_dir", "data")))
    values["data_dir"] = (REPO_ROOT / data_dir).resolve() if not data_dir.is_absolute() else data_dir.resolve()
    values["output_dir"] = HERE / "results"
    config = ExperimentConfig(**values)
    if not (config.uses_compact_logic or config.uses_classrouted_compact_head):
        raise ValueError("CAI requires a CompactLogic candidate model variant.")
    config.validate()
    output = (args.output_dir if args.output_dir is not None else
              HERE / "results" / config.dataset / config.model_variant /
              args.infusion_profile / f"alpha_{args.initial_alpha:g}" /
              f"seed_{config.seed}")
    return config, reference, output.expanduser().resolve()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args):
    config, reference, output = prepare_config(args)
    # Validate controller parameters before loading the dataset or model.
    class _UnappliedPolicy:
        def set_alpha(self, alpha):
            pass

    ContinuousAdaptiveInfusion(
        _UnappliedPolicy(), exploration_steps=config.compact_exploration_steps,
        initial_alpha=args.initial_alpha, step_up=args.step_up, step_down=args.step_down,
        step_down_mode=args.step_down_mode,
        smoothing=args.smoothing, ema_beta=args.ema_beta,
        accuracy_tolerance=args.accuracy_tolerance, rollback_drop=args.rollback_drop,
        rollback_confirmations=args.rollback_confirmations,
    )
    if args.dry_run:
        print(f"reference_config={reference}")
        print(f"dataset={config.dataset} model={config.model_variant} seed={config.seed} device={config.device} backend={config.compact_backend} freeze={config.compact_freeze_mode}")
        print(f"steps={config.iterations} exploration={config.compact_exploration_steps} evaluation_every={config.eval_frequency}")
        print(f"initial_alpha={args.initial_alpha} step_down={args.step_down} "
              f"step_down_mode={args.step_down_mode} infusion_profile={args.infusion_profile} output={output}")
        return
    if output.exists():
        raise FileExistsError(f"CAI output directory already exists: {output}")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    output.mkdir(parents=True)
    cai_settings = {
        "initial_alpha": args.initial_alpha, "step_up": args.step_up,
        "step_down": args.step_down, "smoothing": args.smoothing,
        "step_down_mode": args.step_down_mode,
        "ema_beta": args.ema_beta, "accuracy_tolerance": args.accuracy_tolerance,
        "rollback_drop": args.rollback_drop,
        "rollback_confirmations": args.rollback_confirmations,
    }
    _json(output / "config.json", {
        "reference_config": str(reference), "base_config": config.to_dict(),
        "cai": cai_settings, "infusion_profile": args.infusion_profile,
        "resolved_device": str(device),
    })
    train_loader, train_eval_loader, validation_loader, test_loader, threshold_dataset = (
        build_progressive_infusion_loaders(config)
    )
    thresholds = build_cifar10_input_thresholds(config, threshold_dataset)
    model = build_model_for_dataset(
        device, dataset=config.dataset, model_variant=config.model_variant,
        parametrization="compact", input_thresholds=thresholds,
        input_encoding=config.input_encoding, num_quantile_bits=config.quantile_bits,
        num_sine_bits=config.sine_bits, sine_sigma=config.sine_sigma,
        sine_seed=config.sine_seed, compact_backend=config.compact_backend,
        hybrid_features=config.hybrid_features,
        mrlb_conv3_relational_channels=config.mrlb_conv3_relational_channels,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    compact = (HybridFreezeController(model, config) if config.compact_freeze_mode == "hybrid"
               else CompactLogicController(model, config))
    policy = ContinuousCandidatePolicy(model, infusion_profile=args.infusion_profile)
    gate_tracker = GateStabilityTracker(model)
    cai = ContinuousAdaptiveInfusion(
        policy, exploration_steps=config.compact_exploration_steps, **cai_settings,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _json(output / "model_report.json", {
        "model_variant": config.model_variant, "parameter_count": parameter_count,
        "infusion_profile": args.infusion_profile,
        "logic_layer_widths": logic_layer_width_report(model),
        "target_draw_mixtures": policy.draw_mixtures(alpha=1.0),
        "source_thresholds": thresholds.tolist() if thresholds is not None else None,
    })
    freeze_batches = (repeat_loader(train_eval_loader)
                      if config.compact_freeze_mode == "hybrid" else None)
    safe_path = output / "cai_safe.pt"
    started = time.perf_counter()
    best_validation = -1.0
    final = None
    metric_rows = []
    running_loss = running_correct = running_examples = 0
    print(f"[CAI] model={config.model_variant} parameters={parameter_count} seed={config.seed} "
          f"device={device} alpha={cai.alpha:.3f} results={output}", flush=True)

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as metrics_file, \
         (output / "freeze_probes.csv").open("w", newline="", encoding="utf-8") as freeze_file:
        writer = csv.DictWriter(
            metrics_file, fieldnames=BASE_METRIC_FIELDS + gate_tracker.metric_fields,
        )
        freeze_writer = csv.DictWriter(freeze_file, fieldnames=FREEZE_PROBE_FIELDS)
        writer.writeheader()
        freeze_writer.writeheader()
        for step, (inputs, targets) in enumerate(repeat_loader(train_loader), start=1):
            if step > config.iterations:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            model.train()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            running_loss += loss.item() * targets.numel()
            running_correct += (outputs.argmax(-1) == targets).sum().item()
            running_examples += targets.numel()
            compact.step(step, optimizer)
            if freeze_batches is not None and compact.pending_probe is not None:
                probe = []
                for _ in range(config.compact_freeze_probe_batches):
                    x, y = next(freeze_batches)
                    probe.append((x.to(device, non_blocking=True), y.to(device, non_blocking=True)))
                event = compact.probe_and_freeze(
                    model, *probe[0], step, additional_batches=probe[1:],
                )
                freeze_writer.writerow(event)
                freeze_file.flush()
                print(f"  freeze_probe step={step} layer={event['layer']} passed={event['passed']} "
                      f"frozen={event['frozen']}", flush=True)
            if step % config.eval_frequency != 0 and step != config.iterations:
                continue
            validation = evaluate(model, validation_loader, device, relaxed=False)
            discrete = evaluate(model, test_loader, device, relaxed=False)
            relaxed = evaluate(model, test_loader, device, relaxed=True)
            previous_alpha = cai.alpha
            decision = cai.observe(step=step, accuracy=validation["acc"], loss=validation["loss"])
            if decision.action == "increase":
                if (cai.safe_accuracy is None or
                        validation["acc"] > cai.safe_accuracy + cai.accuracy_tolerance):
                    cai.prepare_safe_point(step=step, accuracy=validation["acc"])
                    save_safe_checkpoint(
                        safe_path, step=step, model=model, optimizer=optimizer,
                        compact_controller=compact, cai_controller=cai,
                        config={"base": config.to_dict(), "cai": cai_settings,
                                "infusion_profile": args.infusion_profile},
                        metrics={"validation": validation, "test": discrete},
                    )
            elif decision.action == "rollback":
                load_safe_checkpoint(
                    safe_path, model=model, optimizer=optimizer,
                    compact_controller=compact, cai_controller=cai, device=device,
                )
                cai.recover_after_rollback(step=step)
                # Report the restored network, including when this is the final step.
                validation = evaluate(model, validation_loader, device, relaxed=False)
                discrete = evaluate(model, test_loader, device, relaxed=False)
                relaxed = evaluate(model, test_loader, device, relaxed=True)
            if decision.action != "rollback" and validation["acc"] > best_validation:
                best_validation = validation["acc"]
                torch.save({
                    "step": step, "model_state": model.state_dict(),
                    "config": config.to_dict(), "cai_state": cai.state_dict(),
                    "infusion_profile": args.infusion_profile,
                    "metrics": {"validation": validation, "test": discrete},
                }, output / "best_model.pt")
            final = {
                "step": step, "train_loss": running_loss / running_examples,
                "train_acc": running_correct / running_examples,
                "validation_loss_discrete": validation["loss"],
                "validation_acc_discrete": validation["acc"],
                "test_loss_discrete": discrete["loss"],
                "test_acc_discrete": discrete["acc"],
                "test_acc_relaxed": relaxed["acc"],
                "alpha": cai.alpha, "target_alpha": cai.u,
                "accuracy_ema": cai.accuracy_ema,
                "feedback_action": decision.action,
                "weak_rows_resampled": compact.weak_rows,
                "strong_rows_refreshed": compact.strong_rows,
                "frozen_convs": len(compact.summary()["frozen_conv_layers"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
            final.update(gate_tracker.measure())
            writer.writerow(final)
            metrics_file.flush()
            metric_rows.append(dict(final))
            print(f"  step={step} val_acc={validation['acc']:.4f} "
                  f"test_acc_discrete={discrete['acc']:.4f} "
                  f"alpha={previous_alpha:.3f}->{cai.alpha:.3f} "
                  f"action={decision.action} weak_resampled={compact.weak_rows} "
                  f"strong_refreshed={compact.strong_rows} "
                  f"frozen_convs={final['frozen_convs']}/{len(compact.convs)}", flush=True)
            running_loss = running_correct = running_examples = 0
    plot_created = plot_gate_dynamics(metric_rows, output / "gate_dynamics.png")
    _json(output / "summary.json", {
        "model_variant": config.model_variant, "seed": config.seed,
        "infusion_profile": args.infusion_profile,
        "parameter_count": parameter_count,
        "best_validation_accuracy": best_validation,
        "final": final, "cai_state": cai.state_dict(),
        "compact": compact.summary(),
        "gate_stability_layers": gate_tracker.layer_names,
        "gate_dynamics_plot": "gate_dynamics.png" if plot_created else None,
    })
    print(f"Finished CAI: final discrete accuracy={final['test_acc_discrete']:.4f}; results={output}")


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
