"""Restore a CompactLogic checkpoint and draw its actual module-level graph.

Example:
    python -m examples.phase_logic.post_training.load_checkpoint \
        --checkpoint path/to/best_model.pt --output-dir path/to/inspection

No training, dataset downloads, or circuit export is performed. The CPU reference
backend is the default, including for checkpoints originally trained with CUDA.
"""

import argparse
from collections.abc import Mapping
from dataclasses import fields
import json
import math
from pathlib import Path

import torch

from ..compact import CandidateLogicLayer, UniformThermometerEncoder
from ..config import ExperimentConfig
from ..dense_freezing import enable_dense_group_freezing
from ..model import HybridFeatureEncoder, build_model_for_dataset, logic_layer_width_report
from torchlogix.layers import GroupSum


# Inverse connectivity acceleration data, not learned inference connections.
CUDA_BACKWARD_BUFFERS = (
    "given_x_indices_of_y_start", "given_x_indices_of_y", "given_x_indices_of_gate",
)


class ArchitectureTracer(torch.fx.Tracer):
    """Trace wiring, not the internals of encoders or custom candidate kernels."""

    def is_leaf_module(self, module, qualified_name):
        if isinstance(module, (CandidateLogicLayer, HybridFeatureEncoder,
                               UniformThermometerEncoder, GroupSum)):
            return True
        return super().is_leaf_module(module, qualified_name)


def architecture_graph(model):
    """Describe real forward dependencies, including cat skips and branches.

    Shapes omit the batch dimension. FX tracing executes Python wiring only;
    leaf layers are not evaluated and no image/model inference is necessary.
    """
    modules = dict(model.named_modules())
    graph = ArchitectureTracer().trace(model)
    shapes = {}
    nodes, edges = [], []
    conv_index = dense_index = 0
    for node in graph.nodes:
        shape = None
        label = str(node.target)
        kind = node.op
        if node.op == "placeholder":
            label, shape, kind = "RGB input", (3, 32, 32), "input"
        elif node.op == "call_module":
            module = modules[node.target]
            if isinstance(module, (HybridFeatureEncoder, UniformThermometerEncoder)):
                shape = (module.output_channels, 32, 32)
                label, kind = "Encoder", "encoder"
                if isinstance(module, HybridFeatureEncoder):
                    bits = f"Q{module.num_quantile_bits}"
                    bits += (f" + LBP{module.num_lbp_bits}" if getattr(module, "num_lbp_bits", 0)
                             else f" + S{module.num_sine_bits}" if module.num_sine_bits else "")
                    label += "\n" + bits
            elif isinstance(module, CandidateLogicLayer):
                if module.is_convolution:
                    conv_index += 1
                    shape = (module.out_dim, *[(d + module.stride - 1) // module.stride
                                               for d in module.in_shape[1:]])
                    kernel = 2 * module.impl.padding + 1
                    label = f"Conv/branch {conv_index}\n{kernel}x{kernel}, stride {module.stride}"
                    kind = "conv"
                else:
                    dense_index += 1
                    label, shape, kind = f"Shared Dense {dense_index}", (module.out_dim,), "dense"
            elif isinstance(module, GroupSum):
                label, shape, kind = f"GroupSum({module.k})\ntau={module.tau:g}", (module.k,), "sum"
            elif isinstance(module, torch.nn.Flatten):
                source = shapes.get(node.all_input_nodes[0].name)
                shape = (math.prod(source),) if source else None
                label, kind = "Flatten", "wiring"
            else:
                raise ValueError(f"Unsupported diagram module {node.target}: {type(module).__name__}.")
        elif node.op == "call_function" and node.target is torch.cat:
            sources = [shapes[n.name] for n in node.args[0]]
            dimension = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0)
            if dimension != 1 or not all(sources):
                raise ValueError("Architecture visualization expects channel/feature concatenation.")
            shape = (sum(s[0] for s in sources), *sources[0][1:])
            label, kind = "Concatenate", "wiring"
        elif node.op == "output":
            shape = shapes.get(node.all_input_nodes[0].name)
            label, kind = "Class scores", "output"
        else:
            raise ValueError(f"Unsupported architecture graph operation: {node.op} {node.target}.")
        shapes[node.name] = shape
        if shape:
            label += "\n" + " x ".join(f"{d:,}" for d in shape)
        nodes.append({"id": node.name, "module": str(node.target), "kind": kind,
                      "label": label, "shape": list(shape) if shape else None})
        edges.extend({"source": source.name, "target": node.name} for source in node.all_input_nodes)
    return {"nodes": nodes, "edges": edges}


def load_checkpoint(checkpoint_path, *, config_path=None, device="cpu", backend="torch"):
    """Return ``(model_in_discrete_eval_mode, JSON_serializable_report)``.

    Supports training checkpoint dictionaries and raw state dictionaries with a
    sibling config.json (or --config). Reuses existing builders; strict loading
    rejects incompatible historical architecture revisions rather than guessing.
    Optimizer/controller/RNG state is not restored: this is inference inspection.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(saved, Mapping):
        raise ValueError("Expected a checkpoint dictionary, not a pickled model object.")
    state = saved
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in saved:
            state = saved[key]
            break
    if not isinstance(state, Mapping) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Missing or invalid model state dictionary.")
    config_file = Path(config_path).expanduser().resolve() if config_path else checkpoint_path.parent / "config.json"
    if config_path or not isinstance(saved.get("config"), Mapping):
        if not config_file.is_file():
            raise ValueError("No saved configuration found. Supply --config path/to/config.json.")
        configuration = json.loads(config_file.read_text())
    else:
        configuration = dict(saved["config"])
    required = {"dataset", "model_variant", "parametrization"}
    if not isinstance(configuration, Mapping) or not required.issubset(configuration):
        raise ValueError("Configuration must specify dataset, model_variant, and parametrization.")
    config = ExperimentConfig(**{k: v for k, v in configuration.items()
                                 if k in {f.name for f in fields(ExperimentConfig)}})
    if config.dataset != "cifar10" or config.parametrization != "compact" or not (
        config.uses_compact_logic or config.uses_classrouted_compact_head
    ):
        raise ValueError("This loader currently supports the CompactLogic CIFAR-10 model variants only.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Use --device cpu --backend torch.")
    if backend not in {"auto", "cuda", "torch"}:
        raise ValueError("backend must be auto, cuda, or torch.")
    resolved_backend = ("cuda" if device.type == "cuda" else "torch") if backend == "auto" else backend
    rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        model = build_model_for_dataset(
            device=device, dataset=config.dataset, model_variant=config.model_variant,
            parametrization=config.parametrization, input_thresholds=state.get("0.thresholds"),
            input_encoding=config.input_encoding, hybrid_features=config.hybrid_features,
            num_quantile_bits=config.quantile_bits, num_sine_bits=config.sine_bits,
            sine_sigma=config.sine_sigma, sine_seed=config.sine_seed,
            compact_backend=resolved_backend,
            mrlb_conv3_relational_channels=config.mrlb_conv3_relational_channels,
        )
    if any(key.endswith(".frozen_rows") for key in state):
        enable_dense_group_freezing(model)
    restored_state = dict(state)
    ignored = []
    if resolved_backend == "torch":
        for name, module in model.named_modules():
            if not isinstance(module, CandidateLogicLayer):
                continue
            prefix = name + ".impl."
            for suffix in CUDA_BACKWARD_BUFFERS:
                key = prefix + suffix
                if key in restored_state:
                    restored_state.pop(key)
                    ignored.append(key)
            key = prefix + "w"
            if key in restored_state:
                if not torch.equal(restored_state[key].long(), state[prefix + "weights"].argmax(-1)):
                    raise ValueError(f"{key}: cached selection differs from logits; use the CUDA backend.")
                restored_state.pop(key)
                ignored.append(key)
    try:
        model.load_state_dict(restored_state, strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint does not match this builder revision. No weights were silently skipped. "
                         "Use the architecture code matching the original run.\n" + str(error)) from error
    # GroupSum tau/beta are Python attributes, not state_dict tensors.
    warnings = []
    report_file = checkpoint_path.parent / "model_report.json"
    if report_file.is_file():
        original_report = json.loads(report_file.read_text())
        tau = original_report.get("compact", {}).get("group_sum_tau")
        if tau is not None:
            for module in model.modules():
                if isinstance(module, GroupSum):
                    module.tau = float(tau)
        else:
            warnings.append("Saved model report has no GroupSum tau; using the current builder default.")
    else:
        warnings.append("No model_report.json: GroupSum tau uses the current builder default; verify for historical runs.")
    model.eval()
    for module in model.modules():
        if isinstance(module, CandidateLogicLayer):
            module.impl.eval()  # Keep inner modes consistent before the first forward.
    widths = logic_layer_width_report(model)
    report = {
        "checkpoint": str(checkpoint_path), "model_variant": config.model_variant,
        "dataset": config.dataset, "device": str(device), "backend": resolved_backend,
        "mode": "discrete eval", "step": saved.get("step"), "metrics": saved.get("metrics", {}),
        "configuration": dict(configuration), "warnings": warnings,
        "ignored_backend_buffers": ignored,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "logic_layer_widths": widths,
        "boolean_evaluations_per_image": sum(r["lut_node_applications"] for r in widths),
        "group_sum_tau": model[-1].tau,
        "architecture": architecture_graph(model),
    }
    return model, report


def save_architecture(report, output_dir):
    """Save a module-level PNG/SVG and matching JSON (not a gate-level netlist)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ImportError as error:
        raise ImportError("Install plotting support: python -m pip install '.[plots]'") from error
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = report["architecture"]
    depth, levels = {}, {}
    for node in graph["nodes"]:
        parents = [e["source"] for e in graph["edges"] if e["target"] == node["id"]]
        depth[node["id"]] = 1 + max((depth[p] for p in parents), default=-1)
        levels.setdefault(depth[node["id"]], []).append(node["id"])
    positions = {}
    for level, ids in levels.items():
        for i, node_id in enumerate(ids):
            positions[node_id] = (level * 1.9, (i - (len(ids) - 1) / 2) * 1.8)
    colors = {"conv": "#d9eafa", "dense": "#e2f0d9", "encoder": "#fff0c2",
              "wiring": "#eeeeee", "sum": "#f4dcec"}
    last = max(depth.values())
    height = max(len(ids) for ids in levels.values())
    fig, ax = plt.subplots(figsize=(max(12, (last + 1) * 1.7), max(4.5, height * 2.0)))
    for edge in graph["edges"]:
        start, end = positions[edge["source"]], positions[edge["target"]]
        skip = depth[edge["target"]] - depth[edge["source"]] > 1
        start_point = (start[0], start[1] + 0.53) if skip else (start[0] + 0.78, start[1])
        end_point = (end[0], end[1] + 0.53) if skip else (end[0] - 0.78, end[1])
        ax.add_patch(FancyArrowPatch(
            start_point, end_point,
            arrowstyle="-|>", mutation_scale=12, color="#2577aa" if skip else "#536779",
            connectionstyle=f"arc3,rad={-0.42 if skip else 0}", linewidth=1.3,
            zorder=1,
        ))
    for node in graph["nodes"]:
        x, y = positions[node["id"]]
        ax.add_patch(FancyBboxPatch((x - 0.75, y - 0.47), 1.5, 0.94,
                                   boxstyle="round,pad=0.04", edgecolor="#536779",
                                   facecolor=colors.get(node["kind"], "#f2f4f7"), zorder=2))
        ax.text(x, y, node["label"], ha="center", va="center", fontsize=8, zorder=3)
    ax.set_xlim(-1.1, last * 1.9 + 1.1)
    ax.set_ylim(-max(1.8, height), max(2.0, height))
    ax.axis("off")
    ax.set_title(f"{report['model_variant']}\n"
                 f"{report['parameter_count']:,} candidate parameters | "
                 f"{report['boolean_evaluations_per_image']:,} Boolean evaluations/image "
                 "(encoder and GroupSum excluded)", fontsize=12, pad=20)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"architecture.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (output_dir / "architecture.json").write_text(json.dumps(report, indent=2) + "\n")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Explicit config JSON; otherwise use embedded or sibling config.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("torch", "cuda", "auto"), default="torch")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: post_training/ beside the checkpoint.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run one synthetic image through discrete inference; not an accuracy test.")
    args = parser.parse_args()
    try:
        model, report = load_checkpoint(args.checkpoint, config_path=args.config,
                                        device=args.device, backend=args.backend)
        if args.smoke_test:
            with torch.random.fork_rng(devices=[]), torch.inference_mode():
                image = torch.rand(1, 3, 32, 32).to(args.device)
                scores = model(image)
            if scores.shape != (1, 10) or not bool(torch.isfinite(scores).all()):
                raise ValueError("Synthetic forward produced invalid class scores.")
            print("Synthetic forward passed: output shape (1, 10).")
        output_dir = save_architecture(report, args.output_dir or args.checkpoint.resolve().parent / "post_training")
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Loaded {report['model_variant']} at step {report['step']} ({report['mode']}, {report['backend']}).")
    print(f"Parameters={report['parameter_count']:,}; Boolean evaluations/image={report['boolean_evaluations_per_image']:,}.")
    for message in report["warnings"]:
        print("Warning:", message)
    print(f"Saved architecture.png, architecture.svg, architecture.json under {output_dir}")


if __name__ == "__main__":
    main()
