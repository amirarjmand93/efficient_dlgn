"""Save a 3D layer-by-Boolean-operation histogram from a CompactLogic checkpoint.

Run independently of model construction or the optional CompactLogic CUDA backend:
python -m examples.phase_logic.plot_gate_histogram --checkpoint results/best_model.pt
"""

import argparse
import csv
from collections.abc import Mapping
from pathlib import Path
import pickle

import torch


# Upstream truth-table convention: ID 3 is A, ID 5 is B.
OPERATION_NAMES = (
    "FALSE", "AND", "A & ~B", "A", "~A & B", "B", "XOR", "OR",
    "NOR", "XNOR", "~B", "A | ~B", "~A", "~A | B", "NAND", "TRUE",
)


def load_gate_frequencies(checkpoint_path: Path) -> list[dict]:
    """Count selected operations, never confuse a candidate slot with its opcode.

    Convolutions count one selected gate per output kernel, NOT every spatial
    application. Dense layers count one gate per output. Pinned dense rows use
    their saved operation codes. All tensors are loaded safely onto the CPU.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a state dictionary or a checkpoint dictionary.")
    state = checkpoint
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    if not isinstance(state, Mapping):
        raise ValueError("The checkpoint's model state is not a dictionary.")

    result = []
    type_counts = {"conv": 0, "dense": 0, "logic": 0}
    for key, gates in state.items():
        if not isinstance(key, str) or not (key == "gate_sequence" or key.endswith(".gate_sequence")):
            continue
        prefix = key.removesuffix("gate_sequence")
        weights = state.get(prefix + "weights")
        if not isinstance(gates, torch.Tensor) or gates.ndim != 2 or not gates.numel():
            raise ValueError(f"{key}: expected a nonempty [outputs,candidates] gate sequence.")
        if (not isinstance(weights, torch.Tensor) or weights.shape != gates.shape
                or not bool(torch.isfinite(weights).all())):
            raise ValueError(f"{prefix}weights: missing, nonfinite, or mismatched candidate logits.")
        if not bool(((gates >= 0) & (gates <= 15) & (gates == gates.long())).all()):
            raise ValueError(f"{key}: Boolean operation IDs must be integers from 0 to 15.")
        slots = state.get(prefix + "w", weights.argmax(-1))
        if (not isinstance(slots, torch.Tensor) or slots.shape != (gates.shape[0],)
                or not bool(((slots >= 0) & (slots < gates.shape[1]) & (slots == slots.long())).all())):
            raise ValueError(f"{prefix}w: invalid cached candidate selection.")
        operations = gates.long().gather(1, slots.long()[:, None]).squeeze(1)
        impl_name = prefix.rstrip(".")
        layer_name = ("" if impl_name == "impl" else
                      impl_name.removesuffix(".impl") if impl_name.endswith(".impl") else impl_name)
        wrapper_prefix = layer_name + "." if layer_name else ""
        frozen_rows = state.get(wrapper_prefix + "frozen_rows")
        if frozen_rows is not None:
            codes = state.get(wrapper_prefix + "frozen_codes")
            if (not isinstance(frozen_rows, torch.Tensor) or frozen_rows.dtype != torch.bool
                    or frozen_rows.shape != operations.shape or not isinstance(codes, torch.Tensor)
                    or codes.shape != operations.shape):
                raise ValueError(f"{layer_name}: invalid frozen-row masks/codes.")
            if not bool(((codes[frozen_rows] >= 0) & (codes[frozen_rows] <= 15)
                         & (codes[frozen_rows] == codes[frozen_rows].long())).all()):
                raise ValueError(f"{layer_name}: invalid pinned Boolean operation IDs.")
            operations[frozen_rows] = codes[frozen_rows].long()
        kind = ("conv" if prefix + "offsets_ch" in state else
                "dense" if prefix + "indices" in state else "logic")
        type_counts[kind] += 1
        display_type = {"conv": "Conv/branch", "dense": "Dense", "logic": "Logic"}[kind]
        result.append({
            "layer": layer_name or "root", "kind": kind,
            "label": f"{display_type} {type_counts[kind]} [{layer_name or 'root'}]",
            "counts": torch.bincount(operations, minlength=16).tolist(),
            "total": operations.numel(),
        })
    if not result:
        raise ValueError("No CompactLogic gate_sequence/weights layers found in this checkpoint.")
    return result


def save_histogram(layers: list[dict], output_path: Path, *, frequency="count", dpi=180,
                   checkpoint_name="") -> Path:
    """Save a headless Matplotlib bar3d figure and an exact-count CSV companion."""
    if frequency not in {"count", "percent"}:
        raise ValueError("frequency must be 'count' or 'percent'.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    if output_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("Output must have a .png, .pdf, or .svg extension.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise ImportError("Install plotting support with: python -m pip install '.[plots]'") from error

    values = np.array([layer["counts"] for layer in layers], dtype=float)
    if frequency == "percent":
        values = values / np.array([layer["total"] for layer in layers])[:, None] * 100
    gate_ids = np.tile(np.arange(16), len(layers))
    layer_ids = np.repeat(np.arange(len(layers)), 16)
    colors = plt.get_cmap("tab20")(np.arange(16) / 19)
    fig = plt.figure(figsize=(20, max(10, min(18, 6 + 0.7 * len(layers)))))
    try:
        ax = fig.add_subplot(111, projection="3d")
        ax.bar3d(gate_ids - 0.35, layer_ids - 0.35, np.zeros(values.size),
                 0.7, 0.7, values.ravel(), color=colors[gate_ids],
                 edgecolor="white", linewidth=0.25, shade=True, alpha=0.95)
        ax.set_xticks(range(16), [f"{i}: {name}" for i, name in enumerate(OPERATION_NAMES)],
                      rotation=55, ha="right", fontsize=9)
        ax.set_yticks(range(len(layers)), [layer["label"] for layer in layers], fontsize=9)
        ax.set_xlabel("Selected Boolean operation", labelpad=42)
        ax.set_ylabel("Layer (checkpoint module index in brackets)", labelpad=24)
        ax.set_zlabel("Gate frequency (%) within layer" if frequency == "percent" else "Selected gate count",
                      labelpad=14)
        ax.set_box_aspect((2.1, max(1, min(2.2, len(layers) / 5)), 1.1))
        ax.view_init(elev=26, azim=-58)
        total = sum(layer["total"] for layer in layers)
        fig.suptitle("Discrete Boolean gate selection by layer", fontsize=20, fontweight="bold", y=0.96)
        fig.text(0.5, 0.91, f"{checkpoint_name} | {len(layers)} layers | {total:,} selected gates",
                 ha="center", fontsize=11)
        fig.text(0.5, 0.035,
                 "Conv/branch counts are per output kernel; dense counts are per output. "
                 "Spatial reuse is not counted (these are not BOP/image counts).",
                 ha="center", fontsize=10)
        fig.subplots_adjust(left=0.04, right=0.89, bottom=0.16, top=0.87)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "layer", "label", "kind", "operation_id", "operation", "count", "percent", "layer_total"))
        writer.writeheader()
        for layer in layers:
            for operation, count in enumerate(layer["counts"]):
                writer.writerow({
                    "layer": layer["layer"], "label": layer["label"], "kind": layer["kind"],
                    "operation_id": operation, "operation": OPERATION_NAMES[operation], "count": count,
                    "percent": 100 * count / layer["total"], "layer_total": layer["total"],
                })
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Relative to the current working directory, or an absolute .pt path.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Image path (.png/.pdf/.svg); default: next to checkpoint.")
    parser.add_argument("--frequency", choices=("count", "percent"), default="count",
                        help="Use percent to compare small conv layers with wide dense layers.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else
              checkpoint.with_name(checkpoint.stem + "_gate_histogram_3d.png"))
    try:
        layers = load_gate_frequencies(checkpoint)
        csv_path = save_histogram(layers, output, frequency=args.frequency, dpi=args.dpi,
                                  checkpoint_name=checkpoint.name)
    except (OSError, ValueError, RuntimeError, ImportError, pickle.UnpicklingError) as error:
        parser.error(str(error))
    for layer in layers:
        print(f"{layer['label']}: {layer['total']:,} selected gates")
    print(f"Image saved: {output}")
    print(f"Frequencies saved: {csv_path}")


if __name__ == "__main__":
    main()
