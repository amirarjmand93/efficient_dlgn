"""Build an hls4ml ModelGraph from hls4ml_lut6_ir_v1 and optionally write C++."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np

from .reference import load_ir, predict
from .input_control_wrapper import write_input_control_wrapper


def _make_generated_top_scalable(output_dir, project_name, document, *, bank_input_control=False):
    """Replace each intermediate bit array with independent 64-bit banks."""
    source = Path(output_dir).expanduser().resolve() / "firmware" / f"{project_name}.cpp"
    text = source.read_text()
    input_bits = int(np.prod(document["input"]["shape"]))
    if input_bits > 8191:
        raise ValueError(f"Encoded input has {input_bits} bits; the Vitis port limit is 8191.")
    input_name = re.escape(document["input"]["name"])
    input_declaration = re.search(rf"\binput_t\s+{input_name}\[([\d* ]+)\]", text)
    if input_declaration is None:
        raise RuntimeError("Generated top has no expected encoded input declaration.")
    declared_input_bits = 1
    for factor in input_declaration.group(1).replace(" ", "").split("*"):
        declared_input_bits *= int(factor)
    if declared_input_bits != input_bits:
        raise RuntimeError("Generated input width does not match the JSON.")
    pipeline = "    #pragma HLS PIPELINE\n"
    if text.count(pipeline) != 1:
        raise RuntimeError(
            f"Expected exactly one generated top-level pipeline pragma in {source}."
        )
    text = text.replace(pipeline, "", 1)
    expected_sizes = []
    for layer in document["layers"][:-1]:
        if layer["op_type"] == "LUT6Conv2D":
            if int(np.prod(layer["output_shape"][1:])) % 64:
                raise ValueError(f"{layer['name']} spatial output does not fill 64-bit banks.")
        expected_sizes.append(int(np.prod(layer["output_shape"])))
    declaration = re.compile(r"^\s*\w+_t\s+(layer\d+_out)\[([\d* ]+)\];$", re.MULTILINE)
    found = list(declaration.finditer(text))
    if len(found) != len(expected_sizes):
        raise RuntimeError("Generated top has unexpected intermediate declarations.")
    for match, expected in reversed(list(zip(found, expected_sizes))):
        name, expression = match.group(1), match.group(2)
        actual = 1
        for factor in expression.replace(" ", "").split("*"):
            actual *= int(factor)
        if actual != expected:
            raise RuntimeError(f"{name} has {actual} bits; JSON declares {expected}.")
        banks = (actual + 63) // 64
        pragma = f"#pragma HLS ARRAY_PARTITION variable={name} complete dim=0"
        if text.count(pragma) != 1:
            raise RuntimeError(f"Expected one partition directive for {name}.")
        text = text[:match.start()] + f"    ap_uint<64> {name}[{banks}];" + text[match.end():]
    if "ARRAY_RESHAPE variable=layer" in text:
        raise RuntimeError("Unexpected complete reshape on an intermediate layer.")
    if bank_input_control:
        # ap_vld creates a resettable 6,144-bit HLS input holding register.
        # The generated wrapper instead captures data in reset-free local banks.
        interface = re.compile(
            rf"^[ \t]*#pragma HLS INTERFACE ap_vld port={input_name},(\w+)[ \t]*$",
            re.MULTILINE,
        )
        matches = list(interface.finditer(text))
        if len(matches) != 1:
            raise RuntimeError("Expected one HLS ap_vld input/output interface pragma.")
        output_name = matches[0].group(1)
        replacement = (f"    #pragma HLS INTERFACE ap_none port={document['input']['name']}\n"
                       f"    #pragma HLS INTERFACE ap_vld port={output_name}")
        text = text[:matches[0].start()] + replacement + text[matches[0].end():]
    source.write_text(text)

    if bank_input_control:
        wrapper = write_input_control_wrapper(
            source.parents[1], project_name, input_bits, output_name,
            int(document["layers"][-1]["classes"]),
        )
        print(f"Generated reset-free input-bank wrapper: {wrapper}")

    # Permit a low-memory C++ compiler for the optional reference comparison.
    # The generated script otherwise hardcodes g++ -O3, which is expensive for
    # this large fully expanded Boolean graph.
    build_script = source.parents[1] / "build_lib.sh"
    script = build_script.read_text()
    if script.count("CC=g++\n") != 1 or script.count('CFLAGS="-O3 -fPIC"\n') != 1:
        raise RuntimeError("Unexpected hls4ml C simulation build script.")
    script = script.replace("CC=g++\n", 'CC="${HLS4ML_CXX:-g++}"\n', 1)
    script = script.replace(
        'CFLAGS="-O3 -fPIC"\n',
        'CFLAGS="${HLS4ML_CXXFLAGS:--O3 -fPIC}"\n',
        1,
    )
    build_script.write_text(script)


def _require_hls4ml():
    try:
        import hls4ml
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "hls4ml is required to construct the ModelGraph; it is not installed in this environment."
        ) from error
    return hls4ml


def _layer_list(document):
    input_spec = document["input"]
    result = [{
        "name": input_spec["name"],
        "class_name": "InputLayer",
        "input_shape": input_spec["shape"],
    }]
    for source in document["layers"]:
        layer = dict(source)
        layer["class_name"] = layer["op_type"]
        layer["inputs"] = [layer["input"]]
        if layer["op_type"] == "LUT6Conv2D":
            layer["input_is_flat"] = layer["input"] == input_spec["name"]
        result.append(layer)
    return result


def build_model_graph(document, *, output_dir, project_name="lut6_student",
                      backend="Vitis", part=None, clock_period=5.0):
    """Construct but do not compile or synthesize an hls4ml ModelGraph."""
    hls4ml = _require_hls4ml()
    from hls4ml.model import ModelGraph
    from hls4ml.utils.config import create_config

    from .layers import register_layers
    from .templates import register_backend

    register_layers()
    register_backend(backend)
    config_options = {
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "project_name": project_name,
        "backend": backend,
        "clock_period": float(clock_period),
        "io_type": "io_parallel",
    }
    if part:
        config_options["part"] = part
    config = create_config(**config_options)
    config["HLSConfig"] = {
        "Model": {
            "Precision": "ap_uint<1>",
            "ReuseFactor": 1,
            "Strategy": "Latency",
        },
        "LayerName": {
            document["input"]["name"]: {
                "Precision": {"result": "ap_uint<1>"},
            },
        },
    }
    layers = _layer_list(document)
    model = ModelGraph.from_layer_list(
        config,
        layers,
        inputs=[document["input"]["name"]],
        outputs=[document["layers"][-1]["name"]],
    )
    expected = {
        document["input"]["name"]: 1,
        **{layer["name"]: 1 for layer in document["layers"][:-1]},
        document["layers"][-1]["name"]: 10,
    }
    for name, width in expected.items():
        precision = model.graph[name].get_output_variable().type.precision
        if precision.width != width or precision.signed:
            raise RuntimeError(
                f"Unexpected output precision for {name}: expected ap_uint<{width}>, got {precision}."
            )
    return model


def _compare(model, document, vector_file, samples):
    with np.load(vector_file) as vectors:
        if "encoded" not in vectors:
            raise ValueError("Equivalence NPZ must contain an 'encoded' array.")
        encoded = vectors["encoded"][:samples]
        saved_counts = vectors["class_counts"][:samples] if "class_counts" in vectors else None
    expected = predict(document, encoded)
    if saved_counts is not None and not np.array_equal(expected, saved_counts):
        raise ValueError("Python reference disagrees with class_counts stored in the NPZ file.")
    # ModelGraph.compile() calls write() again and would discard the banked top.
    model._compile()
    actual = np.asarray(model.predict(encoded.astype(np.float32)))
    actual = np.rint(actual).astype(np.uint16).reshape(expected.shape)
    if not np.array_equal(actual, expected):
        difference = int(np.abs(actual.astype(np.int32) - expected.astype(np.int32)).max())
        raise ValueError(f"hls4ml C simulation mismatch; maximum class-count difference={difference}.")
    print(f"Exact reference/C-simulation agreement for {len(encoded)} samples.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True, help="hls4ml_lut6_ir_v1 JSON file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Generated hls4ml project directory")
    parser.add_argument("--project-name", default="lut6_student")
    parser.add_argument("--backend", choices=("Vivado", "Vitis"), default="Vitis")
    parser.add_argument("--part", help="Optional FPGA part passed to hls4ml")
    parser.add_argument("--clock-period", type=float, default=5.0)
    parser.add_argument("--write", action="store_true", help="Generate the HLS C++ project; do not synthesize it")
    parser.add_argument("--bank-input-control", action="store_true",
                        help="Use ap_none on core input and generate a reset-free 96-bit-bank RTL wrapper")
    parser.add_argument("--compare-npz", type=Path,
                        help="Compile C simulation and compare with reference vectors")
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    if args.clock_period <= 0 or args.samples <= 0:
        parser.error("--clock-period and --samples must be positive.")
    if args.bank_input_control and not (args.write or args.compare_npz):
        parser.error("--bank-input-control requires --write or --compare-npz.")
    try:
        document = load_ir(args.json)
        model = build_model_graph(
            document,
            output_dir=args.output_dir,
            project_name=args.project_name,
            backend=args.backend,
            part=args.part,
            clock_period=args.clock_period,
        )
        print("Constructed ModelGraph:", " -> ".join(layer.name for layer in model.get_layers()))
        if args.write or args.compare_npz:
            model.write()
            _make_generated_top_scalable(
                args.output_dir, args.project_name, document,
                bank_input_control=args.bank_input_control,
            )
            print(f"Generated HLS C++ project in {args.output_dir.resolve()}")
        if args.compare_npz:
            _compare(model, document, args.compare_npz, args.samples)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
