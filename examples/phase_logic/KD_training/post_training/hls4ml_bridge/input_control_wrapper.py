"""Generate a reset-free data input bank around the HLS compute core."""

from pathlib import Path
import re


BANK_WIDTH = 96


def write_input_control_wrapper(output_dir, project_name, input_bits, output_name, classes):
    """Write a wrapper; the HLS core itself keeps the exact LUT6 computation."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", project_name):
        raise ValueError("Project name must be a Verilog identifier.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", output_name):
        raise ValueError("Output name must be a Verilog identifier.")
    if input_bits <= 0 or input_bits % BANK_WIDTH:
        raise ValueError(f"Banked input requires a multiple of {BANK_WIDTH} bits.")
    if classes != 10:
        raise ValueError("Banked wrapper expects the ten class-count outputs.")

    banks = input_bits // BANK_WIDTH
    wrapper_name = f"{project_name}_banked"
    output_ports = [
        f"{output_name}_{i}, {output_name}_{i}_ap_vld" for i in range(classes)
    ]
    ports = ",\n    ".join([
        "ap_clk", "ap_rst", "ap_start", "ap_done", "ap_idle", "ap_ready",
        "encoded_input", "encoded_input_ap_vld", *output_ports,
    ])
    output_declarations = "\n".join(
        f"    output wire [9:0] {output_name}_{i};\n"
        f"    output wire {output_name}_{i}_ap_vld;"
        for i in range(classes)
    )
    core_outputs = ",\n".join(
        f"        .{output_name}_{i}({output_name}_{i}),\n"
        f"        .{output_name}_{i}_ap_vld({output_name}_{i}_ap_vld)"
        for i in range(classes)
    )
    rtl = f"""// Generated input-control wrapper. Input data registers have no reset.
// Accepted inputs are sampled on the ap_start/ap_ready/valid edge. The core
// starts two cycles later; outputs retain the core's exact class-count bits.
module {wrapper_name} (
    {ports}
);
    input wire ap_clk;
    input wire ap_rst;
    input wire ap_start;
    output wire ap_done;
    output wire ap_idle;
    output wire ap_ready;
    input wire [{input_bits - 1}:0] encoded_input;
    input wire encoded_input_ap_vld;
{output_declarations}

    localparam [1:0] IDLE = 2'd0, CAPTURE = 2'd1,
                     START = 2'd2, RUN = 2'd3;
    reg [1:0] state;
    wire core_done;
    wire accept = ap_ready && ap_start && encoded_input_ap_vld;
    wire [{input_bits - 1}:0] core_input;

    assign ap_ready = (state == IDLE) && !ap_rst;
    assign ap_idle = (state == IDLE) && !ap_rst;
    assign ap_done = (state == RUN) && core_done;

    always @(posedge ap_clk) begin
        if (ap_rst) state <= IDLE;
        else case (state)
            IDLE: if (accept) state <= CAPTURE;
            CAPTURE: state <= START;
            START: state <= RUN;
            RUN: if (core_done) state <= IDLE;
            default: state <= IDLE;
        endcase
    end

    // The unconditional first stage captures the frame on the accept edge.
    // Each locally registered load controls only one {BANK_WIDTH}-bit held bank.
    genvar bank;
    generate for (bank = 0; bank < {banks}; bank = bank + 1) begin: input_bank
        reg [{BANK_WIDTH - 1}:0] incoming;
        reg [{BANK_WIDTH - 1}:0] held;
        (* DONT_TOUCH = "true" *) reg load_local;
        always @(posedge ap_clk) begin
            incoming <= encoded_input[bank*{BANK_WIDTH} +: {BANK_WIDTH}];
            if (ap_rst) load_local <= 1'b0;
            else load_local <= accept;
            if (load_local) held <= incoming;
        end
        assign core_input[bank*{BANK_WIDTH} +: {BANK_WIDTH}] = held;
    end endgenerate

    {project_name} core (
        .ap_clk(ap_clk), .ap_rst(ap_rst),
        .ap_start(state == START), .ap_done(core_done),
        .ap_idle(), .ap_ready(),
        .encoded_input(core_input),
{core_outputs}
    );
endmodule
"""
    destination = Path(output_dir) / f"{wrapper_name}.v"
    destination.write_text(rtl)
    return destination
