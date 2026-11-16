#!/usr/bin/env python3
"""Standard-library LUT6 IR -> sequential SystemVerilog bridge.

Supported: hls4ml_lut6_ir_v1 with a linear chain of LUT6Conv2D / LUT6Dense
followed by BooleanGroupSum. No residual links, pooling, quantization or raw
RGB encoding is inferred. Inputs are pre-encoded Boolean CHW or flattened bits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
TEMPLATE = BASE / 'rtl' / 'lut6_engine.sv.in'


class IRValidationError(ValueError):
    pass


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise IRValidationError(msg)


def size(shape: list[int]) -> int:
    require(isinstance(shape, list) and bool(shape) and all(type(n) is int and n > 0 for n in shape), f'Invalid shape: {shape}')
    return math.prod(shape)


def bits_for_max_index(count: int) -> int:
    return max(1, (count - 1).bit_length())


def validated_model(ir: dict[str, Any]) -> dict[str, Any]:
    require(ir.get('format') == 'hls4ml_lut6_ir_v1', 'Expected format=hls4ml_lut6_ir_v1')
    inp = ir.get('input', {})
    require(inp.get('dtype') == 'boolean', 'Only pre-encoded boolean input is supported')
    input_shape = inp.get('shape')
    n_input = size(input_shape)
    require(len(input_shape) in (1, 3), 'Input shape must be [N] or CHW [C,H,W]')
    if len(input_shape) == 3:
        require(inp.get('layout', 'CHW') == 'CHW', 'Only CHW layout supported')
        require(inp.get('flattening', {}).get('order', 'channel,row,column') == 'channel,row,column', 'Flattening must be CHW')

    layers = ir.get('layers')
    require(isinstance(layers, list) and len(layers) >= 2, 'Expected at least one LUT6 layer plus BooleanGroupSum')
    head = layers[-1]
    require(head.get('op_type') == 'BooleanGroupSum', 'Final layer must be BooleanGroupSum')
    compute = layers[:-1]
    prev_shape = input_shape
    prev_name = inp.get('name', 'encoded_input')
    stage_sizes = []
    for i, layer in enumerate(compute):
        label = f'layer {i} ({layer.get("name", "?")})'
        require(layer.get('op_type') in ('LUT6Conv2D', 'LUT6Dense'), f'{label}: Unsupported op {layer.get("op_type")}')
        next_in_shape=layer.get('input_shape')
        flatten_ok=(layer['op_type']=='LUT6Dense' and len(prev_shape)==3 and next_in_shape==[size(prev_shape)])
        require(next_in_shape == prev_shape or flatten_ok, f'{label}: input_shape does not match preceding output (except legal Conv-to-Dense flattening)')
        require(layer.get('input') == prev_name, f'{label}: skip or nonsequential dependency is unsupported')
        require(layer.get('pin0_is_address_lsb') is True, f'{label}: Expected pin0_is_address_lsb true')
        out_shape = layer.get('output_shape')
        n_out = size(out_shape)
        connections, inits = layer.get('connections'), layer.get('init_hex')
        require(isinstance(connections, list) and isinstance(inits, list), f'{label}: Missing connections/init_hex')
        if layer['op_type'] == 'LUT6Conv2D':
            require(len(prev_shape) == len(out_shape) == 3, f'{label}: Conv requires CHW input and output')
            c, h, w = prev_shape
            co, ho, wo = out_shape
            kernel = layer.get('kernel_size')
            stride = layer.get('stride')
            padding = layer.get('padding')
            require(all(isinstance(x, list) and len(x) == 2 and all(type(t) is int for t in x) for x in [kernel, stride, padding]), f'{label}: invalid kernel/stride/padding')
            kh, kw = kernel
            sy, sx = stride
            py, px = padding
            require(min(kh,kw,sy,sx) > 0 and min(py,px) >= 0, f'{label}: invalid geometry')
            require(layer.get('padding_value') == 0, f'{label}: only zero padding supported')
            require(layer.get('local_patch_flattening', {}).get('order') == 'channel,row,column', f'{label}: patch order must be CHW')
            expected_h = (h + 2*py - kh) // sy + 1
            expected_w = (w + 2*px - kw) // sx + 1
            require((ho,wo) == (expected_h,expected_w), f'{label}: output shape inconsistent with convolution')
            require(len(connections) in (co, co*ho*wo), f'{label}: expected output-channel shared or per-output LUT parameters')
            max_conn = c*kh*kw
            if len(connections) == co:
                sharing = layer.get('weight_sharing', {}).get('parameter_index', 'output_channel')
                require(sharing == 'output_channel', f'{label}: only channel-shared LUT parameters supported')
        else:
            require(len(layer['input_shape']) == len(out_shape) == 1, f'{label}: Dense requires flattened one-dimensional shapes')
            require(len(connections) == n_out, f'{label}: expected {n_out} Dense LUTs')
            max_conn = size(prev_shape)
        require(len(connections) == len(inits), f'{label}: connections/init_hex length mismatch')
        for j, (pins, hextable) in enumerate(zip(connections,inits)):
            require(isinstance(pins,list) and len(pins) == 6, f'{label} gate {j}: needs six pins')
            require(all(type(p) is int and 0 <= p < max_conn for p in pins), f'{label} gate {j}: connection out of range')
            require(isinstance(hextable,str) and len(hextable) == 16, f'{label} gate {j}: INIT must be 16 hex digits')
            try:
                int(hextable,16)
            except ValueError:
                raise IRValidationError(f'{label} gate {j}: bad INIT hex') from None
        stage_sizes.append(n_out)
        prev_shape = out_shape
        prev_name = layer['name']
    require(head.get('input') == prev_name and head.get('input_shape') == prev_shape, 'GroupSum must read last compute layer')
    n_classes = head.get('classes')
    require(type(n_classes) is int and n_classes > 0, 'Invalid GroupSum classes')
    require(head.get('output_shape') == [n_classes], 'GroupSum output_shape mismatch')
    groups = head.get('class_groups')
    require(isinstance(groups,list) and len(groups) >= n_classes, 'Missing GroupSum class_groups')
    intervals=[]
    for g in groups:
        c=g.get('class_index'); lo=g.get('node_start'); hi=g.get('node_stop_exclusive')
        require(all(type(x) is int for x in (c,lo,hi)), 'GroupSum class indices/ranges must be integers')
        require(0<=c<n_classes and 0<=lo<hi<=stage_sizes[-1], f'Invalid GroupSum interval: {g}')
        intervals.append((lo,hi,c))
    intervals.sort()
    require(all(intervals[i-1][1] <= intervals[i][0] for i in range(1,len(intervals))), 'Overlapping GroupSum intervals not supported')
    require(set(g[2] for g in intervals) == set(range(n_classes)), 'Each class must have at least one group interval')
    per_class=[sum(hi-lo for lo,hi,c in intervals if c==cls) for cls in range(n_classes)]
    class_w=bits_for_max_index(n_classes)
    score_w=max(16,max(per_class).bit_length())
    # Max address must be strictly smaller than the padding sentinel (all ones).
    # For max_length 16383, width 14 yields sentinel 16383 and max addr 16382.
    bank0_depth=max([n_input]+[stage_sizes[i] for i in range(1,len(stage_sizes)-1,2)])
    bank1_depth=max([1]+[stage_sizes[i] for i in range(0,len(stage_sizes)-1,2)])
    addr_w=max(2, max(bank0_depth,bank1_depth).bit_length())
    require(addr_w <= 30, 'Tensor too large for practical inferred on-chip memory')
    return dict(n_input=n_input,n_classes=n_classes,class_w=class_w,score_w=score_w,
                layers=compute,n_stages=len(compute),stage_sizes=stage_sizes,
                n_program=sum(stage_sizes),bank0_depth=bank0_depth,bank1_depth=bank1_depth,
                addr_w=addr_w,intervals=intervals,per_class=per_class)


def compile_instructions(spec: dict[str,Any]) -> list[tuple[int,tuple[int,...]]]:
    pad=(1<<spec['addr_w'])-1
    out=[]
    for layer in spec['layers']:
        if layer['op_type'] == 'LUT6Conv2D':
            cin,h,w = layer['input_shape']
            co,ho,wo = layer['output_shape']
            kh,kw = layer['kernel_size']
            sy,sx = layer['stride']
            py,px = layer['padding']
            shared = len(layer['init_hex']) == co
            for ch in range(co):
                for oy in range(ho):
                    for ox in range(wo):
                        idx = ch if shared else (ch*ho+oy)*wo+ox
                        addresses=[]
                        for p in layer['connections'][idx]:
                            c,k = divmod(p,kh*kw)
                            ky,kx=divmod(k,kw)
                            iy=oy*sy+ky-py
                            ix=ox*sx+kx-px
                            addr=(c*h+iy)*w+ix if 0<=iy<h and 0<=ix<w else pad
                            addresses.append(addr)
                        out.append((int(layer['init_hex'][idx],16),tuple(addresses)))
        else:
            for h,pins in zip(layer['init_hex'],layer['connections']):
                out.append((int(h,16),tuple(pins)))
    require(len(out)==spec['n_program'], 'Internal compiler: instruction count mismatch')
    return out


def encode_program(inst: list[tuple[int,tuple[int,...]]], addr_w: int, encoding: str) -> tuple[list[str],list[str],int,int]:
    pad=(1<<addr_w)-1
    if encoding == 'full':
        width=64+6*addr_w
        words=[]
        for init,addresses in inst:
            val=init
            for k,addr in enumerate(addresses):
                require(0<=addr<=pad, 'Address is too large')
                val|=addr<<(64+k*addr_w)
            words.append(f'{val:0{(width+3)//4}X}')
        return words,[],width,1
    require(encoding == 'split','Unknown encoding mode')
    # Truth tables are deduplicated; addressing microinstructions reference the table ROM.
    init_ids={}
    tables=[]
    for init,_ in inst:
        if init not in init_ids:
            init_ids[init]=len(tables)
            tables.append(init)
    id_w=bits_for_max_index(len(tables))
    width=6*addr_w+id_w
    words=[]
    for init,addresses in inst:
        val=init_ids[init] << (6*addr_w)
        for k,addr in enumerate(addresses):
            val|=addr<<(k*addr_w)
        words.append(f'{val:0{(width+3)//4}X}')
    table_hex=[f'{x:016X}' for x in tables]
    return words,table_hex,width,id_w


def decode_program(spec:dict[str,Any],prog:list[str],tables:list[str], encoding:str,id_w:int) -> list[tuple[int,tuple[int,...]]]:
    addr_w=spec['addr_w']; mask=(1<<addr_w)-1
    result=[]
    for word in prog:
        raw=int(word,16)
        if encoding == 'full':
            init=raw & ((1<<64)-1)
            pins=tuple((raw>>(64+i*addr_w))&mask for i in range(6))
        else:
            ti=raw>>(6*addr_w)
            require(ti<len(tables) and ti<(1<<id_w),'Invalid LUT truth ID')
            init=int(tables[ti],16)
            pins=tuple((raw>>(i*addr_w))&mask for i in range(6))
        result.append((init,pins))
    return result


def eval_compiled(spec:dict[str,Any],instr:list[tuple[int,tuple[int,...]]],bits:list[int]) -> list[int]:
    require(len(bits)==spec['n_input'], 'Wrong input length')
    stage_offset=0
    pad=(1<<spec['addr_w'])-1
    src=bits
    for n in spec['stage_sizes']:
        dst=[]
        for init,addrs in instr[stage_offset:stage_offset+n]:
            addr=0
            for k,x in enumerate(addrs):
                if x != pad:
                    addr|=src[x]<<k
            dst.append((init>>addr)&1)
        src=dst
        stage_offset+=n
    counts=[0]*spec['n_classes']
    for lo,hi,c in spec['intervals']:
        counts[c]+=sum(src[lo:hi])
    return counts


def eval_reference(ir:dict[str,Any], spec:dict[str,Any],bits:list[int]) -> list[int]:
    """Independent JSON evaluator: recomputes conv geometry, no packed ROM read."""
    src=bits
    for layer in spec['layers']:
        dst=[]
        if layer['op_type']=='LUT6Conv2D':
            cin,h,w=layer['input_shape']; co,ho,wo=layer['output_shape']
            kh,kw=layer['kernel_size']; sy,sx=layer['stride']; py,px=layer['padding']
            shared=len(layer['init_hex'])==co
            for ch in range(co):
                for oy in range(ho):
                    for ox in range(wo):
                        j=ch if shared else (ch*ho+oy)*wo+ox
                        address=0
                        for bit,p in enumerate(layer['connections'][j]):
                            channel=p//(kh*kw)
                            ky=(p%(kh*kw))//kw
                            kx=p%kw
                            yy=oy*sy+ky-py
                            xx=ox*sx+kx-px
                            if 0<=yy<h and 0<=xx<w:
                                address |= src[(channel*h+yy)*w+xx]<<bit
                        dst.append((int(layer['init_hex'][j],16)>>address)&1)
        else:
            for truth,pins in zip(layer['init_hex'],layer['connections']):
                address=sum(src[p]<<bit for bit,p in enumerate(pins))
                dst.append((int(truth,16)>>address)&1)
        src=dst
    counts=[0]*spec['n_classes']
    for lo,hi,c in spec['intervals']:
        counts[c]+=sum(src[lo:hi])
    return counts


def create_vectors(spec:dict[str,Any], ir:dict[str,Any], inst:list[tuple[int,tuple[int,...]]], out:Path, n_random:int, seed:int) -> list[list[int]]:
    rng=random.Random(seed)
    n=spec['n_input']
    patterns=[[0]*n,[1]*n]
    patterns.append([rng.getrandbits(1) for _ in range(n)])
    patterns.append([int((i*11+(i//17)*3)%19<9) for i in range(n)])
    for _ in range(n_random):
        patterns.append([rng.getrandbits(1) for _ in range(n)])
    vecdir=out/'vectors';vecdir.mkdir(parents=True,exist_ok=True)
    results=[]
    for i,v in enumerate(patterns):
        expected=eval_reference(ir,spec,v)
        observed=eval_compiled(spec,inst,v)
        require(expected==observed, f'CASE {i}: packed ROM and reference disagree: {expected} vs {observed}')
        results.append(expected)
        (vecdir/f'case_{i:03d}.mem').write_text(''.join(f'{x}\n' for x in v))
    # Bit-packed first class in least-significant score_w bits.
    packed=[]
    for values in results:
        raw=sum(v << (i*spec['score_w']) for i,v in enumerate(values))
        packed.append(f'{raw:0{(spec["n_classes"]*spec["score_w"]+3)//4}X}')
    (vecdir/'golden.mem').write_text('\n'.join(packed)+'\n')
    (vecdir/'golden.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


def render_rtl(spec:dict[str,Any], encoding:str, prog_width:int, init_id_w:int, out:Path) -> None:
    s=TEMPLATE.read_text()
    values={
      '__N_INPUT__':spec['n_input'], '__N_PROGRAM__':spec['n_program'],
      '__ADDR_W__':spec['addr_w'],'__PROG_W__':prog_width,
      '__N_STAGES__':spec['n_stages'],'__STAGE_W__':bits_for_max_index(spec['n_stages']),
      '__INDEX_W__':max(1,max(spec['stage_sizes']).bit_length()),
      '__PC_W__':bits_for_max_index(spec['n_program']),
      '__LOAD_W__':bits_for_max_index(spec['n_input']),
      '__BANK0_DEPTH__':spec['bank0_depth'],'__BANK1_DEPTH__':spec['bank1_depth'],
      '__N_CLASSES__':spec['n_classes'],'__CLASS_W__':spec['class_w'],
      '__SCORE_W__':spec['score_w'],
      '__USE_SPLIT__':1 if encoding=='split' else 0,
      '__INIT_ID_W__':init_id_w,
      '__N_TRUTH__':len(spec.get('truths',[])) or 1,
    }
    for key,value in values.items():s=s.replace(key,str(value))
    stage_w=bits_for_max_index(spec['n_stages']); index_w=max(1,max(spec['stage_sizes']).bit_length())
    stage_cases='\n'.join(f"            {stage_w}'d{i}: stage_length = {index_w}'d{n};" for i,n in enumerate(spec['stage_sizes']))
    s=s.replace('__STAGE_LENGTH_CASES__',stage_cases)
    class_w=spec['class_w']
    class_cases='\n'.join(f"            if ((idx >= {index_w}'d{lo}) && (idx < {index_w}'d{hi})) class_of_index = {{1'b1, {class_w}'d{c}}};" for lo,hi,c in spec['intervals'])
    s=s.replace('__CLASS_CASES__',class_cases)
    require('__N_INPUT__' not in s and '__CLASS_CASES__' not in s and '__STAGE_LENGTH_CASES__' not in s, 'Unreplaced template placeholder')
    (out/'rtl'/'lut6_sequential.sv').write_text(s)


def render_tb(spec:dict[str,Any], expected: list[list[int]], out:Path) -> None:
    n=spec['n_input']; scores_w=spec['n_classes']*spec['score_w']; ntests=len(expected)
    watchdog_ns=(n*2+(spec['expected_cycles']+20)*ntests+10000)*10+100000
    tb=f'''`timescale 1ns/1ps
`default_nettype none
// Auto-generated, reference-checked end-to-end testbench.
module tb_lut6_sequential;
    localparam integer N_INPUT={n}, SCORE_BITS={scores_w}, TESTS={ntests};
    reg clk=1'b0;
    always #5 clk=~clk;
    reg rst=1'b1, input_valid=1'b0, input_bit=1'b0, start=1'b0;
    wire input_ready, frame_ready, busy, done;
    wire [SCORE_BITS-1:0] scores_flat;
    reg [0:0] vector_bits [0:N_INPUT-1];
    reg [SCORE_BITS-1:0] golden [0:TESTS-1];
    integer i,j,cycles;
    lut6_sequential #(.PROGRAM_FILE("rtl/lut6_program.mem"),
                      .TRUTH_FILE("rtl/lut6_truth.mem")) dut (
       .clk(clk),.rst(rst),.input_valid(input_valid),.input_bit(input_bit),
       .input_ready(input_ready),.frame_ready(frame_ready),.start(start),
       .busy(busy),.done(done),.scores_flat(scores_flat));
    initial begin
       $readmemh("vectors/golden.mem", golden);
       repeat(5) @(negedge clk);
       rst=1'b0;
       // A start pulse before the complete frame must be ignored.
       start=1'b1;
       @(negedge clk);
       start=1'b0;
       if(busy !== 1'b0 || input_ready !== 1'b1)
           $fatal(1,"Premature start was not ignored");
       for(j=0;j<TESTS;j=j+1) begin
          case(j)
__VECTOR_CASES__
          endcase
          for(i=0;i<N_INPUT;i=i+1) begin
             // Deliberate input_valid gaps exercise streaming backpressure.
             if ((i % 53) == 0) begin
                @(negedge clk);
                input_valid=1'b0;
             end
             @(negedge clk);
             if (input_ready !== 1'b1) $fatal(1,"input_ready low case=%0d bit=%0d",j,i);
             input_valid=1'b1;
             input_bit=vector_bits[i];
          end
          @(negedge clk);
          input_valid=1'b0;
          if(frame_ready !== 1'b1) $fatal(1,"frame_ready missing case=%0d",j);
          start=1'b1;
          @(negedge clk);
          start=1'b0;
          if(busy !== 1'b1) $fatal(1,"busy missing case=%0d",j);
          cycles=0;
          while (done !== 1'b1 && cycles < {spec['expected_cycles']+10}) begin
             @(negedge clk);
             cycles=cycles+1;
          end
          if(done !== 1'b1) $fatal(1,"Timeout waiting for done case=%0d",j);
          if(scores_flat !== golden[j]) $fatal(1,"Score mismatch case=%0d actual=%h expected=%h",j,scores_flat,golden[j]);
          if(cycles != {spec['expected_cycles']}) $fatal(1,"Cycles mismatch case=%0d actual=%0d expected=%0d",j,cycles,{spec['expected_cycles']});
          $display("PASS case=%0d exact_scores=%h cycles=%0d",j,scores_flat,cycles);
          @(negedge clk);
          if (done !== 1'b0) $fatal(1,"done did not deassert after one cycle");
       end
       $display("ALL_TESTS_PASS cases=%0d",TESTS);
       $finish;
    end
    initial begin
       #(64'd{watchdog_ns});
       $fatal(1,"Global testbench watchdog");
    end
endmodule
`default_nettype wire
'''
    file_cases='\n'.join(f'             {j}: $readmemb("vectors/case_{j:03d}.mem",vector_bits);' for j in range(ntests))
    tb=tb.replace('__VECTOR_CASES__',file_cases)
    (out/'sim'/'tb_lut6_sequential.sv').write_text(tb)


def write_support(out: Path, clock_ns:float) -> None:
    (out/'timing.xdc').write_text(f'create_clock -name core_clk -period {clock_ns:g} [get_ports clk]\n')
    (out/'vivado').mkdir(exist_ok=True)
    (out/'vivado'/'implement.tcl').write_text('''# Usage: vivado -mode batch -source vivado/implement.tcl -tclargs <part> <tag>
# Placement/routing feasibility: no board pins or I/O timing are defined.
set root [file normalize [file join [file dirname [info script]] ..]]
set part [lindex $argv 0]
set tag [lindex $argv 1]
if {$part eq ""} { set part xczu3cg-sfvc784-1-e }
if {$tag eq ""} { set tag default }
set projdir [file join $root vivado_build_$tag]
set outdir [file join $root reports_$tag]
file mkdir $outdir
create_project -force "lut6_bridge_$tag" $projdir -part $part
add_files -norecurse [file join $root rtl lut6_sequential.sv]
add_files -norecurse [file join $root rtl lut6_program.mem]
set_property file_type {Memory Initialization Files} [get_files [file join $root rtl lut6_program.mem]]
if {[file exists [file join $root rtl lut6_truth.mem]]} {
    add_files -norecurse [file join $root rtl lut6_truth.mem]
    set_property file_type {Memory Initialization Files} [get_files [file join $root rtl lut6_truth.mem]]
}
set_property top lut6_sequential [get_filesets sources_1]
add_files -fileset constrs_1 -norecurse [file join $root timing.xdc]
update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "Synthesis failed; inspect synth_1/runme.log" }
open_run synth_1
report_utilization -file [file join $outdir synthesis_utilization.rpt]
opt_design
place_design
write_checkpoint -force [file join $outdir placed.dcp]
report_timing_summary -file [file join $outdir placed_timing.rpt]
catch {report_design_analysis -congestion -file [file join $outdir congestion.rpt]}
route_design
write_checkpoint -force [file join $outdir routed.dcp]
report_route_status -file [file join $outdir route_status.rpt]
report_timing_summary -file [file join $outdir timing.rpt]
report_utilization -hierarchical -file [file join $outdir utilization.rpt]
report_drc -file [file join $outdir drc.rpt]
report_power -file [file join $outdir power.rpt]
set routed [report_route_status -boolean_check ROUTED_FULLY]
set errors [report_route_status -boolean_check ERRORS_IN_ROUTES]
if {!$routed || $errors} { error "Routing incomplete: ROUTED_FULLY=$routed ERRORS=$errors" }
puts "ROUTE_PASS. Check post-route timing.rpt, DRC and I/O constraints."
''')
    (out/'run_sim.ps1').write_text('''param([string]$VivadoBin="C:\\Xilinx\\Vivado\\2022.2\\bin")
$ErrorActionPreference="Stop"
Set-Location $PSScriptRoot
& (Join-Path $VivadoBin "xvlog.bat") -sv .\\rtl\\lut6_sequential.sv .\\sim\\tb_lut6_sequential.sv
if ($LASTEXITCODE -ne 0) { throw "xvlog failed" }
& (Join-Path $VivadoBin "xelab.bat") tb_lut6_sequential -s bridge_tb_snapshot
if ($LASTEXITCODE -ne 0) { throw "xelab failed" }
& (Join-Path $VivadoBin "xsim.bat") bridge_tb_snapshot -runall -log simulation.log
if ($LASTEXITCODE -ne 0) { throw "xsim failed" }
if (-not (Select-String -Path simulation.log -Pattern 'ALL_TESTS_PASS' -Quiet)) { throw "PASS marker not found" }
Write-Host "SIMULATION_PASS" -ForegroundColor Green
''')
    (out/'run_impl.ps1').write_text('''param(
    [string]$Part="xczu3cg-sfvc784-1-e",
    [string]$Tag="zu3cg",
    [string]$VivadoBin="C:\\Xilinx\\Vivado\\2022.2\\bin"
)
$ErrorActionPreference="Stop"
Set-Location $PSScriptRoot
& (Join-Path $VivadoBin "vivado.bat") -mode batch -source .\\vivado\\implement.tcl -tclargs $Part $Tag
if ($LASTEXITCODE -ne 0) { throw "Vivado implementation failed. Inspect the logs." }
Write-Host "Implementation Tcl completed; inspect reports_$Tag for timing/DRC." -ForegroundColor Green
''')
    (out/'run_sim.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
xvlog -sv rtl/lut6_sequential.sv sim/tb_lut6_sequential.sv
xelab tb_lut6_sequential -s bridge_tb_snapshot
xsim bridge_tb_snapshot -runall -log simulation.log
grep -q 'ALL_TESTS_PASS' simulation.log
''')
    (out/'run_sim.sh').chmod(0o755)


def build(ir_path:Path,out:Path,encoding:str='full',n_random:int=2,seed:int=42,clock_ns:float=10.0,copy_ir:bool=False) -> dict[str,Any]:
    require(clock_ns>0,'Clock period must be positive')
    require(n_random>=0,'n_random must be nonnegative')
    ir=json.loads(ir_path.read_text())
    spec=validated_model(ir)
    inst=compile_instructions(spec)
    prog,truth,prog_width,id_width=encode_program(inst,spec['addr_w'],encoding)
    require(inst==decode_program(spec,prog,truth,encoding,id_width),'Microcode encode/decode mismatch')
    pad=(1<<spec['addr_w'])-1
    pad_count=sum(sum(x==pad for x in pins) for _,pins in inst)
    # Full: FETCH, ROM_WAIT, 6 pin states, EXEC = 9 + 2*real_reads;
    # split: one extra TRUTH_WAIT state per instruction.
    expected_cycles = len(inst)*21 - 2*pad_count + (len(inst) if encoding=='split' else 0)
    spec['expected_cycles']=expected_cycles
    spec['truths']=truth
    for folder in ('rtl','sim','vectors'):(out/folder).mkdir(parents=True,exist_ok=True)
    (out/'rtl'/'lut6_program.mem').write_text('\n'.join(prog)+'\n')
    if encoding=='split':
        (out/'rtl'/'lut6_truth.mem').write_text('\n'.join(truth)+'\n')
    else:
        (out/'rtl'/'lut6_truth.mem').unlink(missing_ok=True)
    render_rtl(spec,encoding,prog_width,id_width,out)
    expected=create_vectors(spec,ir,inst,out,n_random,seed)
    render_tb(spec,expected,out)
    write_support(out,clock_ns)
    if copy_ir:shutil.copyfile(ir_path,out/'model_ir.json')
    raw=(out/'rtl'/'lut6_program.mem').read_bytes()
    manifest={
      'input_shape':ir['input']['shape'],'input_bits':spec['n_input'],
      'layer_names':[layer['name'] for layer in spec['layers']],
      'layer_shapes':[layer['output_shape'] for layer in spec['layers']],
      'stage_instruction_counts':spec['stage_sizes'],
      'n_classes':spec['n_classes'],'class_ranges':spec['intervals'],
      'score_width_bits':spec['score_w'],'address_width_bits':spec['addr_w'],
      'bank_depths':[spec['bank0_depth'],spec['bank1_depth']],
      'encoding':encoding,'program_width_bits':prog_width,
      'program_instructions':len(inst),'rom_bits_theoretical':len(inst)*prog_width + len(truth)*64,
      'truth_entries':len(truth),'truth_id_bits':id_width,
      'padding_pins':pad_count,'inference_core_cycles':expected_cycles,
      'clock_period_ns_target':clock_ns,'reference_outputs':expected,
      'microprogram_sha256':hashlib.sha256(raw).hexdigest(),
      'notes':['Raw RGB encoder is not included','Physical timing and routing require Vivado','Only linear LUT6Conv2D/LUT6Dense + BooleanGroupSum models supported'],
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'PASS reference-vs-ROM: {len(expected)} patterns; first scores={expected[0]}')
    print(f'Generated: {out.resolve()}')
    print(f'{len(inst)} LUT evaluations; {encoding} ROM: {len(inst)}x{prog_width} bits; {len(truth)} truth entries')
    print(f'Core clocks (per frame): {expected_cycles}; input bits: {spec["n_input"]}; scores: {spec["n_classes"]}')
    return manifest


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ir',type=Path,required=True,help='Input hls4ml_lut6_ir_v1 JSON')
    p.add_argument('--out',type=Path,required=True,help='New output folder (generated FPGA project)')
    p.add_argument('--encoding',choices=['full','split'],default='full',help='full = known baseline layout; split = deduplicated truth table ROM')
    p.add_argument('--random-tests',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--clock-ns',type=float,default=10.0)
    p.add_argument('--copy-ir',action='store_true')
    a=p.parse_args()
    try:build(a.ir,a.out,a.encoding,a.random_tests,a.seed,a.clock_ns,a.copy_ir)
    except (IRValidationError,KeyError,TypeError) as e:p.error(str(e))

if __name__=='__main__':main()
