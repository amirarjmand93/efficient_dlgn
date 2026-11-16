# Usage: vivado -mode batch -source vivado/implement.tcl -tclargs <part> <tag>
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
