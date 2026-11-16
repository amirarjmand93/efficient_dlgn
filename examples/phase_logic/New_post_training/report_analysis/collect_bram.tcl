# vivado -mode batch -source collect_bram.tcl -tclargs <routed_design.dcp> <reports_dir>
set checkpoint [lindex $argv 0]
set reports_dir [lindex $argv 1]
if {$checkpoint eq "" || $reports_dir eq ""} {
    error "Supply routed_design.dcp and its reports directory"
}
open_checkpoint $checkpoint
file mkdir $reports_dir
report_utilization -hierarchical -file [file join $reports_dir routed_hierarchical_utilization.rpt]
set ram_file [open [file join $reports_dir routed_bram_instances.csv] w]
puts $ram_file "instance,primitive"
foreach cell [get_cells -hierarchical -filter {REF_NAME =~ RAMB36* || REF_NAME =~ RAMB18*}] {
    puts $ram_file "[get_property NAME $cell],[get_property REF_NAME $cell]"
}
close $ram_file
