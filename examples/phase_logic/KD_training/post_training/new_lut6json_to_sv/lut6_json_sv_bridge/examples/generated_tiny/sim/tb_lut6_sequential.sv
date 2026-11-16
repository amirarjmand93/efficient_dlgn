`timescale 1ns/1ps
`default_nettype none
// Auto-generated, reference-checked end-to-end testbench.
module tb_lut6_sequential;
    localparam integer N_INPUT=32, SCORE_BITS=32, TESTS=5;
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
             0: $readmemb("vectors/case_000.mem",vector_bits);
             1: $readmemb("vectors/case_001.mem",vector_bits);
             2: $readmemb("vectors/case_002.mem",vector_bits);
             3: $readmemb("vectors/case_003.mem",vector_bits);
             4: $readmemb("vectors/case_004.mem",vector_bits);
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
          while (done !== 1'b1 && cycles < 955) begin
             @(negedge clk);
             cycles=cycles+1;
          end
          if(done !== 1'b1) $fatal(1,"Timeout waiting for done case=%0d",j);
          if(scores_flat !== golden[j]) $fatal(1,"Score mismatch case=%0d actual=%h expected=%h",j,scores_flat,golden[j]);
          if(cycles != 945) $fatal(1,"Cycles mismatch case=%0d actual=%0d expected=%0d",j,cycles,945);
          $display("PASS case=%0d exact_scores=%h cycles=%0d",j,scores_flat,cycles);
          @(negedge clk);
          if (done !== 1'b0) $fatal(1,"done did not deassert after one cycle");
       end
       $display("ALL_TESTS_PASS cases=%0d",TESTS);
       $finish;
    end
    initial begin
       #(64'd248890);
       $fatal(1,"Global testbench watchdog");
    end
endmodule
`default_nettype wire
