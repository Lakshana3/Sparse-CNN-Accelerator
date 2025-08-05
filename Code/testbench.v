`timescale 1ns/1ps

module testbench;
    // Clock and reset signals
    reg clk;
    reg reset_async;

    // Clock generation: 100 MHz
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // Toggle every 5 ns for 100 MHz
    end

    // Inputs to DUT
    reg  [15:0]  input_valid_bus;
    reg  [639:0] weight_mask_buffer;
    reg  [511:0] activation_buffer;

    // Outputs from DUT
    wire [2303:0] result_buffer;
    wire [127:0]  valid_buffer;

    // Instantiate Device Under Test (DUT)
    sparse_cnn_accelerator_top dut (
      .clk(clk),
      .reset_async(reset_async),
      .input_valid_bus(input_valid_bus),
      .weight_mask_buffer(weight_mask_buffer),
      .activation_buffer(activation_buffer),
      .result_buffer(result_buffer),
      .valid_buffer(valid_buffer)
    );

    // Test stimulus and monitoring block
    initial begin
        $dumpfile("testbench.vcd");
        $dumpvars(0, testbench);

        // Initial reset asserted and inputs cleared
        reset_async = 1;
        input_valid_bus = 0;
        weight_mask_buffer = 0;
        activation_buffer = 0;

        #20;  // Hold reset for 20 ns
        reset_async = 0;  // Release reset
        #20;

        // Enable all 16 engines
        input_valid_bus = 16'hFFFF;
        // Setup activation buffer with a known pattern repeated for 8 banks (each 64 bits)
        activation_buffer[63:0]     = 64'h0706050403020100;
        activation_buffer[127:64]   = 64'h0706050403020100;
        activation_buffer[191:128]  = 64'h0706050403020100;
        activation_buffer[255:192]  = 64'h0706050403020100;
        activation_buffer[319:256]  = 64'h0706050403020100;
        activation_buffer[383:320]  = 64'h0706050403020100;
        activation_buffer[447:384]  = 64'h0706050403020100;
        activation_buffer[511:448]  = 64'h0706050403020100;
        // Setup weight_mask_buffer masks with increasing density patterns for engines
        weight_mask_buffer[39:0]     = {32'h01010101, 8'b00000000};
        weight_mask_buffer[79:40]    = {32'h01010101, 8'b00000001};
        weight_mask_buffer[119:80]   = {32'h01010101, 8'b00000011};
        weight_mask_buffer[159:120]  = {32'h01010101, 8'b00000111};
        weight_mask_buffer[199:160]  = {32'h01010101, 8'b00001111};
        weight_mask_buffer[239:200]  = {32'h01010101, 8'b00011111};
        weight_mask_buffer[279:240]  = {32'h01010101, 8'b00111111};
        weight_mask_buffer[319:280]  = {32'h01010101, 8'b01111111};
        weight_mask_buffer[359:320]  = {32'h01010101, 8'b11111111};
        weight_mask_buffer[399:360]  = {32'h01010101, 8'b01010101};
        weight_mask_buffer[439:400]  = {32'h01010101, 8'b10101010};
        weight_mask_buffer[479:440]  = {32'h01010101, 8'b11110000};
        weight_mask_buffer[519:480]  = {32'h01010101, 8'b00001111};
        weight_mask_buffer[559:520]  = {32'h01010101, 8'b00110011};
        weight_mask_buffer[599:560]  = {32'h01010101, 8'b11001100};
        weight_mask_buffer[639:600]  = {32'h01010101, 8'b00111100};
        // Temporarily disable engines
      	#30;
        input_valid_bus = 16'h0000;
        // Monitor output for 10 clock cycles
        repeat (10) begin
            @(posedge clk);
            $display("[time=%0t] out_valid=%b out_result=%0d", $time, valid_buffer, result_buffer);
        end

        #20;

        // Enable all engines again
        input_valid_bus = 16'hFFFF;
        // Set weight mask buffer with a different pattern (dense weights example)
        weight_mask_buffer[39:0]     = {32'h11110000, 8'b00000000};
        weight_mask_buffer[79:40]    = {32'h11110000, 8'b00000001};
        weight_mask_buffer[119:80]   = {32'h11110000, 8'b00000011};
        weight_mask_buffer[159:120]  = {32'h11110000, 8'b00000111};
        weight_mask_buffer[199:160]  = {32'h11110000, 8'b00001111};
        weight_mask_buffer[239:200]  = {32'h11110000, 8'b00011111};
        weight_mask_buffer[279:240]  = {32'h11110000, 8'b00111111};
        weight_mask_buffer[319:280]  = {32'h11110000, 8'b01111111};
        weight_mask_buffer[359:320]  = {32'h11110000, 8'b11111111};
        weight_mask_buffer[399:360]  = {32'h11110000, 8'b01010101};
        weight_mask_buffer[439:400]  = {32'h11110000, 8'b10101010};
        weight_mask_buffer[479:440]  = {32'h11110000, 8'b11110000};
        weight_mask_buffer[519:480]  = {32'h11110000, 8'b00001111};
        weight_mask_buffer[559:520]  = {32'h11110000, 8'b00110011};
        weight_mask_buffer[599:560]  = {32'h11110000, 8'b11001100};
        weight_mask_buffer[639:600]  = {32'h11110000, 8'b00111100};
      	#30;
        // Disable engines again
        input_valid_bus = 16'h0000;
        // Monitor output for 10 clock cycles
        repeat (10) begin
            @(posedge clk);
            $display("[time=%0t] out_valid=%b out_result=%0d", $time, valid_buffer, result_buffer);
        end

        // End simulation
        $finish;
    end
endmodule

