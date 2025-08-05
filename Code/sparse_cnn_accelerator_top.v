// Top-level for 16 engines × 8 blocks ASIC implementation
//results appear after 7 clk cycles exactly at same cycle as valid (without the 1-cycle hold requirement)
module sparse_cnn_accelerator_top (
    input  wire         clk,
    input  wire         reset_async,
    input  wire [15:0]  input_valid_bus,
    input  wire [639:0] weight_mask_buffer,
    input  wire [511:0] activation_buffer,
    output wire [2303:0] result_buffer,
    output wire [127:0] valid_buffer
);

  // Engine connections remain unchanged
  wire [2303:0] raw_result_buffer;
  wire [127:0] raw_valid_buffer;
  
  // Convert flat activation buffer to banked structure
  wire [63:0] activation_banks [0:7];
  generate
    genvar b;
    for (b = 0; b < 8; b = b+1) begin : bank_split
      assign activation_banks[b] = activation_buffer[64*b +:64];
    end
  endgenerate

  genvar e;
  generate
    for (e = 0; e < 16; e = e+1) begin : engines
      sparse_cnn_engine engine_i (
        .clk(clk),
        .reset_async(reset_async),
        .input_valid(input_valid_bus[e]),
        .weight_mask(weight_mask_buffer[40*e +: 40]),
        //.activation_buffer(activation_buffer),
        .activation_buffer(activation_banks),  // Connect banked structure
        .result_bus(raw_result_buffer[144*e +: 144]), 
        .valid_bus(raw_valid_buffer[8*e +: 8])
      );
    end
  endgenerate

  // Simple 1-cycle valid delay register
  reg valid_delayed;
  always @(posedge clk or posedge reset_async) begin
    if (reset_async) valid_delayed <= 0;
    else valid_delayed <= |raw_valid_buffer;
  end

  // Output assignments
  assign result_buffer = raw_result_buffer; // Direct connection
  assign valid_buffer = {128{valid_delayed}}; // Expanded delayed valid
endmodule

//------------------------------------------------------------------------------
// Single engine: 8 independent sparse CNN blocks
//------------------------------------------------------------------------------
module sparse_cnn_engine (
    input  wire         clk,
    input  wire         reset_async,
    input  wire         input_valid,
    input  wire [39:0]  weight_mask,
    //input  wire [511:0] activation_buffer,  // 8 × 64-bit activations
  	input  wire [63:0]  activation_buffer [0:7],  // Banked interface
    output wire [143:0] result_bus,         // 8 × 18-bit results
    output wire [7:0]   valid_bus
);
  
  // MODIFIED: Shared Resources for all 8 blocks
  // Shared Booth Pre-Encoder for all 8 blocks (Separates Weights/Masks)
  wire [47:0] shared_booth_enc;
  booth_pre_encoder encoder (
      .clk(clk),
      .reset(reset_async),
      .weight(weight_mask[39:8]),  // Extract 32b weight
      .booth_encoded(shared_booth_enc)
  );
  
  // MODIFIED: Shared mask pipeline register
  reg [7:0] shared_mask_reg;
  always @(posedge clk or posedge reset_async) begin
      if (reset_async) shared_mask_reg <= 0;
      else if (input_valid) shared_mask_reg <= weight_mask[7:0]; // Extract 8b mask
  end
  
  // MODIFIED: Block Instances with shared resources
  genvar b;
  generate
    for (b = 0; b < 8; b = b+1) begin : blocks
      sparse_cnn_accelerator_single blk (
        .clk           (clk),
        .reset_async   (reset_async),
        .input_valid   (input_valid),
        .shared_mask(shared_mask_reg),      // MODIFIED: Shared mask
        .shared_booth_enc(shared_booth_enc), // MODIFIED: Shared encoding
        .banked_activations(activation_buffer[b]),  // Direct bank access
        //.activations(activation_buffer[64*b +:64]),
        .result(result_bus[18*b +:18]),
        .output_valid(valid_bus[b])
      );
    end
  endgenerate
endmodule
// ---------sparse_cnn_accelerator_single------//
module sparse_cnn_accelerator_single (
  input wire clk,
  input wire reset_async,
  input wire input_valid,
  input  wire [7:0]   shared_mask,      // MODIFIED: Shared mask (8b) from engine
  input  wire [47:0]  shared_booth_enc, // MODIFIED: Shared encoded weights (48b) from engine
  //input wire [63:0] activations, // 64b activations (act7..act0)
  input wire [63:0] banked_activations,  // Single bank input
  output wire [17:0] result, // 18b result
  output wire output_valid
);
  // 1. Reset Synchronization (2-stage)
  reg rst_meta, rst_sync;
  always @(posedge clk or posedge reset_async) begin
    if (reset_async) {rst_meta,rst_sync} <= 2'b11;
    else             {rst_meta,rst_sync} <= {1'b0, rst_meta};
  end
  wire reset = rst_sync;

  // 2. Input Latching (on valid strobe)
  reg [63:0] act_reg;
  reg        stage1_valid;
  always @(posedge clk) begin
    if (reset) begin
      act_reg       <= 0;
      stage1_valid  <= 0;
    end else begin
      stage1_valid  <= input_valid;
      if (input_valid) begin
        //act_reg       <= activations;
        act_reg <= banked_activations;  // Direct bank access
      end
    end
  end

  // MODIFIED: Uses shared mask instead of extracting from weight_mask
  wire [3:0] pop_count = shared_mask[0] + shared_mask[1] + shared_mask[2] + shared_mask[3] + 
                        shared_mask[4] + shared_mask[5] + shared_mask[6] + shared_mask[7];
  wire sparsity_ge_50 = (pop_count <= 4);

  // 4. Activation Selector (pipelined, robust: see below)
  wire [31:0] selected_activations;
  wire        selector_done;
  
  activation_selector act_sel (
    .clk(clk),
    .reset(reset),
    .mask(shared_mask), // MODIFIED: Uses shared mask
    .activations(act_reg),
    .selected_act(selected_activations),
    .done(selector_done)
  );
  
  // 6. Valid Pipeline (for output valid alignment)
  // 7‑stage valid pipeline:
  reg [5:0] vpipe;
  always @(posedge clk) begin
    if (reset) vpipe <= 0;
    else       vpipe <= {vpipe[4:0], stage1_valid};
  end

  // MODIFIED: Block MAC with registered output
  wire [17:0] mac_result;
  wire        mac_valid;
  block_mac_unit blk_mac (
    .clk(clk),
    .reset(reset),
    .mask(shared_mask), // MODIFIED: Uses shared mask
    .booth_encoded(shared_booth_enc), // MODIFIED: Uses shared encoding
    .selected_activations(selected_activations),
    .selector_done(selector_done),
    .sparsity_ge_50(sparsity_ge_50),
    .result(mac_result),
    .valid(mac_valid)
  );

  // 8. Output Registering and Assignments
  reg [17:0] result_reg;
  always @(posedge clk) begin
    if (reset) begin
      result_reg      <= 0;
    end else if(vpipe[5]) begin
      // Register the MAC result
      result_reg      <= mac_result;
    end
  end
  assign result       = result_reg;
  assign output_valid = vpipe[5];

endmodule

// ========== Booth Encoder (pipelined output) ==========
module booth_pre_encoder (
  input  wire        clk,
  input  wire        reset,
  input  wire [31:0] weight,
  output reg  [47:0] booth_encoded
);
  function [2:0] booth_encode(input [2:0] bits);
    case (bits)
      3'b000: booth_encode = 3'b100; // 0
      3'b001: booth_encode = 3'b001; // +1
      3'b010: booth_encode = 3'b001; // +1
      3'b011: booth_encode = 3'b000; // +2
      3'b100: booth_encode = 3'b010; // -2
      3'b101: booth_encode = 3'b011; // -1
      3'b110: booth_encode = 3'b011; // -1
      3'b111: booth_encode = 3'b100; // 0
    endcase
  endfunction

  always @(posedge clk) begin
    if (reset) booth_encoded <= 48'd0;
    else begin
      booth_encoded[2:0]    <= booth_encode({weight[1:0], 1'b0});
      booth_encoded[5:3]    <= booth_encode(weight[3:1]);
      booth_encoded[8:6]    <= booth_encode(weight[5:3]);
      booth_encoded[11:9]   <= booth_encode(weight[7:5]);
      booth_encoded[14:12]  <= booth_encode({weight[9:8], 1'b0});
      booth_encoded[17:15]  <= booth_encode(weight[11:9]);
      booth_encoded[20:18]  <= booth_encode(weight[13:11]);
      booth_encoded[23:21]  <= booth_encode(weight[15:13]);
      booth_encoded[26:24]  <= booth_encode({weight[17:16], 1'b0});
      booth_encoded[29:27]  <= booth_encode(weight[19:17]);
      booth_encoded[32:30]  <= booth_encode(weight[21:19]);
      booth_encoded[35:33]  <= booth_encode(weight[23:21]);
      booth_encoded[38:36]  <= booth_encode({weight[25:24], 1'b0});
      booth_encoded[41:39]  <= booth_encode(weight[27:25]);
      booth_encoded[44:42]  <= booth_encode(weight[29:27]);
      booth_encoded[47:45]  <= booth_encode(weight[31:29]);
    end
  end
endmodule

// ========== MAC Unit with Partial Product Generator ==========
module generate_pprow (
  input  wire [2:0] ctrl,
  input  wire [7:0] x,
  output reg  [8:0] pp
);
  wire signed [8:0] sx = {x[7], x}; // Sign-extend

  always @(*) begin
    case (ctrl)
      3'b100: pp = 9'd0;         // 0
      3'b001: pp = sx;           // +1
      3'b000: pp = sx << 1;      // +2
      3'b010: pp = -(sx << 1);   // -2
      3'b011: pp = -sx;          // -1
      default: pp = 9'd0;
    endcase
  end
endmodule

module mac_unit (
    input  wire        clk,
    input  wire        reset,
    input  wire [11:0] enc,       // 4x Radix-4 Booth encodings (3 bits each)
    input  wire [7:0]  act,       // 8-bit activation
    output wire signed [16:0] result  // 17-bit output (for 18-bit accumulation)
);
    // ================================================
    // Clock Gating Logic (Booth-Zero Aware)
    // ================================================
    // MODIFIED: Correct Booth-zero detection
    function automatic logic is_booth_active(input [2:0] booth);
        return (booth != 3'b100);  // '100' = zero in Radix-4 Booth
    endfunction

    wire mac_clk_en = is_booth_active(enc[2:0])   ||  // Lane 0
                      is_booth_active(enc[5:3])   ||  // Lane 1
                      is_booth_active(enc[8:6])   ||  // Lane 2
                      is_booth_active(enc[11:9]);     // Lane 3

    (* gclk_enable = "mac_clk_en" *)
    wire gated_clk;
    assign gated_clk = clk & mac_clk_en;

    // MODIFIED: Partial Product Generation with gated clock
    wire [8:0] pp0, pp1, pp2, pp3;
    generate_pprow pp0_gen(.ctrl(enc[2:0]), .x(act), .pp(pp0));
    generate_pprow pp1_gen(.ctrl(enc[5:3]), .x(act), .pp(pp1));
    generate_pprow pp2_gen(.ctrl(enc[8:6]), .x(act), .pp(pp2));
    generate_pprow pp3_gen(.ctrl(enc[11:9]), .x(act), .pp(pp3));

    // MODIFIED: Wallace Tree Addition with registered output
    wire signed [16:0] sp0 = $signed(pp0);
    wire signed [16:0] sp1 = $signed(pp1) <<< 2;
    wire signed [16:0] sp2 = $signed(pp2) <<< 4;
    wire signed [16:0] sp3 = $signed(pp3) <<< 6;

    // Stage 1: CSA
    wire [16:0] s1 = sp0 ^ sp1 ^ sp2;
    wire [16:0] c1 = ((sp0 & sp1) | (sp0 & sp2) | (sp1 & sp2)) << 1;

    // Stage 2: CSA + Final CPA
    wire [16:0] s2 = s1 ^ c1 ^ sp3;
    wire [16:0] c2 = ((s1 & c1) | (s1 & sp3) | (c1 & sp3)) << 1;

    // MODIFIED: Register outpck
    reg signed [16:0] result_reg;
  always @(posedge gated_clk) begin // MAC result registers (clock-gated domain) sync reset
        if (reset) begin
            result_reg <= 0;
        end else begin
            result_reg <= s2 + c2;  // Final CPA
        end
    end

    assign result = result_reg;
endmodule

// ========== Activation Selector ==========
module activation_selector (
  input  wire        clk,
  input  wire        reset,
  input  wire [7:0]  mask,
  input  wire [63:0] activations,
  output reg  [31:0] selected_act,
  output reg         done
);
  //reg [7:0] mask_reg;
  reg [63:0] act_reg;
  always @(posedge clk) begin
    if (reset) begin
      //mask_reg <= 0;
      act_reg  <= 0;
    end else begin
      //mask_reg <= mask;
      act_reg  <= activations;
    end
  end

  function [2:0] pe(input [7:0] m); // priority encoder
    casez (m)
      8'b???????1: pe = 3'd0;
      8'b??????10: pe = 3'd1;
      8'b?????100: pe = 3'd2;
      8'b????1000: pe = 3'd3;
      8'b???10000: pe = 3'd4;
      8'b??100000: pe = 3'd5;
      8'b?1000000: pe = 3'd6;
      8'b10000000: pe = 3'd7;
      default: pe = 3'd0;
    endcase
  endfunction

  reg        cycle;
  reg [7:0]  rem_mask;

  //wire [3:0] popcount = mask_reg[0]+mask_reg[1]+mask_reg[2]+mask_reg[3]+mask_reg[4]+mask_reg[5]+mask_reg[6]+mask_reg[7];
  wire [3:0] popcount = mask[0]+mask[1]+mask[2]+mask[3]+mask[4]+mask[5]+mask[6]+mask[7];
  wire       sparse   = (popcount <= 4);

  //wire [7:0] sel_mask = (sparse || !cycle) ? mask_reg : rem_mask;.
  wire [7:0] sel_mask = (sparse || !cycle) ? mask : rem_mask;

  wire [2:0] i0 = pe(sel_mask);
  wire [7:0] m1 = sel_mask & ~(8'd1 << i0);
  wire [2:0] i1 = pe(m1);
  wire [7:0] m2 = m1      & ~(8'd1 << i1);
  wire [2:0] i2 = pe(m2);
  wire [7:0] m3 = m2      & ~(8'd1 << i2);
  wire [2:0] i3 = pe(m3);

  wire [7:0] a0 = act_reg[i0*8 +:8];
  wire [7:0] a1 = act_reg[i1*8 +:8];
  wire [7:0] a2 = act_reg[i2*8 +:8];
  wire [7:0] a3 = act_reg[i3*8 +:8];

  always @(posedge clk) begin
    if (reset) begin
      selected_act <= 0;
      done         <= 0;
      cycle        <= 0;
      rem_mask     <= 0;
    end else if (sparse) begin
      // Single cycle operation: output selection
      selected_act <= {a3,a2,a1,a0};
      done         <= 1;
      cycle        <= 0;
    end else if (!cycle) begin // first half dense
      selected_act <= {a3,a2,a1,a0};
      rem_mask     <= m3 & ~(8'd1 << i3);
      done         <= 1;
      cycle        <= 1;
    end else begin // second half dense
      selected_act <= {a3,a2,a1,a0};
      done         <= 1;
      cycle        <= 0;
    end
  end
endmodule

// ========== Block MAC Unit ==========
// Always update inputs on selector_done; only asserts valid after the second cycle if low sparsity
module block_mac_unit (
  input  wire        clk,
  input  wire        reset,
  input  wire [7:0]  mask,
  input  wire [47:0] booth_encoded,
  input  wire [31:0] selected_activations,
  input  wire        selector_done,
  input  wire        sparsity_ge_50,
  output reg  [17:0] result,
  output reg         valid
);
  // Input pipelining (latch on selector_done)
  reg [47:0] booth_encoded_r;
  reg [31:0] selected_activations_r;
  reg        selector_done_r;

  always @(posedge clk) begin
    if (reset) begin
      booth_encoded_r        <= 48'd0;
      selected_activations_r <= 32'd0;
      selector_done_r        <= 1'b0;
    end else begin
      selector_done_r <= selector_done;
      if (selector_done) begin
        booth_encoded_r        <= booth_encoded;
        selected_activations_r <= selected_activations;
      end
    end
  end
  
  // MACs for 4 lanes
  wire signed [16:0] mac0_result, mac1_result, mac2_result, mac3_result;
  mac_unit mac0 (.clk(clk), .reset(reset), .enc(booth_encoded[11:0]), .act(selected_activations[7:0]), .result(mac0_result));
  mac_unit mac1 (.clk(clk), .reset(reset), .enc(booth_encoded[23:12]), .act(selected_activations[15:8]), .result(mac1_result));
  mac_unit mac2 (.clk(clk), .reset(reset), .enc(booth_encoded[35:24]), .act(selected_activations[23:16]), .result(mac2_result));
  mac_unit mac3 (.clk(clk), .reset(reset), .enc(booth_encoded[47:36]), .act(selected_activations[31:24]), .result(mac3_result));

  wire signed [17:0] mac_sum =
  {mac0_result[16], mac0_result} +
  {mac1_result[16], mac1_result} +
  {mac2_result[16], mac2_result} +
  {mac3_result[16], mac3_result};

  // Dense-mode accumulator
  reg [17:0] partial_sum;
  reg accumulation_cycle;

  always @(posedge clk) begin
    if (reset) begin
      result <= 0;
      valid <= 0;
      partial_sum <= 0;
      accumulation_cycle <= 0;
    end else begin
      valid <= 0;         // Default: valid low  
      if (selector_done_r) begin
        if (sparsity_ge_50) begin
          result <= mac_sum;
          valid <= 1;
        end else begin
          if (!accumulation_cycle) begin
            // First cycle of accumulation
            partial_sum <= mac_sum;
            accumulation_cycle <= 1;
          end else begin
            result <= partial_sum + mac_sum;
            valid <= 1;
            accumulation_cycle <= 0;
          end
        end
      end
    end
  end
endmodule
