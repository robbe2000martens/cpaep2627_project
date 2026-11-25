//---------------------------
// GeMM accelerator top: H x W x D output-stationary broadcast MAC array
//
// Description:
// H x W PEs; PE (m, n) is a general_mac_pe with NumInputs = D and owns
// the output C[mt*H + m][nt*W + n] of the current tile.
//   * SRAM A word (mt, kt) = A[mt*H +: H][kt*D +: D], broadcast along PE row m
//   * SRAM B word (nt, kt) = B[kt*D +: D][nt*W +: W], broadcast along PE column n
//   * The D products of a PE are summed by its adder tree, then accumulated.
// No operand moves from PE to PE (no systolic forwarding): every operand
// goes straight from the SRAM port to all PEs of its row or column.
//
// Memory words (element 0 in the LSBs):
//   SRAM A, address mt*Kt + kt : A(m, k) at bits [(m*D + k)*8  +: 8]
//   SRAM B, address nt*Kt + kt : B(k, n) at bits [(k*W + n)*8  +: 8]
//   SRAM C, address = running count of written words. Tiles are drained in
//   loop order; within a tile, output p = n*H + m (column-major) goes to
//   word p / CLanes, lane p % CLanes, i.e. bits [(p % CLanes)*32 +: 32].
//
// Parameters:
// - ArrayHeight/Width/Depth : H, W, D of the MAC array (H*W*D <= 512)
// - APortWidth, BPortWidth  : one k-step per word: 8*H*D and 8*W*D bits
// - CPortWidth              : SRAM C word, a multiple of 32 bits
//   (A + B + C port widths must stay within the 1,024-bit budget)
//---------------------------

module gemm_accelerator_new_top #(
  parameter int unsigned InDataWidth   = 8,
  parameter int unsigned OutDataWidth  = 32,
  parameter int unsigned AddrWidth     = 17,
  parameter int unsigned SizeAddrWidth = 12,
  parameter int unsigned ArrayHeight   = 64,
  parameter int unsigned ArrayWidth    = 8,
  parameter int unsigned ArrayDepth    = 1,
  parameter int unsigned APortWidth    = InDataWidth * ArrayHeight * ArrayDepth,
  parameter int unsigned BPortWidth    = InDataWidth * ArrayWidth  * ArrayDepth,
  parameter int unsigned CPortWidth    = 448
) (
  input  logic                            clk_i,
  input  logic                            rst_ni,
  input  logic                            start_i,
  input  logic        [SizeAddrWidth-1:0] M_size_i,
  input  logic        [SizeAddrWidth-1:0] K_size_i,
  input  logic        [SizeAddrWidth-1:0] N_size_i,
  output logic        [    AddrWidth-1:0] sram_a_addr_o,
  output logic        [    AddrWidth-1:0] sram_b_addr_o,
  output logic        [    AddrWidth-1:0] sram_c_addr_o,
  output logic                            sram_a_re_o,
  output logic                            sram_b_re_o,
  input  logic signed [   APortWidth-1:0] sram_a_rdata_i,
  input  logic signed [   BPortWidth-1:0] sram_b_rdata_i,
  output logic signed [   CPortWidth-1:0] sram_c_wdata_o,
  output logic                            sram_c_we_o,
  output logic                            done_o
);

  localparam int unsigned CLanes    = CPortWidth / OutDataWidth;
  localparam int unsigned NumOut    = ArrayHeight * ArrayWidth;
  localparam int unsigned MaxGroups = (NumOut + CLanes - 1) / CLanes;
  localparam int unsigned GroupWidth = (MaxGroups <= 1) ? 1 : $clog2(MaxGroups);

  //---------------------------
  // Controller: tile counters, FSM and drain sequencing
  //---------------------------
  logic [SizeAddrWidth-1:0] mt, nt, kt, Kt;
  logic                     mem_re, valid_data, init_save, busy;
  logic [ArrayHeight-1:0]   row_valid;
  logic [ArrayWidth-1:0]    col_valid;
  logic [GroupWidth-1:0]    drain_idx;

  gemm_controller_new #(
    .SizeWidth    ( SizeAddrWidth ),
    .ArrayHeight  ( ArrayHeight   ),
    .ArrayWidth   ( ArrayWidth    ),
    .ArrayDepth   ( ArrayDepth    ),
    .CLanes       ( CLanes        ),
    .AddrWidth    ( AddrWidth     )
  ) i_gemm_controller (
    .clk_i        ( clk_i         ),
    .rst_ni       ( rst_ni        ),
    .start_i      ( start_i       ),
    .M_size_i     ( M_size_i      ),
    .K_size_i     ( K_size_i      ),
    .N_size_i     ( N_size_i      ),
    .mt_o         ( mt            ),
    .nt_o         ( nt            ),
    .kt_o         ( kt            ),
    .Kt_o         ( Kt            ),
    .mem_re_o     ( mem_re        ),
    .data_valid_o ( valid_data    ),
    .init_save_o  ( init_save     ),
    .row_valid_o  ( row_valid     ),
    .col_valid_o  ( col_valid     ),
    .c_we_o       ( sram_c_we_o   ),
    .drain_idx_o  ( drain_idx     ),
    .c_addr_o     ( sram_c_addr_o ),
    .busy_o       ( busy          ),
    .done_o       ( done_o        )
  );

  //---------------------------
  // Address generation (one word = one k-step of one tile)
  //---------------------------
  assign sram_a_re_o   = mem_re;
  assign sram_b_re_o   = mem_re;
  assign sram_a_addr_o = AddrWidth'(mt * Kt + kt);
  assign sram_b_addr_o = AddrWidth'(nt * Kt + kt);

  //---------------------------
  // SRAM words viewed as tiles
  //---------------------------
  logic [ArrayHeight-1:0][ArrayDepth-1:0][InDataWidth-1:0]  a_tile;
  logic [ArrayDepth-1:0][ArrayWidth-1:0][InDataWidth-1:0]   b_tile;
  // Outputs stored column-major: c_cm[n][m] sits at bits (n*H + m)*32
  logic [ArrayWidth-1:0][ArrayHeight-1:0][OutDataWidth-1:0] c_cm;

  assign a_tile = sram_a_rdata_i;
  assign b_tile = sram_b_rdata_i;

  //---------------------------
  // The PE grid
  // Operand isolation: a PE column without valid data (n >= N in the
  // current tile, e.g. columns 1..W-1 in a GeMV) gets a constant zero A
  // operand, a PE row without valid data (m >= M) a constant zero B
  // operand, so their multipliers do not toggle (no switching energy).
  //---------------------------
  genvar m, n, k;
  for (m = 0; m < ArrayHeight; m++) begin : gen_pe_row
    for (n = 0; n < ArrayWidth; n++) begin : gen_pe_col
      logic signed [ArrayDepth-1:0][InDataWidth-1:0] a_pe, b_pe;

      assign a_pe = col_valid[n] ? a_tile[m] : '0;
      for (k = 0; k < ArrayDepth; k++) begin : gen_b
        assign b_pe[k] = row_valid[m] ? b_tile[k][n] : '0;
      end

      general_mac_pe #(
        .InDataWidth  ( InDataWidth  ),
        .NumInputs    ( ArrayDepth   ),
        .OutDataWidth ( OutDataWidth )
      ) i_mac_pe (
        .clk_i        ( clk_i        ),
        .rst_ni       ( rst_ni       ),
        .a_i          ( a_pe         ),
        .b_i          ( b_pe         ),
        .a_valid_i    ( valid_data   ),
        .b_valid_i    ( valid_data   ),
        .init_save_i  ( init_save    ),
        .acc_clr_i    ( !busy        ),
        .c_o          ( c_cm[n][m]   )
      );
    end
  end

  //---------------------------
  // Drain: word drain_idx of the tile = outputs p = drain_idx*CLanes ...
  // (a MaxGroups-to-1 multiplexer of CPortWidth-bit slices)
  //---------------------------
  logic [MaxGroups*CPortWidth-1:0] c_flat;

  always_comb begin
    c_flat                            = '0;
    c_flat[NumOut*OutDataWidth-1:0]   = c_cm;
  end

  assign sram_c_wdata_o = c_flat[drain_idx*CPortWidth +: CPortWidth];

endmodule
