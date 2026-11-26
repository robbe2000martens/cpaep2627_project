//---------------------------
// GeMM accelerator top: H x W x D output-stationary broadcast MAC array
//
// Description:
// H x W PEs; PE (m, n) is a general_mac_pe with NumInputs = D. The D
// products of a PE are summed by its adder tree, then accumulated.
// No operand moves from PE to PE (no systolic forwarding): every operand
// goes straight from the SRAM port to the PEs that use it.
//
// Two modes; the controller picks one from the workload size:
//
// GeMM (N > 1): PE (m, n) owns C[mt*H + m][nt*W + n] of the current tile.
//   * SRAM A word (mt, kt) = A[mt*H +: H][kt*D +: D], broadcast along PE row m
//   * SRAM B word (nt, kt) = B[kt*D +: D][nt*W +: W], broadcast along PE column n
//   The other ARows - H rows of an A word are not used (zero padding).
//
// GeMV (N == 1, y = A x): a tile is ARows = APortWidth / (8*D) rows of y,
// more than the H PEs of column 0 can hold. PE (m, n) owns row
// r = n*H + m of the tile: rows 0 .. H-1 fill PE column 0, the rest the
// top of column 1 (64 + 32 = 96 rows for a 768-bit A port).
//   * SRAM A word (mt, kt) = A[mt*ARows +: ARows][kt*D +: D]; row r goes
//     to PE (r % H, r / H) only
//   * SRAM B word (0, kt)  = x[kt*D +: D] in lane 0, as in a GeMM; every
//     PE column that holds rows of y takes lane 0 as its B operand
//
// Memory words (element 0 in the LSBs):
//   SRAM A, address mt*Kt + kt : A(r, k) at bits [(r*D + k)*8  +: 8]
//   SRAM B, address nt*Kt + kt : B(k, n) at bits [(k*W + n)*8  +: 8]
//   SRAM C, address = running count of written words. Tiles are drained in
//   loop order; within a tile, output p = n*H + m (column-major) goes to
//   word p / CLanes, lane p % CLanes, i.e. bits [(p % CLanes)*32 +: 32].
//   In GeMV mode p is the row r of y within the tile.
//
// Parameters:
// - ArrayHeight/Width/Depth : H, W, D of the MAC array (H*W*D <= 512)
// - APortWidth              : SRAM A word, ARows rows of D int8 values,
//                             H <= ARows <= H*W (ARows = H: a GeMV only
//                             uses PE column 0)
// - BPortWidth              : one k-step of B per word: 8*W*D bits
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
  parameter int unsigned APortWidth    = 768,
  parameter int unsigned BPortWidth    = InDataWidth * ArrayWidth  * ArrayDepth,
  parameter int unsigned CPortWidth    = 192
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

  localparam int unsigned ARows     = APortWidth / (InDataWidth * ArrayDepth);  // rows per A word
  localparam int unsigned CLanes    = CPortWidth / OutDataWidth;
  localparam int unsigned NumOut    = ArrayHeight * ArrayWidth;
  localparam int unsigned MaxGroups = (NumOut + CLanes - 1) / CLanes;
  localparam int unsigned GroupWidth = (MaxGroups <= 1) ? 1 : $clog2(MaxGroups);

  //---------------------------
  // Controller: tile counters, FSM and drain sequencing
  //---------------------------
  logic [SizeAddrWidth-1:0] mt, nt, kt, Kt;
  logic                     mem_re, valid_data, init_save, busy, gemv;
  logic [ARows-1:0]         row_valid;
  logic [ArrayWidth-1:0]    col_valid;
  logic [GroupWidth-1:0]    drain_idx;

  gemm_controller_new #(
    .SizeWidth    ( SizeAddrWidth ),
    .ArrayHeight  ( ArrayHeight   ),
    .ArrayWidth   ( ArrayWidth    ),
    .ArrayDepth   ( ArrayDepth    ),
    .GemvRows     ( ARows         ),
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
    .gemv_o       ( gemv          ),
    .row_valid_o  ( row_valid     ),
    .col_valid_o  ( col_valid     ),
    .c_we_o       ( sram_c_we_o   ),
    .drain_idx_o  ( drain_idx     ),
    .c_addr_o     ( sram_c_addr_o ),
    .busy_o       ( busy          ),
    .done_o       ( done_o        )
  );

  //---------------------------
  // Address generation (one word = one k-step of one tile; in GeMV mode
  // mt counts tiles of ARows rows)
  //---------------------------
  assign sram_a_re_o   = mem_re;
  assign sram_b_re_o   = mem_re;
  assign sram_a_addr_o = AddrWidth'(mt * Kt + kt);
  assign sram_b_addr_o = AddrWidth'(nt * Kt + kt);

  //---------------------------
  // SRAM words viewed as tiles
  //---------------------------
  logic [ARows-1:0][ArrayDepth-1:0][InDataWidth-1:0]        a_tile;  // row r of A
  logic [ArrayDepth-1:0][ArrayWidth-1:0][InDataWidth-1:0]   b_tile;
  // Outputs stored column-major: c_cm[n][m] sits at bits (n*H + m)*32
  logic [ArrayWidth-1:0][ArrayHeight-1:0][OutDataWidth-1:0] c_cm;

  assign a_tile = sram_a_rdata_i;
  assign b_tile = sram_b_rdata_i;

  //---------------------------
  // B operand of each PE column, broadcast down the column. In GeMV mode
  // every column that holds rows of y multiplies with x, which is lane 0.
  //---------------------------
  logic [ArrayWidth-1:0][ArrayDepth-1:0][InDataWidth-1:0] b_col;

  genvar m, n, k;
  for (n = 0; n < ArrayWidth; n++) begin : gen_b_col
    for (k = 0; k < ArrayDepth; k++) begin : gen_k
      if (n > 0 && n*ArrayHeight < ARows) begin : gen_gemv
        assign b_col[n][k] = gemv ? b_tile[k][0] : b_tile[k][n];
      end else begin : gen_gemm
        assign b_col[n][k] = b_tile[k][n];
      end
    end
  end

  //---------------------------
  // The PE grid
  // In GeMV mode PE (m, n) holds row R = n*H + m of y if R < ARows. This
  // needs an A multiplexer only for n > 0: in column 0, R = m and both
  // modes read the same A row.
  //
  // Operand isolation: a PE without valid data gets constant zero
  // operands, so its multiplier does not toggle (no switching energy).
  //   GeMM: PE (m, n) is busy if row m and column n of the tile exist
  //   GeMV: PE (m, n) is busy if row R of the tile exists in y
  //         (R < ARows and mt*ARows + R < M)
  // e.g. columns 2..W-1 and the lower half of column 1 idle in a GeMV.
  //---------------------------
  for (m = 0; m < ArrayHeight; m++) begin : gen_pe_row
    for (n = 0; n < ArrayWidth; n++) begin : gen_pe_col
      localparam int unsigned R = n*ArrayHeight + m;  // row of y in GeMV mode

      logic                                          pe_en;
      logic signed [ArrayDepth-1:0][InDataWidth-1:0] a_src, a_pe, b_pe;

      if (n > 0 && R < ARows) begin : gen_gemv
        assign a_src = gemv ? a_tile[R]    : a_tile[m];
        assign pe_en = gemv ? row_valid[R] : (row_valid[m] && col_valid[n]);
      end else begin : gen_gemm
        assign a_src = a_tile[m];
        assign pe_en = row_valid[m] && col_valid[n];
      end

      assign a_pe = pe_en ? a_src    : '0;
      assign b_pe = pe_en ? b_col[n] : '0;

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
  // (a MaxGroups-to-1 multiplexer of CPortWidth-bit slices). The same in
  // both modes: the rows of y of a GeMV tile are its first outputs in
  // column-major order.
  //---------------------------
  logic [MaxGroups*CPortWidth-1:0] c_flat;

  always_comb begin
    c_flat                            = '0;
    c_flat[NumOut*OutDataWidth-1:0]   = c_cm;
  end

  assign sram_c_wdata_o = c_flat[drain_idx*CPortWidth +: CPortWidth];

endmodule
