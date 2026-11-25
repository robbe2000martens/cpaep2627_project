//---------------------------
// Tiled GeMM controller for an H x W x D output-stationary MAC array
//
// Description:
// The array computes one H x W tile of C at a time. Each PE keeps one
// output in its accumulator (output stationary) while the k-steps of the
// reduction stream past it; one k-step covers D values of k and takes one
// cycle. When the tile is finished it is drained through SRAM C, G words
// of CLanes int32 values, and the next tile starts.
//
// Loop nest (only the sequential loops live here; the parfor loops over
// m1 < H, n1 < W and k1 < D are the MAC array itself):
//
//   for mt = 0 .. Mt-1          // block of H rows of C       (M counter)
//     for nt = 0 .. Nt-1        // block of W columns of C    (N counter)
//       for kt = 0 .. Kt-1      // one k-step per cycle       (K counter)
//         read SRAM A word (mt, kt) and SRAM B word (nt, kt)
//       drain the tile: G words to SRAM C
//
// Timing of one tile (the SRAMs have a one cycle read latency):
//   Compute : Kt cycles, one A and one B read per cycle
//   Wait    : 1 cycle, the last k-step is being accumulated
//   Drain   : G cycles, one C word per cycle. In the last drain cycle the
//             first read of the next tile is already issued: its data
//             reaches the PEs (init_save) in the cycle after the last
//             write, so nothing is overwritten too early.
//   => Kt + G cycles per tile.
//
// G depends on the number of valid columns of the tile (the drain order
// is column-major, see gemm_accelerator_top.sv):
//   n_valid = min(W, N - nt*W),  G = ceil(H * n_valid / CLanes)
//
// Parameters:
// - SizeWidth   : width of the M, K, N size inputs (in elements)
// - ArrayHeight : H, PE rows    (parallel over m)
// - ArrayWidth  : W, PE columns (parallel over n)
// - ArrayDepth  : D, multipliers per PE (parallel over k)
// - CLanes      : int32 values per SRAM C word
// - AddrWidth   : width of the SRAM C address
//---------------------------

module gemm_controller_new #(
  parameter int unsigned SizeWidth   = 12,
  parameter int unsigned ArrayHeight = 64,
  parameter int unsigned ArrayWidth  = 8,
  parameter int unsigned ArrayDepth  = 1,
  parameter int unsigned CLanes      = 14,
  parameter int unsigned AddrWidth   = 17,
  // Derived: most drain words of one tile, and the drain index width
  parameter int unsigned MaxGroups   = (ArrayHeight*ArrayWidth + CLanes - 1) / CLanes,
  parameter int unsigned GroupWidth  = (MaxGroups <= 1) ? 1 : $clog2(MaxGroups)
)(
  input  logic                  clk_i,
  input  logic                  rst_ni,
  input  logic                  start_i,
  // Workload sizes in elements: C (M x N) = A (M x K) * B (K x N)
  input  logic [SizeWidth-1:0]  M_size_i,
  input  logic [SizeWidth-1:0]  K_size_i,
  input  logic [SizeWidth-1:0]  N_size_i,
  // Tile counters and the number of k-steps, for the address generation
  output logic [SizeWidth-1:0]  mt_o,
  output logic [SizeWidth-1:0]  nt_o,
  output logic [SizeWidth-1:0]  kt_o,
  output logic [SizeWidth-1:0]  Kt_o,
  // Datapath control
  output logic                  mem_re_o,      // read SRAM A and B this cycle
  output logic                  data_valid_o,  // A/B data at the PEs is valid
  output logic                  init_save_o,   // ... and it is the first k-step
  output logic [ArrayHeight-1:0] row_valid_o,  // PE rows with real data
  output logic [ArrayWidth-1:0] col_valid_o,   // PE columns with real data
  // Drain control
  output logic                  c_we_o,        // write one SRAM C word
  output logic [GroupWidth-1:0] drain_idx_o,   // which word of the tile
  output logic [AddrWidth-1:0]  c_addr_o,      // SRAM C address
  output logic                  busy_o,
  output logic                  done_o
);

  //-----------------------
  // Tile counts: ceil(size / array dimension)
  // (constant divisors: shifts when the dimensions are powers of two)
  //-----------------------
  logic [SizeWidth-1:0] Mt, Nt, Kt;

  assign Mt   = (M_size_i + ArrayHeight - 1) / ArrayHeight;
  assign Nt   = (N_size_i + ArrayWidth  - 1) / ArrayWidth;
  assign Kt   = (K_size_i + ArrayDepth  - 1) / ArrayDepth;
  assign Kt_o = Kt;

  //-----------------------
  // Loop counters: K innermost, then N, then M
  //-----------------------
  logic move_k;   // issue one k-step (one A and one B read)
  logic k_wrap;   // ... and it is the last k-step of the tile
  logic n_wrap;   // ... and the tile is the last of its row of tiles
  logic m_wrap;   // ... and the tile is the last tile of the workload
  logic clear;

  ceiling_counter #(
    .Width        ( SizeWidth ),
    .HasCeiling   ( 1         )
  ) i_k_counter (
    .clk_i        ( clk_i     ),
    .rst_ni       ( rst_ni    ),
    .tick_i       ( move_k    ),
    .clear_i      ( clear     ),
    .ceiling_i    ( Kt        ),
    .count_o      ( kt_o      ),
    .last_value_o ( k_wrap    )
  );

  ceiling_counter #(
    .Width        ( SizeWidth ),
    .HasCeiling   ( 1         )
  ) i_n_counter (
    .clk_i        ( clk_i     ),
    .rst_ni       ( rst_ni    ),
    .tick_i       ( k_wrap    ),
    .clear_i      ( clear     ),
    .ceiling_i    ( Nt        ),
    .count_o      ( nt_o      ),
    .last_value_o ( n_wrap    )
  );

  ceiling_counter #(
    .Width        ( SizeWidth ),
    .HasCeiling   ( 1         )
  ) i_m_counter (
    .clk_i        ( clk_i     ),
    .rst_ni       ( rst_ni    ),
    .tick_i       ( n_wrap    ),
    .clear_i      ( clear     ),
    .ceiling_i    ( Mt        ),
    .count_o      ( mt_o      ),
    .last_value_o ( m_wrap    )
  );

  //-----------------------
  // Valid rows/columns and drain length of the tile at the counters
  //-----------------------
  // Sized to what they can hold (32-bit ints would leave many bits that
  // never toggle and spoil the toggle coverage)
  localparam int unsigned MValidWidth = $clog2(ArrayHeight + 1);
  localparam int unsigned NValidWidth = $clog2(ArrayWidth + 1);

  logic [SizeWidth-1:0]   m_left, n_left;    // rows/columns left from the tile on
  logic [MValidWidth-1:0] m_valid;           // valid rows of the tile
  logic [NValidWidth-1:0] n_valid;           // valid columns of the tile
  logic [GroupWidth:0]    groups;            // drain words of the tile
  logic [ArrayHeight-1:0] row_mask;
  logic [ArrayWidth-1:0]  col_mask;

  always_comb begin
    m_left  = M_size_i - SizeWidth'(mt_o * ArrayHeight);
    m_valid = (m_left >= ArrayHeight) ? MValidWidth'(ArrayHeight) : MValidWidth'(m_left);
    n_left  = N_size_i - SizeWidth'(nt_o * ArrayWidth);
    n_valid = (n_left >= ArrayWidth) ? NValidWidth'(ArrayWidth) : NValidWidth'(n_left);
    groups  = (GroupWidth+1)'((ArrayHeight * n_valid + CLanes - 1) / CLanes);
    for (int m = 0; m < ArrayHeight; m++) row_mask[m] = (m < m_valid);
    for (int n = 0; n < ArrayWidth;  n++) col_mask[n] = (n < n_valid);
  end

  // Latched at the last k-step of a tile, used while that tile drains
  // (by then the counters already point at the next tile)
  logic [GroupWidth-1:0] g_last_q;     // G - 1 of the draining tile
  logic                  tile_last_q;  // the draining tile is the last one

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      g_last_q    <= '0;
      tile_last_q <= 1'b0;
    end else if (k_wrap) begin
      g_last_q    <= GroupWidth'(groups - 1);
      tile_last_q <= m_wrap;
    end
  end

  //-----------------------
  // Data pipeline: follows the one cycle read latency of SRAM A and B
  //   cycle t   : address of k-step kt issued (move_k)
  //   cycle t+1 : the data is at the PEs (data_valid, init_save if kt == 0)
  //-----------------------
  logic                   data_valid_q;
  logic                   first_k_q;
  logic [ArrayHeight-1:0] row_valid_q;
  logic [ArrayWidth-1:0]  col_valid_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      data_valid_q <= 1'b0;
      first_k_q    <= 1'b0;
      row_valid_q  <= '0;
      col_valid_q  <= '0;
    end else begin
      data_valid_q <= move_k;
      first_k_q    <= move_k && (kt_o == '0);
      if (move_k) begin
        row_valid_q <= row_mask;
        col_valid_q <= col_mask;
      end
    end
  end

  assign mem_re_o     = move_k;
  assign data_valid_o = data_valid_q;
  assign init_save_o  = first_k_q;
  assign row_valid_o  = row_valid_q;
  assign col_valid_o  = col_valid_q;

  //-----------------------
  // Main state machine
  //-----------------------
  typedef enum logic [1:0] {
    Idle,
    Compute,
    Wait,
    Drain
  } state_t;

  state_t                state_q, state_d;
  logic [GroupWidth-1:0] g_q;        // drain word counter
  logic [AddrWidth-1:0]  c_addr_q;   // next SRAM C address
  logic                  c_we;

  always_comb begin
    state_d = state_q;
    move_k  = 1'b0;
    c_we    = 1'b0;
    done_o  = 1'b0;

    unique case (state_q)
      // Wait for start, then issue the first k-step at once
      Idle: begin
        if (start_i) begin
          move_k  = 1'b1;
          state_d = k_wrap ? Wait : Compute;
        end
      end

      // One k-step per cycle until the last one of the tile is issued
      Compute: begin
        move_k = 1'b1;
        if (k_wrap) state_d = Wait;
      end

      // The last k-step is accumulated in this cycle
      Wait: begin
        state_d = Drain;
      end

      // Write G words; the last cycle also starts the next tile
      Drain: begin
        c_we = 1'b1;
        if (g_q == g_last_q) begin
          if (tile_last_q) begin
            done_o  = 1'b1;     // same cycle as the last C write
            state_d = Idle;
          end else begin
            move_k  = 1'b1;     // first k-step of the next tile
            state_d = k_wrap ? Wait : Compute;
          end
        end
      end

      default: state_d = Idle;
    endcase
  end

  // All counters have wrapped back to zero after the last tile;
  // the clear only matters if a run is ever cut short.
  assign clear = done_o;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q  <= Idle;
      g_q      <= '0;
      c_addr_q <= '0;
    end else begin
      state_q  <= state_d;
      g_q      <= (state_q == Drain) ? g_q + 1'b1 : '0;
      if (state_q == Idle) c_addr_q <= '0;
      else if (c_we)       c_addr_q <= c_addr_q + 1'b1;
    end
  end

  assign c_we_o      = c_we;
  assign drain_idx_o = g_q;
  assign c_addr_o    = c_addr_q;
  assign busy_o      = (state_q != Idle);

endmodule
