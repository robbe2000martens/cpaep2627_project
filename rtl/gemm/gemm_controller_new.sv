//---------------------------
// Tiled GeMM controller for an H x W x D output-stationary MAC array
//
// Description:
// The array computes one tile of C at a time. Each PE keeps one output in
// its accumulator (output stationary) while the k-steps of the reduction
// stream past it; one k-step covers D values of k and takes one cycle.
// When the tile is finished it is drained through SRAM C, G words of
// CLanes int32 values.
//
// Tile shape (see gemm_accelerator_new_top.sv):
//   GeMM (N > 1)  : TM x W = H x W outputs, one per PE
//   GeMV (N == 1) : TM = GemvRows rows of y, held by one of PeBanks PE
//                   banks. Counting the PEs column-major, PE bank b is
//                   PEs b*S .. b*S + GemvRows-1, S = GemvWords*CLanes
//                   (GemvRows rounded up to whole C words). For H = 64
//                   and 96 rows: bank 0 = PE column 0 and the top of
//                   column 1, bank 1 = the bottom of column 1 and column 2
//
// SRAM A is two banks with the same address: A_lo holds rows 0 .. H-1 of
// a word, A_hi rows H .. GemvRows-1. Only a GeMV tile with more than H
// rows of y reads A_hi (a_hi_re_o), so a GeMM never pays for it.
//
// Loop nest (only the sequential loops live here; the parfor loops over
// the PEs and k1 < D are the MAC array itself):
//
//   for mt = 0 .. Mt-1          // block of TM rows of C      (M counter)
//     for nt = 0 .. Nt-1        // block of W columns of C    (N counter)
//       for kt = 0 .. Kt-1      // one k-step per cycle       (K counter)
//         read SRAM A word (mt, kt) and SRAM B word (nt, kt)
//       drain the tile: G words to SRAM C
//
// Timing of one tile (the SRAMs have a one cycle read latency):
//   Read  : Kt cycles, one A and one B read (one k-step) per cycle
//   Wait  : 1 cycle, the last k-step is being accumulated
//   Drain : G cycles, one C word per cycle
// Reading and draining are two state machines, so the tiles can follow
// each other in two ways:
//   One by one (a GeMM, which needs every PE, and a GeMV with
//   Kt <= GemvWords): the next tile's first read is issued in the last
//   drain cycle. Its data reaches the PEs (init_save) in the cycle after
//   the last write, so nothing is overwritten too early.
//     => Kt + G cycles per tile
//   Overlapped (a GeMV with Kt > GemvWords and two PE banks): the tiles
//   take turns between the banks. The next tile is read into the other
//   bank right after the last k-step, while this tile waits and drains.
//   The drain (G <= GemvWords words) ends no later than the next tile's
//   last k-step, so two drains never collide, and a bank is only read
//   again two tiles later, after its drain.
//     => Kt cycles per tile, plus 1 + G for the drain of the last tile
//
// G depends on the outputs of the tile that are drained (the drain order
// is column-major, see gemm_accelerator_new_top.sv):
//   GeMM: all H rows of the valid columns
//         n_valid = min(W, N - nt*W),          G = ceil(H * n_valid / CLanes)
//   GeMV: the valid rows of y, which are the first outputs of the PE bank
//         m_valid = min(TM, M - mt*TM),        G = ceil(m_valid / CLanes)
//         PE bank b starts at drain word b*GemvWords
//
// Parameters:
// - SizeWidth   : width of the M, K, N size inputs (in elements)
// - ArrayHeight : H, PE rows    (parallel over m)
// - ArrayWidth  : W, PE columns (parallel over n)
// - ArrayDepth  : D, multipliers per PE (parallel over k)
// - GemvRows    : rows of y per GeMV tile (rows per SRAM A word), H .. H*W
// - CLanes      : int32 values per SRAM C word
// - AddrWidth   : width of the SRAM C address
// - PeBanks     : PE banks for GeMV tiles: 2 overlaps the tiles, 1 runs
//                 them one by one (2 needs 2*GemvWords*CLanes <= H*W)
//---------------------------

module gemm_controller_new #(
  parameter int unsigned SizeWidth   = 12,
  parameter int unsigned ArrayHeight = 64,
  parameter int unsigned ArrayWidth  = 8,
  parameter int unsigned ArrayDepth  = 1,
  parameter int unsigned GemvRows    = 96,
  parameter int unsigned CLanes      = 6,
  parameter int unsigned AddrWidth   = 17,
  parameter int unsigned PeBanks     = 2,
  // Derived: most drain words of one tile, and the drain index width
  parameter int unsigned MaxGroups   = (ArrayHeight*ArrayWidth + CLanes - 1) / CLanes,
  parameter int unsigned GroupWidth  = (MaxGroups <= 1) ? 1 : $clog2(MaxGroups),
  // Derived: drain words of a full GeMV tile, which is also the first
  // drain word of PE bank 1
  parameter int unsigned GemvWords   = (GemvRows + CLanes - 1) / CLanes
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
  output logic                  mem_re_o,      // read SRAM A_lo and B this cycle
  output logic                  a_hi_re_o,     // ... and SRAM A_hi too
  output logic                  data_valid_o,  // A/B data at the PEs is valid
  output logic                  init_save_o,   // ... and it is the first k-step
  output logic                  gemv_o,        // ... and it belongs to a GeMV
  output logic                  pe_bank_o,     // ... in this PE bank (GeMV only)
  output logic [GemvRows-1:0]   row_valid_o,   // rows of the tile with real data
  output logic [ArrayWidth-1:0] col_valid_o,   // PE columns with real data
  // Drain control
  output logic                  c_we_o,        // write one SRAM C word
  output logic [GroupWidth-1:0] drain_idx_o,   // which word of the PE array
  output logic [AddrWidth-1:0]  c_addr_o,      // SRAM C address
  output logic                  busy_o,
  output logic                  done_o
);

  //-----------------------
  // Mode: a GeMV (N == 1) uses tiles of GemvRows rows of y, a GeMM
  // tiles of H x W outputs. The sizes only change between runs.
  //-----------------------
  logic gemv;

  assign gemv = (N_size_i == SizeWidth'(1));

  //-----------------------
  // Tile counts: ceil(size / tile dimension)
  // (constant divisors: shifts when the dimensions are powers of two;
  // GemvRows = 96 is not, but the sizes are static during a run)
  //-----------------------
  logic [SizeWidth-1:0] Mt, Nt, Kt;

  assign Mt   = gemv ? (M_size_i + GemvRows    - 1) / GemvRows
                     : (M_size_i + ArrayHeight - 1) / ArrayHeight;
  assign Nt   = (N_size_i + ArrayWidth  - 1) / ArrayWidth;
  assign Kt   = (K_size_i + ArrayDepth  - 1) / ArrayDepth;
  assign Kt_o = Kt;

  //-----------------------
  // Overlap the tiles (see the timing above): only a GeMV, only with two
  // PE banks, and only if a drain ends in time, Kt > GemvWords.
  // Static during a run, like the sizes.
  //-----------------------
  logic overlap;

  assign overlap = (PeBanks > 1) && gemv && (Kt > SizeWidth'(GemvWords));

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
  localparam int unsigned MValidWidth = $clog2(GemvRows + 1);
  localparam int unsigned NValidWidth = $clog2(ArrayWidth + 1);

  logic [SizeWidth-1:0]   m_left, n_left;    // rows/columns left from the tile on
  logic [MValidWidth-1:0] m_valid;           // valid rows of the tile
  logic [NValidWidth-1:0] n_valid;           // valid columns of the tile
  logic [GroupWidth:0]    groups;            // drain words of the tile
  logic [GemvRows-1:0]    row_mask;          // only the first H bits in a GeMM
  logic [ArrayWidth-1:0]  col_mask;

  always_comb begin
    if (gemv) begin
      m_left  = M_size_i - SizeWidth'(mt_o * GemvRows);
      m_valid = (m_left >= GemvRows) ? MValidWidth'(GemvRows) : MValidWidth'(m_left);
    end else begin
      m_left  = M_size_i - SizeWidth'(mt_o * ArrayHeight);
      m_valid = (m_left >= ArrayHeight) ? MValidWidth'(ArrayHeight) : MValidWidth'(m_left);
    end
    n_left  = N_size_i - SizeWidth'(nt_o * ArrayWidth);
    n_valid = (n_left >= ArrayWidth) ? NValidWidth'(ArrayWidth) : NValidWidth'(n_left);
    groups  = gemv ? (GroupWidth+1)'((m_valid + CLanes - 1) / CLanes)
                   : (GroupWidth+1)'((ArrayHeight * n_valid + CLanes - 1) / CLanes);
    for (int r = 0; r < GemvRows;   r++) row_mask[r] = (r < m_valid);
    for (int n = 0; n < ArrayWidth; n++) col_mask[n] = (n < n_valid);
  end

  // PE bank of the tile at the counters: with overlap the tiles take
  // turns, even tiles in bank 0 and odd ones in bank 1 (in a GeMV nt is
  // always 0, so mt counts the tiles). Without overlap: always bank 0.
  logic pe_bank;

  assign pe_bank = overlap && mt_o[0];

  // Latched at the last k-step of a tile, used while that tile drains
  // (by then the counters already point at the next tile)
  logic [GroupWidth-1:0] g_last_q;      // G - 1 of the draining tile
  logic                  drain_bank_q;  // PE bank of the draining tile

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      g_last_q     <= '0;
      drain_bank_q <= 1'b0;
    end else if (k_wrap) begin
      g_last_q     <= GroupWidth'(groups - 1);
      drain_bank_q <= pe_bank;
    end
  end

  //-----------------------
  // Data pipeline: follows the one cycle read latency of SRAM A and B
  //   cycle t   : address of k-step kt issued (move_k)
  //   cycle t+1 : the data is at the PEs (data_valid, init_save if kt == 0)
  //-----------------------
  logic                   data_valid_q;
  logic                   first_k_q;
  logic                   gemv_q;
  logic                   pe_bank_q;
  logic [GemvRows-1:0]    row_valid_q;
  logic [ArrayWidth-1:0]  col_valid_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      data_valid_q <= 1'b0;
      first_k_q    <= 1'b0;
      gemv_q       <= 1'b0;
      pe_bank_q    <= 1'b0;
      row_valid_q  <= '0;
      col_valid_q  <= '0;
    end else begin
      data_valid_q <= move_k;
      first_k_q    <= move_k && (kt_o == '0);
      if (move_k) begin
        gemv_q      <= gemv;
        pe_bank_q   <= pe_bank;
        row_valid_q <= row_mask;
        col_valid_q <= col_mask;
      end
    end
  end

  assign mem_re_o     = move_k;
  assign data_valid_o = data_valid_q;
  assign init_save_o  = first_k_q;
  assign gemv_o       = gemv_q;
  assign pe_bank_o    = pe_bank_q;
  assign row_valid_o  = row_valid_q;
  assign col_valid_o  = col_valid_q;

  //-----------------------
  // SRAM A_hi read enable. A_hi holds rows H .. GemvRows-1 of an A word,
  // so a k-step reads it only if its tile has more than H valid rows.
  // row_mask belongs to the tile at the counters, which is the one being
  // read, and has only its first H bits set in a GeMM: A_hi stays off in
  // every GeMM and in a last GeMV tile of at most H rows.
  //-----------------------
  if (GemvRows > ArrayHeight) begin : gen_a_hi_re
    assign a_hi_re_o = move_k && row_mask[ArrayHeight];
  end else begin : gen_no_a_hi
    assign a_hi_re_o = 1'b0;   // an A word has no rows for A_hi
  end

  //-----------------------
  // State machines
  //
  // Read side: issues the k-steps (move_k)
  //   Idle    : wait for start, then issue the first k-step at once
  //   Compute : one k-step per cycle
  //   Hold    : one by one: the tile drains before the next one starts,
  //             whose first k-step is issued in the last drain cycle
  //   Finish  : every k-step is issued, wait for the last drain
  // After a k-step (after_k) the same tile goes on, or its last k-step
  // hands the tile to the drain side and the read side goes on with the
  // next tile (overlap), holds, or finishes.
  //
  // Drain side: writes the finished tile to SRAM C
  //   DrainIdle : nothing to drain
  //   DrainWait : the last k-step of the tile is accumulated
  //   Drain     : one C word per cycle, G cycles. With overlap the next
  //               tile can be handed over in the last drain cycle
  //               (Kt = GemvWords + 1); its drain then follows at once.
  //-----------------------
  typedef enum logic [1:0] {
    Idle,
    Compute,
    Hold,
    Finish
  } read_state_t;

  typedef enum logic [1:0] {
    DrainIdle,
    DrainWait,
    Drain
  } drain_state_t;

  read_state_t           read_q, read_d, after_k;
  drain_state_t          drain_q, drain_d;
  logic [GroupWidth-1:0] g_q;          // drain word counter
  logic [AddrWidth-1:0]  c_addr_q;     // next SRAM C address
  logic                  drain_last;   // the last word of the tile is written
  logic                  c_we;

  assign drain_last = (drain_q == Drain) && (g_q == g_last_q);

  always_comb begin
    if      (!k_wrap) after_k = Compute;
    else if (m_wrap)  after_k = Finish;
    else if (overlap) after_k = Compute;
    else              after_k = Hold;
  end

  always_comb begin
    read_d = read_q;
    move_k = 1'b0;
    done_o = 1'b0;

    unique case (read_q)
      Idle: begin
        if (start_i) begin
          move_k = 1'b1;
          read_d = after_k;
        end
      end

      Compute: begin
        move_k = 1'b1;
        read_d = after_k;
      end

      Hold: begin
        if (drain_last) begin
          move_k = 1'b1;        // first k-step of the next tile
          read_d = after_k;
        end
      end

      Finish: begin
        if (drain_last) begin
          done_o = 1'b1;        // same cycle as the last C write
          read_d = Idle;
        end
      end

      default: read_d = Idle;
    endcase
  end

  always_comb begin
    drain_d = drain_q;
    c_we    = 1'b0;

    unique case (drain_q)
      DrainIdle: begin
        if (k_wrap) drain_d = DrainWait;
      end

      DrainWait: begin
        drain_d = Drain;
      end

      Drain: begin
        c_we = 1'b1;
        if (drain_last) drain_d = k_wrap ? DrainWait : DrainIdle;
      end

      default: drain_d = DrainIdle;
    endcase
  end

  // All counters have wrapped back to zero after the last tile;
  // the clear only matters if a run is ever cut short.
  assign clear = done_o;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      read_q   <= Idle;
      drain_q  <= DrainIdle;
      g_q      <= '0;
      c_addr_q <= '0;
    end else begin
      read_q   <= read_d;
      drain_q  <= drain_d;
      g_q      <= (drain_q == Drain && !drain_last) ? g_q + 1'b1 : '0;
      if (read_q == Idle) c_addr_q <= '0;
      else if (c_we)      c_addr_q <= c_addr_q + 1'b1;
    end
  end

  assign c_we_o      = c_we;
  assign drain_idx_o = drain_bank_q ? GroupWidth'(GemvWords + g_q) : g_q;
  assign c_addr_o    = c_addr_q;
  assign busy_o      = (read_q != Idle);

  //-----------------------
  // Simulation checks of the overlap; Kt > GemvWords rules both out:
  // - a tile is handed to the drain side only when that side is idle or
  //   in its last cycle (else g_last_q and drain_bank_q would change
  //   under a running drain)
  // - k-step data never reaches a PE bank that still has words to drain
  //   (its accumulators would change before they are written out)
  //-----------------------
  always @(posedge clk_i) begin
    if (rst_ni && k_wrap && !(drain_q == DrainIdle || drain_last))
      $error("%m: tile %0d is done while the previous tile still drains", mt_o);
    if (rst_ni && data_valid_q && drain_q == Drain && !drain_last &&
        pe_bank_q == drain_bank_q)
      $error("%m: k-step data reaches PE bank %0d while it drains", pe_bank_q);
  end

endmodule
