module tb_3d_mac_gemm;
  //---------------------------
  // Design Time Parameters
  //---------------------------

  // Matrices in the testbench (golden model, packing): large enough for
  // the biggest operand, A of GeMV 3 = 2816 x 2048 elements
  parameter int unsigned InDataWidth   = 8;
  parameter int unsigned OutDataWidth  = 32;
  parameter int unsigned DataDepth     = 6000000;
  parameter int unsigned AddrWidth     = (DataDepth <= 1) ? 1 : $clog2(DataDepth);
  parameter int unsigned SizeAddrWidth = 12;     // sizes up to 4095

  // The MAC array: H x W x D
  parameter int unsigned ArrayHeight   = 64;     // H, PE rows    (m)
  parameter int unsigned ArrayWidth    = 8;      // W, PE columns (n)
  parameter int unsigned ArrayDepth    = 1;      // D, MACs per PE (k)

  // SRAM ports: 1024 bits cumulative
  parameter int unsigned APortWidth    = InDataWidth * ArrayHeight * ArrayDepth;
  parameter int unsigned BPortWidth    = InDataWidth * ArrayWidth  * ArrayDepth;
  parameter int unsigned CPortWidth    = ((1024 - APortWidth - BPortWidth) / OutDataWidth) * OutDataWidth;
  localparam int unsigned CLanes       = CPortWidth / OutDataWidth;

  initial begin
    if (APortWidth + BPortWidth + CPortWidth > 1024)
      $fatal(1, "SRAM ports use %0d bits, budget is 1024",
             APortWidth + BPortWidth + CPortWidth);
    if (ArrayHeight * ArrayWidth * ArrayDepth > 512)
      $fatal(1, "%0d MACs, budget is 512", ArrayHeight * ArrayWidth * ArrayDepth);
  end

  //---------------------------
  // Test cases
  //---------------------------
  // GeMM: C (M x N) = A (M x K) * B (K x N)
  // GeMV: y (M x 1) = A (M x K) * x (K x 1), which is a GeMM with N = 1
  // The small cases first: they hit the corners of the tiling
  // (one k-step, partial row and column tiles, a single element).
  // DESIGN NOTE: the target workloads are the last six lines.
  //---------------------------
  typedef struct {
    string       name;
    int unsigned M;
    int unsigned K;
    int unsigned N;
  } test_case_t;

  localparam int unsigned NumCases = 6;

  test_case_t test_cases [NumCases] = '{
    //  name        M     K     N
    '{"GeMM 1",    64,   64,   32},
    '{"GeMM 2",    64,  128,   32},
    '{"GeMM 3",    64,  576,   32},
    '{"GeMV 1",  1536,  576,    1},
    '{"GeMV 2",  3072, 1024,    1},
    '{"GeMV 3",  2816, 2048,    1}
  };

  // Per-case results, printed in the summary at the end
  int unsigned case_cycles [NumCases];
  real         case_energy [NumCases];
  int unsigned case_errors [NumCases];

  //---------------------------
  // Wires
  //---------------------------
  logic [SizeAddrWidth-1:0] M_i, K_i, N_i;   // sizes in elements

  logic clk_i;
  logic rst_ni;
  logic start;
  logic done;
  logic [AddrWidth-1:0] test_depth;

  // Golden result and the plain row-major matrices
  logic signed [OutDataWidth-1:0] G_memory [DataDepth];
  logic signed [ InDataWidth-1:0] A_matrix [DataDepth];
  logic signed [ InDataWidth-1:0] B_matrix [DataDepth];
  logic signed [OutDataWidth-1:0] C_matrix [DataDepth];

  // Memory control
  logic [AddrWidth-1:0] sram_a_addr;
  logic [AddrWidth-1:0] sram_b_addr;
  logic [AddrWidth-1:0] sram_c_addr;
  logic                 sram_a_re;
  logic                 sram_b_re;

  // Memory access
  logic signed [APortWidth-1:0] sram_a_rdata;
  logic signed [BPortWidth-1:0] sram_b_rdata;
  logic signed [CPortWidth-1:0] sram_c_wdata;
  logic                         sram_c_we;

  //---------------------------
  // Memories: one single_port_memory per port, only the word width changed
  //---------------------------
  single_port_memory #(
    .DataWidth     ( APortWidth    ),
    .DataDepth     ( DataDepth     ),
    .AddrWidth     ( AddrWidth     )
  ) i_sram_a (
    .clk_i         ( clk_i         ),
    .rst_ni        ( rst_ni        ),
    .mem_addr_i    ( sram_a_addr   ),
    .mem_re_i      ( sram_a_re     ),
    .mem_we_i      ( '0            ),
    .mem_wr_data_i ( '0            ),
    .mem_rd_data_o ( sram_a_rdata  )
  );

  single_port_memory #(
    .DataWidth     ( BPortWidth    ),
    .DataDepth     ( DataDepth     ),
    .AddrWidth     ( AddrWidth     )
  ) i_sram_b (
    .clk_i         ( clk_i         ),
    .rst_ni        ( rst_ni        ),
    .mem_addr_i    ( sram_b_addr   ),
    .mem_re_i      ( sram_b_re     ),
    .mem_we_i      ( '0            ),
    .mem_wr_data_i ( '0            ),
    .mem_rd_data_o ( sram_b_rdata  )
  );

  single_port_memory #(
    .DataWidth     ( CPortWidth    ),
    .DataDepth     ( DataDepth     ),
    .AddrWidth     ( AddrWidth     )
  ) i_sram_c (
    .clk_i         ( clk_i         ),
    .rst_ni        ( rst_ni        ),
    .mem_addr_i    ( sram_c_addr   ),
    .mem_re_i      ( '0            ),
    .mem_we_i      ( sram_c_we     ),
    .mem_wr_data_i ( sram_c_wdata  ),
    .mem_rd_data_o ( /* unused */  )
  );

  //---------------------------
  // DUT
  //---------------------------
  gemm_accelerator_new_top #(
    .InDataWidth   ( InDataWidth   ),
    .OutDataWidth  ( OutDataWidth  ),
    .AddrWidth     ( AddrWidth     ),
    .SizeAddrWidth ( SizeAddrWidth ),
    .ArrayHeight   ( ArrayHeight   ),
    .ArrayWidth    ( ArrayWidth    ),
    .ArrayDepth    ( ArrayDepth    ),
    .APortWidth    ( APortWidth    ),
    .BPortWidth    ( BPortWidth    ),
    .CPortWidth    ( CPortWidth    )
  ) i_dut (
    .clk_i          ( clk_i        ),
    .rst_ni         ( rst_ni       ),
    .start_i        ( start        ),
    .M_size_i       ( M_i          ),
    .K_size_i       ( K_i          ),
    .N_size_i       ( N_i          ),
    .sram_a_addr_o  ( sram_a_addr  ),
    .sram_b_addr_o  ( sram_b_addr  ),
    .sram_c_addr_o  ( sram_c_addr  ),
    .sram_a_re_o    ( sram_a_re    ),
    .sram_b_re_o    ( sram_b_re    ),
    .sram_a_rdata_i ( sram_a_rdata ),
    .sram_b_rdata_i ( sram_b_rdata ),
    .sram_c_wdata_o ( sram_c_wdata ),
    .sram_c_we_o    ( sram_c_we    ),
    .done_o         ( done         )
  );

  //---------------------------
  // Tasks and functions
  //---------------------------
  `include "includes/common_tasks.svh"
  `include "includes/test_tasks.svh"
  `include "includes/test_func.svh"

  function automatic int unsigned ceil_div(input int unsigned a, input int unsigned b);
    return (a + b - 1) / b;
  endfunction

  //---------------------------
  // Packing: the layout must match gemm_accelerator_top.sv
  //   SRAM A word mt*Kt + kt : A(mt*H + m, kt*D + k) at bits [(m*D + k)*8 +: 8]
  //   SRAM B word nt*Kt + kt : B(kt*D + k, nt*W + n) at bits [(k*W + n)*8 +: 8]
  // Elements outside the matrix are zero (padding).
  //---------------------------
  task automatic pack_inputs(input int unsigned M, input int unsigned K, input int unsigned N);
    logic [APortWidth-1:0] a_word;
    logic [BPortWidth-1:0] b_word;
    int unsigned Mt, Kt, Nt, row, col;

    Mt = ceil_div(M, ArrayHeight);
    Kt = ceil_div(K, ArrayDepth);
    Nt = ceil_div(N, ArrayWidth);

    for (int unsigned mt = 0; mt < Mt; mt++) begin
      for (int unsigned kt = 0; kt < Kt; kt++) begin
        a_word = '0;
        for (int unsigned m = 0; m < ArrayHeight; m++) begin
          for (int unsigned k = 0; k < ArrayDepth; k++) begin
            row = mt*ArrayHeight + m;
            col = kt*ArrayDepth  + k;
            if (row < M && col < K)
              a_word[(m*ArrayDepth + k)*InDataWidth +: InDataWidth] = A_matrix[row*K + col];
          end
        end
        i_sram_a.memory[mt*Kt + kt] = a_word;
      end
    end

    for (int unsigned nt = 0; nt < Nt; nt++) begin
      for (int unsigned kt = 0; kt < Kt; kt++) begin
        b_word = '0;
        for (int unsigned k = 0; k < ArrayDepth; k++) begin
          for (int unsigned n = 0; n < ArrayWidth; n++) begin
            row = kt*ArrayDepth + k;
            col = nt*ArrayWidth + n;
            if (row < K && col < N)
              b_word[(k*ArrayWidth + n)*InDataWidth +: InDataWidth] = B_matrix[row*N + col];
          end
        end
        i_sram_b.memory[nt*Kt + kt] = b_word;
      end
    end
  endtask

  //---------------------------
  // Unpacking: SRAM C holds the drained tiles back to back, in loop order
  // (mt outer, nt inner). Tile output p = n*H + m (column-major) is in
  // word p / CLanes of the tile, lane p % CLanes. Only the n_valid
  // columns of a tile are drained: G = ceil(H*n_valid / CLanes) words.
  //---------------------------
  task automatic unpack_output(input int unsigned M, input int unsigned N);
    logic [CPortWidth-1:0] c_word;
    int unsigned Mt, Nt, ptr, n_valid, groups, p, m, n, row, col;

    Mt  = ceil_div(M, ArrayHeight);
    Nt  = ceil_div(N, ArrayWidth);
    ptr = 0;
    for (int unsigned mt = 0; mt < Mt; mt++) begin
      for (int unsigned nt = 0; nt < Nt; nt++) begin
        n_valid = (N - nt*ArrayWidth < ArrayWidth) ? N - nt*ArrayWidth : ArrayWidth;
        groups  = ceil_div(ArrayHeight * n_valid, CLanes);
        for (int unsigned g = 0; g < groups; g++) begin
          c_word = i_sram_c.memory[ptr];
          ptr++;
          for (int unsigned l = 0; l < CLanes; l++) begin
            p   = g*CLanes + l;
            n   = p / ArrayHeight;
            m   = p % ArrayHeight;
            row = mt*ArrayHeight + m;
            col = nt*ArrayWidth  + n;
            if (n < n_valid && row < M)
              C_matrix[row*N + col] = c_word[l*OutDataWidth +: OutDataWidth];
          end
        end
      end
    end
  endtask

  //---------------------------
  // Test control
  //---------------------------
  initial begin
    clk_i = 1'b0;
    forever #5 clk_i = ~clk_i;  // 100MHz clock
  end

  initial begin
    start  = 1'b0;
    M_i    = '0;
    K_i    = '0;
    N_i    = '0;
    rst_ni = 1'b0;
    #50;
    rst_ni = 1'b1;

    // All cases in one simulation, without resetting the DUT in between
    foreach (test_cases[c]) begin
      M_i = test_cases[c].M;
      K_i = test_cases[c].K;
      N_i = test_cases[c].N;

      $display("[%0s] A (%0d x %0d) * B (%0d x %0d) -> C (%0d x %0d)",
               test_cases[c].name, M_i, K_i, K_i, N_i, M_i, N_i);

      // Random row-major matrices
      for (integer i = 0; i < M_i*K_i; i++) A_matrix[i] = $urandom() % (2 ** InDataWidth);
      for (integer i = 0; i < K_i*N_i; i++) B_matrix[i] = $urandom() % (2 ** InDataWidth);

      // Pack them into the SRAM words; poison SRAM C so a missing write fails
      pack_inputs(M_i, K_i, N_i);
      for (integer i = 0; i < DataDepth; i++) i_sram_c.memory[i] = 'x;
      for (integer i = 0; i < M_i*N_i; i++) C_matrix[i] = 'x;

      gemm_golden(M_i, K_i, N_i, A_matrix, B_matrix, G_memory);

      clk_delay(1);

      metrics_pkg::energy = 0;
      start_and_wait_gemm(case_cycles[c]);
      case_energy[c] = metrics_pkg::energy;

      unpack_output(M_i, N_i);
      test_depth = M_i * N_i;
      verify_result_c(G_memory, C_matrix, test_depth, 0, case_errors[c]);

      $display("[%0s] %0s: %0d cycles, %0.0f energy units\n",
               test_cases[c].name, (case_errors[c] == 0) ? "PASS" : "FAIL",
               case_cycles[c], case_energy[c]);
      clk_delay(10);
    end

    print_report();
    $finish;
  end

  //---------------------------
  // Final report
  //---------------------------
  function automatic string pad(string str, int unsigned width);
    while (str.len() < width) str = {str, " "};
    return str;
  endfunction

  task automatic print_report();
    int unsigned num_failed = 0;

    $display("");
    $display("==================================================================");
    $display(" RESULTS  (array %0dx%0dx%0d, ports A %0d + B %0d + C %0d bits)",
             ArrayHeight, ArrayWidth, ArrayDepth, APortWidth, BPortWidth, CPortWidth);
    $display("------------------------------------------------------------------");
    $display(" %s | %4s %4s %4s | %8s | %12s | %s",
             pad("Case", 8), "M", "K", "N", "Cycles", "Energy", "Check");
    $display("------------------------------------------------------------------");
    foreach (test_cases[c]) begin
      $display(" %s | %4d %4d %4d | %8d | %12.0f | %s",
               pad(test_cases[c].name, 8), test_cases[c].M, test_cases[c].K,
               test_cases[c].N, case_cycles[c], case_energy[c],
               (case_errors[c] == 0) ? "PASS" : "FAIL");
      if (case_errors[c] != 0) num_failed++;
    end
    $display("------------------------------------------------------------------");
    $display(" Area: %0.0f area units", metrics_pkg::area);
    $display("==================================================================");
    $display("");

    if (num_failed != 0) begin
      $fatal(1, "%0d of %0d cases FAILED", num_failed, NumCases);
    end
    $display("All %0d cases passed!", NumCases);
  endtask

endmodule
