# CPAEP 2026–2027 Project: GeMM/GeMV Accelerator

**Students:** Brent Jacobs (r0940009), Robbe Martens (r0853096)

A parametrized SystemVerilog accelerator for GeMM (`C = A × B`) and GeMV (`y = A x`), built from the 1-MAC baseline.
The final design is a **64 × 8 output-stationary array of 512 MACs** with broadcast operands, a reconfigured GeMV mode, and an SRAM port split of **A 768 b (A_lo 512 + A_hi 256) + B 64 b + C 192 b = 1024 b**.

Average normalized latency over the six target workloads: **0.006931 cycles per MAC** (144× lower than the 1-MAC baseline), area **2048** units.

---

## Submission contents

| File | What it is |
|---|---|
| `cpaep_2627_report_jacobs_martens.pdf` | Project report (Q1–Q4) |
| `cpaep_2627_genai_form_jacobs_martens.pdf` | GenAI disclosure form |
| `README.md` | This file |
| everything else | The repository: RTL, testbenches, filelists, Makefile, results |

---

## Requirements

- An **ESAT machine** with **Questasim** (computer classroom, or a virtual desktop / VNC session on **Rocky Linux 8**; Questasim does not run on Rocky Linux 9).
- Load Questasim once per shell, from the repository root:

```bash
source ./questa_setup.sh
vsim -help        # prints the help text if the setup worked
```

---

## Running the design

### Main testbench, without GUI (results + coverage report)

```bash
make clean
make TEST_MODULE=tb_3d_mac_gemm questasim-run
```

This compiles the files in `flists/tb_3d_mac_gemm.flist`, runs `tb/tb_3d_mac_gemm.sv` in the command-line simulator, and then writes the coverage reports:

| Output | Content |
|---|---|
| terminal / `transcript` | per-case log and the final results table |
| `cov_tb_3d_mac_gemm.ucdb` | Questa coverage database |
| `covhtmlreport/index.html` | HTML coverage report (open in a browser) |
| `coverage_tb_3d_mac_gemm.txt` | the same coverage details as text |

The coverage types collected are branch, condition, expression, statement, FSM and toggle (`+cover=bcesft`).

### Other random data (seed)

The random matrices come from `$urandom`. The seed defaults to 0; choose another one with `SEED`:

```bash
make clean
make TEST_MODULE=tb_3d_mac_gemm SEED=42 questasim-run
make TEST_MODULE=tb_3d_mac_gemm SEED=random questasim-run   # Questa picks one and prints it
```

The cycle counts do not depend on the data; the energy changes slightly.

### With the GUI (waveforms)

```bash
make TEST_MODULE=tb_3d_mac_gemm questasim-run-gui
```

When the simulation ends and Questasim asks whether to finish, choose **No**, otherwise the GUI closes. All signals are added to the wave window. The coverage report can also be made here: *Tools → Coverage Report → HTML*.

### Unit tests (from the baseline)

```bash
make TEST_MODULE=tb_mac_pe             questasim-run
make TEST_MODULE=tb_ceiling_counter    questasim-run
make TEST_MODULE=tb_single_port_memory questasim-run
make TEST_MODULE=tb_one_mac_gemm       questasim-run   # the original 1-MAC design
```

### Clean up

```bash
make clean   # removes work/, the .do scripts, transcript, waveforms, coverage database and reports
```

`make clean` also deletes `covhtmlreport/`: copy it elsewhere first if you want to keep a report.

---

## What the testbench does

`tb/tb_3d_mac_gemm.sv`:

1. **Checks the design constraints at start-up** (≤ 512 MACs, SRAM ports ≤ 1024 bits, a valid A word) and stops with `$fatal` if one is broken.
2. **Runs 15 test cases in one simulation**, without resetting the DUT between cases.
3. **For each case:**
   - fills A and B with random int8 values and packs them into the SRAM words,
   - fills SRAM C with `X`, so a missing write fails the check,
   - runs the accelerator and counts its cycles and energy,
   - unpacks C and compares it with the golden model.
4. **Checks during the run:**
   - an `$error` if SRAM A_hi is read for a tile that has no rows in it, which would cost energy without changing the result,
   - two `$error`s in the controller for the GeMV tile overlap. These checks are excluded from code coverage, because their conditions can only be fully covered when an error fires.
5. **Prints a results table** with M, K, N, cycles, energy, **normalized latency** (cycles / (M·K·N)) and PASS/FAIL per case, then the total area and the **average normalized latency of the six target workloads**.
6. Ends with `All 15 cases passed!`, or `$fatal` with the number of failed cases.

### Test cases

| # | Case | M | K | N | Purpose |
|---|---|---|---|---|---|
| 1–3 | GeMM WL1–WL3 | 64 | 64 / 128 / 576 | 32 | target workloads |
| 4–6 | GeMV WL1–WL3 | 1536 / 3072 / 2816 | 576 / 1024 / 2048 | 1 | target workloads |
| 7 | GeMM, 1 k-step per tile | 70 | 1 | 9 | partial row tile (6 rows) and column tile (1 column) |
| 8 | GeMM, max K | 1 | 4095 | 2 | one tile of 4095 k-steps |
| 9 | GeMV, tightest overlap | 193 | 17 | 1 | Kt = drain words + 1; 3 tiles, PE bank 0 reused, last tile 1 row |
| 10 | GeMV, no overlap | 166 | 16 | 1 | Kt = drain words: tiles one by one; last tile reads A_hi |
| 11 | GeMM, N = 4088 | 1 | 1 | 4088 | 511 column tiles, every bit of the tile counter set |
| 12 | GeMM, max N | 1 | 1 | 4095 | 512 column tiles, last tile 7 columns |
| 13 | GeMM, max M | 4095 | 1 | 14 | 64 row tiles, last 63 rows; last column tile 6 columns |
| 14 | GeMM, one tile, one k-step | 5 | 1 | 3 | whole workload in one tile (Idle → Finish) |
| 15 | GeMM, reset mid-run | 64 | 64 | 16 | reset twice during a run, then a full run must still be correct |

The six target workloads run first, so their random data (and energy) do not depend on the edge cases. Case 15 is the only one that resets the DUT, on purpose.

### Expected results for the target workloads

Cycles and normalized latency (energy changes slightly with the seed):

| Case | M | K | N | Cycles | Norm. latency |
|---|---|---|---|---|---|
| GeMM WL1 | 64 | 64 | 32 | 599 | 0.004570 |
| GeMM WL2 | 64 | 128 | 32 | 855 | 0.003262 |
| GeMM WL3 | 64 | 576 | 32 | 2647 | 0.002244 |
| GeMV WL1 | 1536 | 576 | 1 | 9231 | 0.010434 |
| GeMV WL2 | 3072 | 1024 | 1 | 32783 | 0.010422 |
| GeMV WL3 | 2816 | 2048 | 1 | 61445 | 0.010654 |
| **Average** | | | | | **0.006931** |

Area: 2048 units (512 MACs × 3 + 512 accumulator registers × 1).

---

## Repository structure

```
rtl/
  common/
    ceiling_counter.sv            loop counter (baseline)
    metrics_pkg.sv                shared area / energy counters (baseline)
    single_port_memory.sv         SRAM model, unchanged (baseline)
  gemm/
    gemm_accelerator_new_top.sv   NEW: top level: 64x8x1 PE grid, A/B routing, GeMV mapping, drain mux
    gemm_controller_new.sv        NEW: tile counters, read FSM, drain FSM, GeMV PE-bank overlap
    general_mac_pe.sv             MAC PE with area/energy hooks (baseline, reused)
    gemm_accelerator_top.sv       1-MAC baseline (kept for reference)
    gemm_controller.sv            1-MAC baseline controller (kept for reference)
tb/
  tb_3d_mac_gemm.sv               NEW: main testbench (packing, golden model, checks, results table)
  tb_one_mac_gemm.sv              baseline testbench
  tb_mac_pe.sv, tb_ceiling_counter.sv, tb_single_port_memory.sv   unit tests (baseline)
flists/
  tb_3d_mac_gemm.flist            NEW: files of the main testbench
includes/                         shared testbench tasks and golden model (baseline)
docs/solution/
  results.txt                     simulation results of every design point we tried
  results_normalized.txt          the same, with MACs and normalized latency per case
  cpaep_results.xlsx              the same in a spreadsheet, with a summary and ranking per design point
Makefile                          Questasim (with coverage and seed) and Verilator targets
questa_setup.sh                   loads Questasim on the ESAT machines (baseline)
```

---

## Design overview

| | GeMM (N > 1) | GeMV (N = 1) |
|---|---|---|
| Tile | 64 × 8 outputs of C, one per PE | 96 rows of y |
| Per cycle | A_lo: 64 int8 (one column slice of A), broadcast along the PE rows; B: 8 int8 (one row slice of B), broadcast down the PE columns | A_lo + A_hi: 96 int8 (rows of A), one per PE; x in lane 0 of B, broadcast |
| PEs used | all 512 | 96, in one of two PE banks: column 0 + top half of column 1, or bottom half of column 1 + column 2 |
| Drain | 86 words of 6 int32 through the drain mux; the array waits | 16 words; overlapped with the next tile in the other PE bank |

The controller picks the mode from the workload size (`N == 1` means GeMV). The report (Q1, Q2) has the block diagram, the loop nests and the memory layout.

**Memory layout** (element 0 in the least significant bits):
- **SRAM A**, address `mt·Kt + kt`: one k-step of one tile; row `r` of the tile at bits `[8r +: 8]`. A_lo holds rows 0–63 and A_hi rows 64–95, at the same address. A GeMM only reads A_lo.
- **SRAM B**, address `nt·Kt + kt`: `B[k][nt·8 + n]` at bits `[8n +: 8]`. In a GeMV, `x[k]` is in lane 0.
- **SRAM C**, address = running count of written words: outputs column-major within a tile, 6 int32 per word.

**Operand isolation:** idle PEs get constant zero inputs, so they add no switching energy.

---

## Changing the configuration

The array shape and port widths are parameters at the top of `tb/tb_3d_mac_gemm.sv`:

| Parameter | Default | Meaning |
|---|---|---|
| `ArrayHeight` | 64 | H, PE rows (parallel over m) |
| `ArrayWidth` | 8 | W, PE columns (parallel over n) |
| `ArrayDepth` | 1 | D, multipliers per PE (parallel over k) |
| `APortWidth` | 768 | SRAM A word (A_lo + A_hi) |
| `BPortWidth` | 8·W·D | SRAM B word, one k-step |
| `CPortWidth` | rest of 1024 | SRAM C word, a multiple of 32 bits |

The start-up checks stop the simulation if a combination breaks the budget. `docs/solution/` lists the results of the shapes we simulated (1×1×1, 2×1×1, 1×2×1, 1×1×2, 2×2×2, 32×16×1, 32×2×2 and the 64×8×1 variants).

---

## Notes

- The reference simulator is **Questasim**. The Verilator target (`make TEST_MODULE=... all`) comes from the baseline; it was not verified for `tb_3d_mac_gemm`.
- The testbench allocates large arrays (`DataDepth = 6,000,000`) for the biggest workload, GeMV WL3 (A = 2816 × 2048), so the simulation needs some memory and time.
