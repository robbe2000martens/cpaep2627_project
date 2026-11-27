# CPAEP 2026–2027 Project: GeMM/GeMV Accelerator

**Students:** Brent Jacobs (r0940009), Robbe Martens (r0853096)

A parametrized SystemVerilog accelerator for GeMM (`C = A × B`) and GeMV (`y = A x`), built from the 1-MAC baseline.
The final design is a **64 × 8 output-stationary array of 512 MACs** with broadcast operands, a reconfigured GeMV mode, and an SRAM port split of **A 768 b (A_lo 512 + A_hi 256) + B 64 b + C 192 b = 1024 b**.

---

## Submission contents

| File | What it is |
|---|---|
| `cpaep_2627_report_jacobs_martens.pdf` | Project report (Q1–Q4) |
| `cpaep_2627_genai_form_jacobs_martens.pdf` | GenAI disclosure form |
| `README.md` | This file |
| everything else | The repository: RTL, testbenches, filelists, Makefile |

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

### Main testbench (all workloads, one simulation)

```bash
make clean
make TEST_MODULE=tb_3d_mac_gemm questasim-run
```

This compiles the files in `flists/tb_3d_mac_gemm.flist` and runs `tb/tb_3d_mac_gemm.sv`. The testbench:

1. checks the design constraints at start-up (≤ 512 MACs, SRAM ports ≤ 1024 bits, a valid A word) and stops with `$fatal` if one is broken,
2. runs the six target workloads, followed by the extra edge cases, **in one simulation without resetting the DUT**,
3. per case: fills A and B with random int8 values, packs them into the SRAM words, fills SRAM C with `X`, runs the accelerator, unpacks C and compares it with the golden model,
4. prints a summary table with cycles, energy and PASS/FAIL per case, and the total area.

It ends with `All N cases passed!`, or `$fatal` with the number of failed cases.

**Expected result for the six target workloads** (energy changes slightly with the random data; cycles do not):

```
 Case     |    M    K    N |   Cycles |       Energy | Check
 GeMM 1   |   64   64   32 |      599 |       950080 | PASS
 GeMM 2   |   64  128   32 |      855 |      1768446 | PASS
 GeMM 3   |   64  576   32 |     2647 |      7496223 | PASS
 GeMV 1   | 1536  576    1 |     9231 |     18971714 | PASS
 GeMV 2   | 3072 1024    1 |    32783 |     67280632 | PASS
 GeMV 3   | 2816 2048    1 |    61445 |    124397426 | PASS
 Area: 2048 area units
```

Average normalized latency (cycles / (M·K·N), averaged over the six workloads): **0.00693**.

### With the GUI (waveforms and coverage)

```bash
make TEST_MODULE=tb_3d_mac_gemm questasim-run-gui
```

When the simulation ends and Questasim asks whether to finish, choose **No**, otherwise the GUI closes.

- **Waveforms:** select an instance (e.g. `tb_3d_mac_gemm/i_dut`), select its signals, *Add Wave*, then `Ctrl+F` to zoom to the full run.
- **Coverage report:** *Tools → Coverage Report → HTML*, choose *Source Code* and *All Coverage Types*, then *OK*. The report is written to `covhtmlreport/index.html`. Coverage collection is already enabled in the Makefile (`vlog +cover`, `vsim -coverage`).

### Unit tests (from the baseline)

```bash
make TEST_MODULE=tb_mac_pe             questasim-run
make TEST_MODULE=tb_ceiling_counter    questasim-run
make TEST_MODULE=tb_single_port_memory questasim-run
make TEST_MODULE=tb_one_mac_gemm       questasim-run   # the original 1-MAC design
```

### Clean up

```bash
make clean        # removes work/, questasim.do, transcripts, waveforms and coverage reports
```

Run `make clean` after changing the Makefile: `questasim.do` is only regenerated when the filelist changes.

---

## Repository structure

```
rtl/
  common/
    ceiling_counter.sv          loop counter (baseline)
    metrics_pkg.sv              shared area / energy counters (baseline)
    single_port_memory.sv       SRAM model, unchanged (baseline)
  gemm/
    gemm_accelerator_new_top.sv NEW: top level, 64x8x1 PE grid, A/B routing, GeMV mapping, drain mux
    gemm_controller_new.sv      NEW: tile counters, read FSM, drain FSM, GeMV PE-bank overlap
    general_mac_pe.sv           MAC PE with area/energy hooks (baseline, reused)
    gemm_accelerator_top.sv     1-MAC baseline (kept for reference)
    gemm_controller.sv          1-MAC baseline controller (kept for reference)
tb/
  tb_3d_mac_gemm.sv             NEW: main testbench (packing, golden model, checks, summary)
  tb_one_mac_gemm.sv            baseline testbench
  tb_mac_pe.sv, tb_ceiling_counter.sv, tb_single_port_memory.sv   unit tests (baseline)
flists/
  tb_3d_mac_gemm.flist          NEW: files for the main testbench
includes/                       shared testbench tasks and golden model (baseline)
docs/solution/results.txt       results of all design points we simulated (see below)
Makefile, questa_setup.sh       build and simulator setup (baseline)
```

---

## Design overview

| | GeMM (N > 1) | GeMV (N = 1) |
|---|---|---|
| Tile | 64 × 8 outputs of C, one per PE | 96 rows of y |
| Per cycle | A_lo: 64 int8 (one column slice of A), broadcast along the PE rows; B: 8 int8 (one row slice of B), broadcast down the PE columns | A_lo + A_hi: 96 int8 rows of A, one per PE; x in lane 0 of B, broadcast |
| PEs used | all 512 | 96, in one of two PE banks (column 0 + top half of column 1, or bottom half of column 1 + column 2) |
| Drain | 86 words of 6 int32 through the 86:1 drain mux; the array waits | 16 words; overlapped with the next tile in the other PE bank |

The mode is chosen by the controller from the workload size (`N == 1` means GeMV). See the report (Q1, Q2) for the block diagram, loop nests and memory layout.

**Memory layout** (element 0 in the least significant bits):

- **SRAM A**, address `mt·Kt + kt`: one k-step of one tile. Row `r` of the tile sits at bits `[8r +: 8]`. A_lo holds rows 0–63, A_hi rows 64–95 at the same address. A GeMM only reads A_lo.
- **SRAM B**, address `nt·Kt + kt`: `B[k][nt·8 + n]` at bits `[8n +: 8]`. In a GeMV, `x[k]` is in lane 0.
- **SRAM C**, address = running count of written words: outputs in column-major order within a tile, 6 int32 per word.

Other features:
- **Operand isolation:** idle PEs get constant zero inputs, so they add no switching energy.
- **Self-checks:** the controller contains `$error` checks for the GeMV overlap, and the testbench checks that A_hi is only read when needed.

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

The start-up checks stop the simulation if a combination breaks the budget. `docs/solution/results.txt` lists the results of the shapes we simulated: 1×1×1, 2×1×1, 1×2×1, 1×1×2, 2×2×2, 32×16×1, 32×2×2 and the 64×8×1 variants.

To run with different random data, add a seed to the `vsim` flags in the Makefile, e.g. `QST_FLAGS += -sv_seed 42`, then `make clean` and run again.

---

## Notes

- The reference simulator is **Questasim**. Verilator was not verified for `tb_3d_mac_gemm`.
- The testbench allocates large arrays (`DataDepth = 6,000,000`) for the biggest workload, GeMV 3 (A = 2816 × 2048), so the simulation needs some memory and time.
