# ==== CONFIGURABLE VARIABLES ====
VERILATOR       := verilator

TEST_MODULE     := tb_one_mac_gemm

INCLUDE_DIRS    := +incdir+.

FLIST_DIRS      := flists
FILELIST        := $(TEST_MODULE).flist
FILE_PATH       := $(FLIST_DIRS)/$(FILELIST)

VLT_FLAGS       := -O3
VLT_FLAGS       += --trace
VLT_FLAGS       += --trace-structs

VLT_WAIVE       := -Wno-CASEINCOMPLETE
VLT_WAIVE       += -Wno-WIDTHTRUNC
VLT_WAIVE       += -Wno-WIDTHEXPAND
VLT_WAIVE       += -Wno-fatal

# Random seed for $$urandom: make ... SEED=42 (use SEED=random for a new one every run)
SEED            ?= 0

# Coverage types collected at compile time:
# b = branch, c = condition, e = expression, s = statement, f = FSM, t = toggle
COVER_TYPES     := bcesft

QST_FLAGS       := -voptargs=\"+acc\"
QST_FLAGS       += -coverage
QST_FLAGS       += -sv_seed $(SEED)

# Coverage output of the command-line run
COV_DB          := cov_$(TEST_MODULE).ucdb
COV_HTML        := covhtmlreport
COV_TXT         := coverage_$(TEST_MODULE).txt

BIN_DIR         := bin
OBJ_DIR         := obj_dir

# Derived from $(FILE_PATH)
SRCS := $(shell cat $(FILE_PATH))

# ==== BUILD TARGET (Verilator) ====
all: $(BIN_DIR)/$(TEST_MODULE)

$(BIN_DIR):
	mkdir -p $@

$(BIN_DIR)/$(TEST_MODULE): $(BIN_DIR) $(FILE_PATH)
	$(VERILATOR) --sv $(SRCS) $(INCLUDE_DIRS) $(VLT_WAIVE) $(VLT_FLAGS) --binary -o $(TEST_MODULE)
	cp $(OBJ_DIR)/$(TEST_MODULE) $(BIN_DIR)/.
	rm -rf $(OBJ_DIR)

# ==== QUESTASIM ====
# GUI script: compile, simulate, show all waveforms (as before)
questasim.do: $(FILE_PATH)
	@echo 'Generating $@'
	@echo vlib work > $@
	@echo vlog +cover=$(COVER_TYPES) +acc -sv -f $(FILE_PATH) $(INCLUDE_DIRS) >> $@
	@echo vsim $(QST_FLAGS) work.$(TEST_MODULE) >> $@
	@echo add wave -r \/\* >> $@
	@echo run -all >> $@

# Command-line script: compile, simulate, save the coverage database, quit.
# -onfinish stop: the testbench's $$finish stops the simulation instead of
# closing vsim, so the coverage can still be saved afterwards.
questasim_cov.do: $(FILE_PATH)
	@echo 'Generating $@'
	@echo vlib work > $@
	@echo vlog +cover=$(COVER_TYPES) +acc -sv -f $(FILE_PATH) $(INCLUDE_DIRS) >> $@
	@echo vsim $(QST_FLAGS) -onfinish stop work.$(TEST_MODULE) >> $@
	@echo run -all >> $@
	@echo coverage save $(COV_DB) >> $@
	@echo quit -f >> $@

# Simulation without GUI, then the coverage reports:
#   $(COV_HTML)/index.html   HTML report (open in a browser)
#   $(COV_TXT)               text report with the details
questasim-run: questasim_cov.do
	@echo 'Running Questasim simulation w/ Command Line Interface'
	vsim -c -do questasim_cov.do
	vcover report -html -output $(COV_HTML) -details -source $(COV_DB)
	@echo 'Coverage report: $(COV_HTML)/index.html and $(COV_TXT)'

questasim-run-gui: questasim.do
	@echo 'Running Questasim simulation w/ GUI'
	vsim -gui -do questasim.do

# ==== CLEAN ====
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) .vcd transcript *.do work *.wlf *covhtmlreport *report.txt *.ucdb coverage_.txt

.PHONY: all clean questasim.do questasim_cov.do questasim-run questasim-run-gui
