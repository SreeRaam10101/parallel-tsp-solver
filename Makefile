# ============================================================================
# Makefile for Parallel TSP Branch & Bound Solver
# ============================================================================
CXX       = g++
MPICXX    = mpic++
CXXFLAGS  = -O2 -std=c++11 -Wall
LDFLAGS   = -lm
SRC_DIR   = src
BIN_DIR   = bin

# Debug build: make DEBUG=1
ifdef DEBUG
    CXXFLAGS += -DDEBUG_ENABLED -g
endif

SEQUENTIAL  = $(BIN_DIR)/sequential_tsp_bb
PARALLEL    = $(BIN_DIR)/parallel_tsp_bb
PARALLEL_CUT = $(BIN_DIR)/parallel_tsp_bb_cutting
QE          = $(BIN_DIR)/parallel_tsp_qe
QE_SER      = $(BIN_DIR)/parallel_tsp_qe_serialization
QE_WEIGHTED = $(BIN_DIR)/parallel_tsp_qe_weighted_full

.PHONY: all sequential parallel qe clean help

all: $(SEQUENTIAL) $(PARALLEL) $(PARALLEL_CUT) $(QE) $(QE_SER) $(QE_WEIGHTED)

sequential: $(SEQUENTIAL)

parallel: $(PARALLEL) $(PARALLEL_CUT)

qe: $(QE) $(QE_SER) $(QE_WEIGHTED)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(SEQUENTIAL): $(SRC_DIR)/sequential_tsp_bb.cpp $(SRC_DIR)/tsp_common.h | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(PARALLEL): $(SRC_DIR)/parallel_tsp_bb.cpp $(SRC_DIR)/tsp_common.h | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(PARALLEL_CUT): $(SRC_DIR)/parallel_tsp_bb_cutting.cpp $(SRC_DIR)/tsp_common.h | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(QE): $(SRC_DIR)/parallel_tsp_qe.cpp $(SRC_DIR)/tsp_common.h | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(QE_SER): $(SRC_DIR)/parallel_tsp_qe_serialization.cpp $(SRC_DIR)/tsp_common.h | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(QE_WEIGHTED): $(SRC_DIR)/parallel_tsp_qe_weighted_full.cpp $(SRC_DIR)/tsp_common.h | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)

help:
	@echo "Targets:"
	@echo "  all         - Build all solver versions"
	@echo "  sequential  - Build sequential solver only"
	@echo "  parallel    - Build basic parallel + cutting solvers"
	@echo "  qe          - Build all QE variants"
	@echo "  clean       - Remove binaries"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1     - Enable debug output"
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/sequential_tsp_bb data/tsp_15.txt"
	@echo "  mpirun -np 4 ./bin/parallel_tsp_qe_weighted_full data/tsp_15.txt"
