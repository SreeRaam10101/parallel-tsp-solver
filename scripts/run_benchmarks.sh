#!/bin/bash
# Run all solver variants on a given input and collect results.
# Usage: ./scripts/run_benchmarks.sh <input_file> <num_mpi_procs>

INPUT=${1:-data/tsp_15.txt}
NP=${2:-4}
RESULTS_DIR=results

mkdir -p $RESULTS_DIR

echo "============================================"
echo "Benchmark: $INPUT | MPI procs: $NP"
echo "============================================"

# Build all
make all 2>&1 | tail -1

echo ""
echo "--- Sequential ---"
./bin/sequential_tsp_bb "$INPUT" 2>&1 | tee "$RESULTS_DIR/seq_$(basename $INPUT .txt).log"

echo ""
echo "--- Parallel (Basic) ---"
mpirun --oversubscribe -np $NP ./bin/parallel_tsp_bb "$INPUT" 2>&1 | tee "$RESULTS_DIR/par_$(basename $INPUT .txt).log"

echo ""
echo "--- Parallel (PQ Cutting) ---"
mpirun --oversubscribe -np $NP ./bin/parallel_tsp_bb_cutting "$INPUT" 2>&1 | tee "$RESULTS_DIR/par_cut_$(basename $INPUT .txt).log"

echo ""
echo "--- QE Load Balanced ---"
mpirun --oversubscribe -np $NP ./bin/parallel_tsp_qe "$INPUT" 2>&1 | tee "$RESULTS_DIR/qe_$(basename $INPUT .txt).log"

echo ""
echo "--- QE Serialization ---"
mpirun --oversubscribe -np $NP ./bin/parallel_tsp_qe_serialization "$INPUT" 2>&1 | tee "$RESULTS_DIR/qe_ser_$(basename $INPUT .txt).log"

echo ""
echo "--- QE Weighted Full ---"
mpirun --oversubscribe -np $NP ./bin/parallel_tsp_qe_weighted_full "$INPUT" 2>&1 | tee "$RESULTS_DIR/qe_wt_$(basename $INPUT .txt).log"

echo ""
echo "============================================"
echo "All benchmarks complete. Results in $RESULTS_DIR/"
