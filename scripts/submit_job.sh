#!/bin/bash
# ============================================================================
# SLURM Job Submission Script for Parallel TSP Solver
# ============================================================================
#SBATCH --partition=batch
#SBATCH --job-name=tsp_qe_job
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=2:30:00
#SBATCH --mem=52G
#SBATCH --output=results/tsp_qe_job_%j.log
#SBATCH --error=results/tsp_qe_job_%j.err

# --- Environment Setup ---
echo "=== Environment Setup ==="
module purge
module load slurm/lakeshore/23.02.4
module load GCC/12.3.0
module load OpenMPI/4.1.5-GCC-12.3.0

echo "Loaded modules:"
module list

# --- Configuration ---
SOLVER=${1:-"parallel_tsp_qe_weighted_full"}  # Default solver
INPUT=${2:-"Input.txt"}                        # Default input file

echo "Solver: $SOLVER"
echo "Input:  $INPUT"

# --- Compilation ---
echo ""
echo "=== Compiling $SOLVER.cpp ==="
mpic++ src/${SOLVER}.cpp -o bin/${SOLVER} -lm -O2 -std=c++11

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi
echo "Compilation successful."

# --- Execution ---
if [ -n "$SLURM_NNODES" ] && [ -n "$SLURM_NTASKS_PER_NODE" ]; then
    NNODES=$SLURM_NNODES
    NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE
    NP=$((NNODES * NTASKS_PER_NODE))
else
    echo "Warning: SLURM env vars not found, defaulting to 4 processes"
    NP=4
fi

echo ""
echo "=== Running on $NP processes ($NNODES nodes x $NTASKS_PER_NODE tasks) ==="

# Copy input file to working directory if needed
if [ ! -f "Input.txt" ] && [ -f "$INPUT" ]; then
    cp "$INPUT" Input.txt
fi

srun --mpi=pmix -n $NP ./bin/${SOLVER}

echo ""
echo "=== Job finished ==="
