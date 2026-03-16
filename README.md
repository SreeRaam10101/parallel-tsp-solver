# Parallel Branch and Bound Solver for TSP with Quality-Equalizing Load Balancing

A high-performance parallel solver for the **Traveling Salesperson Problem (TSP)** using Branch and Bound (B&B), parallelized with **MPI**. Features a novel **Quality-Equalizing (QE) dynamic load balancing** strategy enhanced with a **weighted progress heuristic** for more nuanced work distribution.

> **Course**: CS566 — Parallel Processing  
> **Authors**: Congqi Huang, Sagar, and Lipi Sree Raam

---

## Overview

The TSP is an NP-hard combinatorial optimization problem: find the shortest route visiting every city exactly once and returning to the start. This project implements an exact solver using Branch and Bound, progressively adding parallelism and load balancing to scale to larger problem instances.

### Solver Variants

| Version | File | Description |
|---|---|---|
| **Sequential** | `sequential_tsp_bb.cpp` | Baseline single-process B&B solver |
| **Parallel** | `parallel_tsp_bb.cpp` | MPI-parallel B&B, no dynamic load balancing |
| **Parallel-Cut** | `parallel_tsp_bb_cutting.cpp` | Parallel B&B with PQ size cap to prevent OOM |
| **QE** | `parallel_tsp_qe.cpp` | QE dynamic load balancing (direct struct send) |
| **QE-Serialization** | `parallel_tsp_qe_serialization.cpp` | QE with MPI_Pack/Unpack for portability |
| **QE-Weighted-Full** | `parallel_tsp_qe_weighted_full.cpp` | **Best variant** — QE with weighted progress heuristic |

## Key Features

- **Branch and Bound** with MST-based lower bounding (Prim's algorithm)
- **MPI parallelism** with ring-topology neighbor communication
- **Quality-Equalizing load balancing**: processes exchange quality thresholds with neighbors and dynamically redistribute work to equalize search effort
- **Weighted progress heuristic**: `W(N) = LB(N) + LB(N) × penalty × progress_ratio(N)`, penalizing deeper nodes to influence sharing decisions
- **Distributed termination detection** with wakeup/confirmation protocol
- **Global upper bound synchronization** via `MPI_Allreduce`

## Performance Results

Tested on 32 MPI processes (4 nodes × 8 tasks/node):

| Cities | Sequential | Parallel | QE | QE-Weighted |
|--------|-----------|----------|------|-------------|
| 8 | 0.001s | 0.199s | 0.236s | — |
| 15 | 0.006s | 0.180s | 0.857s | — |
| 28 | 11.0s | 3.8s | 3.5s | 3.6s |
| 34 | 113.4s | 29.7s | 25.4s | **15.7s** |
| 36 | 565.1s | 204.8s | 80.4s | **41.3s** |
| 40 | 10525.0s | 3382.9s | **322.9s** | 1907.8s |

**Key findings:**
- QE load balancing provides **>10× speedup** over basic parallel for N=40
- Weighted heuristic achieves the **best times for N=34 and N=36** (nearly halving QE)
- For N=40, the fixed penalty factor proves suboptimal — adaptive tuning is future work

## Building

### Prerequisites

- C++ compiler with C++11 support (GCC 12+ recommended)
- MPI implementation (OpenMPI 4.1+)
- `make`

### Compile

```bash
# Build all variants
make all

# Build specific targets
make sequential        # Sequential solver only
make parallel          # Basic parallel + cutting
make qe                # All QE variants

# Debug build (enables DEBUG_PRINT macros)
make all DEBUG=1
```

## Usage

### Sequential

```bash
./bin/sequential_tsp_bb data/tsp_15.txt
```

### Parallel (local)

```bash
mpirun -np 4 ./bin/parallel_tsp_qe_weighted_full data/tsp_28.txt
```

### HPC Cluster (SLURM)

```bash
# Edit scripts/submit-job-qe_weighted_full.sh for your cluster config
sbatch scripts/submit-job-qe_weighted_full.sh data/tsp_34.txt
```

### Run Full Benchmark Suite

```bash
./scripts/run_benchmarks.sh data/tsp_15.txt 4
```

### Generate Custom Instances

```bash
python3 scripts/generate_instance.py 50 -o data/tsp_50.txt --seed 123
```

## Input Format

TSPLIB-style EUC_2D format:

```
NAME: example
TYPE: TSP
DIMENSION: 5
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 38 52
2 79 15
3 10 90
4 55 40
5 90 70
EOF
```

## Project Structure

```
parallel-tsp-solver/
├── src/
│   ├── tsp_common.h                         # Shared data structures & utilities
│   ├── sequential_tsp_bb.cpp                # Sequential baseline
│   ├── parallel_tsp_bb.cpp                  # Basic MPI parallel
│   ├── parallel_tsp_bb_cutting.cpp          # Parallel with PQ size cap
│   ├── parallel_tsp_qe.cpp                  # QE load balanced
│   ├── parallel_tsp_qe_serialization.cpp    # QE with Pack/Unpack
│   └── parallel_tsp_qe_weighted_full.cpp    # QE + weighted heuristic
├── data/                                     # TSP input instances
├── scripts/
│   ├── submit-job-qe_weighted_full.sh       # SLURM job script
│   ├── run_benchmarks.sh                    # Benchmark runner
│   └── generate_instance.py                 # Random instance generator
├── results/                                  # Benchmark output logs
├── docs/                                     # Project report
├── Makefile
├── .gitignore
└── README.md
```

## Algorithm Details

### Branch and Bound

1. **Branching**: Extend partial tours by visiting unvisited cities
2. **Bounding**: MST-based lower bound on remaining tour cost
3. **Pruning**: Discard nodes where `LB(N) ≥ Global Upper Bound`
4. **Search**: Best-first via min-heap priority queue on estimated cost

### Quality-Equalizing Load Balancing

- Processes form a **ring topology** and exchange threshold values
- A threshold is the estimated cost of the S-th best node in the local PQ
- Idle or underloaded processes **request work** from neighbors with better thresholds
- Donors send their 2nd through S-th best nodes, keeping their best

### Weighted Progress Heuristic

Augments the QE threshold with a progress penalty:

```
W(N) = LB(N) + LB(N) × WEIGHT_PROGRESS_PENALTY × (visited_cities - 1) / (total_cities - 1)
```

This makes deeper (more progressed) nodes appear costlier in the threshold exchange, influencing which nodes get shared and how busy a process appears.

## Configuration

Key constants in `src/tsp_common.h`:

| Parameter | Default | Description |
|---|---|---|
| `MAXSIZE` | 50 | Maximum cities supported |
| `QE_SPAN_S` | 3 | Top-S nodes for QE threshold |
| `WORK_TRANSFER_COUNT` | 2 | Nodes transferred per request |
| `WEIGHT_PROGRESS_PENALTY` | 0.15 | Progress penalty factor |
| `GUB_UPDATE_INTERVAL` | 500 | Iterations between GUB syncs |
| `LB_CHECK_INTERVAL` | 200 | Iterations between LB checks |

## Known Limitations & Future Work

- **Fixed penalty factor**: `WEIGHT_PROGRESS_PENALTY` is static; adaptive tuning could improve N=40+ performance
- **Termination logic**: Weighted thresholds are compared against unweighted GUB — refinement needed
- **Homogeneous cluster assumption**: Direct struct send requires matching memory layouts
- **Future directions**: Hybrid MPI+OpenMP, Held-Karp lower bounds, work-stealing variants, adaptive heuristics

## License

This project was developed for CS566 (Parallel Processing). Released for educational purposes.
