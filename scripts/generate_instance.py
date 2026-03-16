#!/usr/bin/env python3
"""Generate random TSP instances in TSPLIB format."""
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Generate TSP instance")
    parser.add_argument("n", type=int, help="Number of cities")
    parser.add_argument("-o", "--output", default=None, help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-coord", type=int, default=2500, help="Max coordinate")
    args = parser.parse_args()

    random.seed(args.seed)
    outfile = args.output or f"tsp_{args.n}.txt"

    with open(outfile, "w") as f:
        f.write(f"NAME: tsp_{args.n}\n")
        f.write("TYPE: TSP\n")
        f.write(f"COMMENT: Random {args.n}-city instance (seed={args.seed})\n")
        f.write(f"DIMENSION: {args.n}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(1, args.n + 1):
            x = random.randint(1, args.max_coord)
            y = random.randint(1, args.max_coord)
            f.write(f"{i} {x} {y}\n")
        f.write("EOF\n")
    print(f"Generated {args.n}-city instance: {outfile}")

if __name__ == "__main__":
    main()
