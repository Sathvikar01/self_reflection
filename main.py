"""Main entry point for running experiments."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments.run_baseline import run_baseline
from experiments.run_rl_guided import run_rl_guided
from experiments.run_ablations import run_ablations


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RL-Guided Self-Reflection for LLM Reasoning"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Run baseline experiments")
    baseline_parser.add_argument("--dataset", default="strategy_qa")
    baseline_parser.add_argument("--samples", type=int, default=100)
    baseline_parser.add_argument("--output", default="data/results")
    baseline_parser.add_argument("--seed", type=int, default=42)
    
    # RL command
    rl_parser = subparsers.add_parser("rl", help="Run RL-guided experiments")
    rl_parser.add_argument("--dataset", default="strategy_qa")
    rl_parser.add_argument("--samples", type=int, default=100)
    rl_parser.add_argument("--iterations", type=int, default=50)
    rl_parser.add_argument("--output", default="data/results")
    rl_parser.add_argument("--seed", type=int, default=42)
    
    # Ablation command
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation studies")
    ablation_parser.add_argument("--dataset", default="strategy_qa")
    ablation_parser.add_argument("--samples", type=int, default=50)
    ablation_parser.add_argument("--output", default="data/results")
    ablation_parser.add_argument("--seed", type=int, default=42)
    
    # All command
    all_parser = subparsers.add_parser("all", help="Run all experiments")
    all_parser.add_argument("--dataset", default="strategy_qa")
    all_parser.add_argument("--samples", type=int, default=100)
    all_parser.add_argument("--output", default="data/results")
    all_parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.command == "baseline":
        run_baseline(
            dataset=args.dataset,
            num_samples=args.samples,
            output_dir=args.output,
            seed=args.seed,
        )
    
    elif args.command == "rl":
        run_rl_guided(
            dataset=args.dataset,
            num_samples=args.samples,
            max_iterations=args.iterations,
            output_dir=args.output,
            seed=args.seed,
        )
    
    elif args.command == "ablation":
        run_ablations(
            dataset=args.dataset,
            num_samples=args.samples,
            output_dir=args.output,
            seed=args.seed,
        )
    
    elif args.command == "all":
        print("=" * 60)
        print("Running all experiments")
        print("=" * 60)
        
        print("\n[1/3] Running baseline...")
        run_baseline(
            dataset=args.dataset,
            num_samples=args.samples,
            output_dir=args.output,
            seed=args.seed,
        )
        
        print("\n[2/3] Running RL-guided...")
        run_rl_guided(
            dataset=args.dataset,
            num_samples=args.samples,
            output_dir=args.output,
            seed=args.seed,
        )
        
        print("\n[3/3] Running ablations...")
        run_ablations(
            dataset=args.dataset,
            num_samples=min(50, args.samples),
            output_dir=args.output,
            seed=args.seed,
        )
        
        print("\n" + "=" * 60)
        print("All experiments completed!")
        print("=" * 60)
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
