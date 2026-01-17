#!/usr/bin/env python3
"""
Union ABM Experiment CLI

Run batch experiments from YAML configuration files.

Usage:
    python run_experiment.py config.yaml [options]
    python run_experiment.py --generate-sample  # Generate sample config file

Examples:
    # Run experiment from config
    python run_experiment.py experiment_config.yaml
    
    # Run with specific number of workers
    python run_experiment.py experiment_config.yaml --workers 4
    
    # Run sequentially (useful for debugging)
    python run_experiment.py experiment_config.yaml --sequential
    
    # Generate sample configuration file
    python run_experiment.py --generate-sample -o my_experiment.yaml
    
    # Run and also save network snapshots
    python run_experiment.py experiment_config.yaml --snapshots
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from union_abm.experiment import (
    ExperimentConfig,
    ExperimentRunner,
    save_network_snapshots_svg,
    create_sample_config
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Union ABM batch experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        type=str,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        '--generate-sample', '-g',
        action='store_true',
        help="Generate a sample configuration file"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help="Output directory or file path"
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    
    parser.add_argument(
        '--sequential', '-s',
        action='store_true',
        help="Run simulations sequentially instead of parallel"
    )
    
    parser.add_argument(
        '--snapshots',
        action='store_true',
        help="Generate network snapshots as SVG (uses snapshot_config from YAML)"
    )
    
    parser.add_argument(
        '--snapshot-steps',
        type=str,
        default="0,25,50,75,100",
        help="Comma-separated list of steps to capture for snapshots"
    )
    
    parser.add_argument(
        '--snapshot-combo',
        type=int,
        default=0,
        help="Parameter combination index for snapshots (default: 0)"
    )
    
    parser.add_argument(
        '--snapshot-trial',
        type=int,
        default=0,
        help="Trial index for snapshots (default: 0)"
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Suppress progress output"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Verbose output"
    )
    
    return parser


def print_experiment_summary(config: ExperimentConfig) -> None:
    """Print experiment summary."""
    summary = config.get_summary()
    
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {summary['name']}")
    print("=" * 60)
    
    if summary['description']:
        print(f"\n{summary['description']}")
    
    print(f"\n📊 Experiment Summary:")
    print(f"   Parameter combinations: {summary['n_parameter_combinations']}")
    print(f"   Trials per combination: {summary['n_trials_per_combination']}")
    print(f"   Total simulations: {summary['total_simulations']}")
    print(f"   Ticks per simulation: {summary['n_ticks']}")
    
    if summary['parameter_ranges']:
        print(f"\n🔄 Parameter Ranges:")
        for param, values in summary['parameter_ranges'].items():
            print(f"   {param}: {values}")
    
    if summary['fixed_parameters']:
        print(f"\n📌 Fixed Parameters:")
        for param, value in summary['fixed_parameters'].items():
            print(f"   {param}: {value}")
    
    print()


def progress_callback_cli(completed: int, total: int, quiet: bool = False) -> None:
    """Progress callback for CLI."""
    if not quiet:
        pct = completed / total * 100
        bar_len = 40
        filled = int(bar_len * completed / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {completed}/{total} ({pct:.1f}%)", end="", flush=True)


def run_experiment(args: argparse.Namespace) -> None:
    """Run experiment from configuration."""
    if not args.config:
        print("Error: Configuration file required. Use --generate-sample to create one.")
        sys.exit(1)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    if not args.quiet:
        print(f"Loading configuration from {config_path}...")
    
    config = ExperimentConfig.from_yaml(str(config_path))
    
    # Override output directory if specified
    if args.output:
        config.output_dir = args.output
    
    # Print summary
    if not args.quiet:
        print_experiment_summary(config)
    
    # Create runner
    runner = ExperimentRunner(config)
    
    # Run experiment
    if not args.quiet:
        print("🚀 Running experiment...")
    
    start_time = datetime.now()
    
    if args.sequential:
        if not args.quiet:
            print("   (Running sequentially)")
        df = runner.run_sequential(
            progress_callback=lambda c, t: progress_callback_cli(c, t, args.quiet)
        )
    else:
        workers = args.workers
        if not args.quiet:
            print(f"   (Running with {workers or 'auto'} parallel workers)")
        df = runner.run(
            n_workers=workers,
            progress_callback=lambda c, t: progress_callback_cli(c, t, args.quiet)
        )
    
    if not args.quiet:
        print()  # New line after progress bar
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save results and config to experiment folder
    csv_path = runner.save_results_csv()
    yaml_path = runner.save_config_yaml()
    
    if not args.quiet:
        exp_folder = Path(csv_path).parent
        print(f"\n✅ Experiment complete!")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   📁 Results folder: {exp_folder}")
        print(f"   📄 CSV: {Path(csv_path).name}")
        print(f"   📋 YAML: {Path(yaml_path).name}")
    
    # Print summary statistics
    if args.verbose and not args.quiet:
        summary_df = runner.get_summary_dataframe()
        if not summary_df.empty:
            print("\n📈 Summary Statistics (Final Tick):")
            print(summary_df.to_string())
    
    # Generate network snapshots if requested
    if args.snapshots:
        if not args.quiet:
            print("\n📸 Generating network snapshots...")
        
        # Get parameters for snapshot
        param_grid = config.get_parameter_grid()
        combo_idx = min(args.snapshot_combo, len(param_grid) - 1)
        params = param_grid[combo_idx]
        
        # Calculate seed
        seed = config.base_seed + combo_idx * 1000 + args.snapshot_trial
        
        # Parse steps
        steps = [int(s.strip()) for s in args.snapshot_steps.split(',')]
        
        # Generate snapshots in experiment-specific folder
        svg_files = save_network_snapshots_svg(
            params=params,
            seed=seed,
            steps=steps,
            output_dir=config.output_dir,
            n_ticks=config.n_ticks,
            experiment_name=config.name
        )
        
        if not args.quiet:
            snapshot_dir = Path(config.output_dir) / config.name / "snapshots"
            print(f"   Saved {len(svg_files)} SVG snapshots to: {snapshot_dir}")
            for f in svg_files:
                print(f"     - {Path(f).name}")


def generate_sample(args: argparse.Namespace) -> None:
    """Generate sample configuration file."""
    config = create_sample_config()
    
    output_path = args.output if args.output else "experiment_config.yaml"
    config.to_yaml(output_path)
    
    print(f"✅ Sample configuration saved to: {output_path}")
    print("\nEdit this file to customize your experiment, then run:")
    print(f"   python run_experiment.py {output_path}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.generate_sample:
        generate_sample(args)
    elif args.config:
        run_experiment(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
