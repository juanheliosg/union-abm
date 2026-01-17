"""
McAlevey ABM Command Line Interface

Run the simulation from the command line with configurable parameters.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from union_abm.engine import UnionSim
from union_abm.analytics import (
    SimulationLogger,
    JSONPersistence,
    ExperimentRunner,
    generate_summary_statistics
)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="McAlevey ABM: Agent-Based Model of Labor Organizing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python main.py --ticks 100

  # Run with high organizing strategy
  python main.py --alpha 1.0 --ticks 200

  # Run with mobilizing strategy and save to CSV
  python main.py --alpha 0.0 --ticks 200 --output results.csv

  # Run with stop test at tick 50
  python main.py --ticks 100 --stop-test 50

  # Run parameter sweep for alpha
  python main.py --sweep alpha --sweep-values 0.0,0.25,0.5,0.75,1.0 --ticks 100

  # Load and continue from saved state
  python main.py --load state.json --ticks 50
        """
    )
    
    # Main operation modes
    mode_group = parser.add_argument_group("Operation Mode")
    mode_group.add_argument(
        "--ticks", "-t",
        type=int,
        default=100,
        help="Number of simulation ticks to run (default: 100)"
    )
    mode_group.add_argument(
        "--load",
        type=str,
        metavar="FILE",
        help="Load simulation state from JSON file"
    )
    mode_group.add_argument(
        "--sweep",
        type=str,
        choices=['alpha', 'beta', 'delta', 'p_in', 'p_out', 'omega', 'mobilizing_threshold', 'contagion_power', 'persistence'],
        help="Run parameter sweep experiment"
    )
    mode_group.add_argument(
        "--sweep-values",
        type=str,
        help="Comma-separated values for parameter sweep"
    )
    mode_group.add_argument(
        "--replications",
        type=int,
        default=5,
        help="Number of replications for parameter sweep (default: 5)"
    )
    
    # Network parameters
    network_group = parser.add_argument_group("Network Parameters")
    network_group.add_argument(
        "--departments", "-K",
        type=int,
        default=5,
        help="Number of departments (default: 5)"
    )
    network_group.add_argument(
        "--agents-per-dept",
        type=int,
        default=20,
        help="Agents per department (default: 20)"
    )
    network_group.add_argument(
        "--p-in",
        type=float,
        default=0.4,
        help="Intra-department connection probability (default: 0.4)"
    )
    network_group.add_argument(
        "--p-out",
        type=float,
        default=0.05,
        help="Inter-department connection probability (default: 0.05, range: 0.001-1.0)"
    )
    network_group.add_argument(
        "--seed-size",
        type=int,
        default=3,
        help="Initial organizer seed team size (default: 3)"
    )
    
    # Strategy parameters
    strategy_group = parser.add_argument_group("Strategy Parameters")
    strategy_group.add_argument(
        "--alpha", "-a",
        type=float,
        default=0.8,
        help="Organizing vs mobilizing: 1=pure organizing, 0=pure mobilizing (default: 0.8)"
    )
    
    # Energy dynamics
    energy_group = parser.add_argument_group("Energy Dynamics")
    energy_group.add_argument(
        "--beta", "-b",
        type=float,
        default=0.3,
        help="Complex contagion coefficient (default: 0.3, range: 0-3)"
    )
    energy_group.add_argument(
        "--contagion-power",
        type=float,
        default=1.0,
        help="Exponent for complex contagion term (default: 1.0, range: 0-3)"
    )
    energy_group.add_argument(
        "--delta", "-d",
        type=float,
        default=0.01,
        help="Energy decay rate (default: 0.01)"
    )
    energy_group.add_argument(
        "--mobilizing-threshold",
        type=float,
        default=0.3,
        help="Energy threshold to become mobilized (default: 0.3)"
    )
    energy_group.add_argument(
        "--omega",
        type=float,
        default=0.8,
        help="Organizer threshold (default: 0.8)"
    )
    energy_group.add_argument(
        "--persistence",
        type=int,
        default=3,
        help="Persistence gate: ticks above threshold to become organizer (default: 3)"
    )
    energy_group.add_argument(
        "--outreach-energy",
        type=float,
        default=0.5,
        help="Energy boost from 1-to-1 organizing (default: 0.5)"
    )
    energy_group.add_argument(
        "--broadcast-energy",
        type=float,
        default=0.1,
        help="Energy boost from broadcast mobilizing (default: 0.1)"
    )
    energy_group.add_argument(
        "--broadcast-reach-ratio",
        type=float,
        default=0.1,
        help="Fraction of agents reached by broadcast action (default: 0.1)"
    )
    
    # Stop test
    stop_group = parser.add_argument_group("Stop Test")
    stop_group.add_argument(
        "--stop-test",
        type=int,
        metavar="TICK",
        help="Activate stop test at specified tick"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        metavar="FILE",
        help="Output CSV file for metrics"
    )
    output_group.add_argument(
        "--save-state",
        type=str,
        metavar="FILE",
        help="Save final state to JSON file"
    )
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files (default: output)"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with detailed metrics"
    )
    
    # Reproducibility
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser


def run_single_simulation(args: argparse.Namespace) -> None:
    """Run a single simulation with the given parameters."""
    # Build simulation parameters
    params = {
        'n_departments': args.departments,
        'agents_per_dept': args.agents_per_dept,
        'p_in': args.p_in,
        'p_out': args.p_out,
        'seed_size': args.seed_size,
        'alpha': args.alpha,
        'beta': args.beta,
        'delta': args.delta,
        'omega': args.omega,
        'mobilizing_threshold': args.mobilizing_threshold,
        'persistence_threshold': args.persistence,
        'outreach_energy': args.outreach_energy,
        'broadcast_energy': args.broadcast_energy,
        'broadcast_reach_ratio': args.broadcast_reach_ratio,
        'contagion_power': args.contagion_power,
        'seed': args.seed,
    }
    
    # Initialize simulation
    if args.load:
        if not args.quiet:
            print(f"Loading simulation state from {args.load}...")
        sim = JSONPersistence.load_state(args.load)
    else:
        if not args.quiet:
            print("Initializing new simulation...")
            print(f"  Network: {args.departments} departments × {args.agents_per_dept} agents")
            print(f"  Strategy: α = {args.alpha}")
        sim = UnionSim(**params)
        
        # Validate parameter balance
        validation = sim.validate_parameter_balance()
        if not validation['balanced']:
            print("\n" + "!"*60)
            print("PARAMETER BALANCE WARNING")
            print("!"*60)
            for warning in validation['warnings']:
                print(warning)
            if validation['recommendations']:
                print("\nRecommendations:")
                for rec in validation['recommendations']:
                    print(rec)
            print("\nContinuing simulation with current parameters...")
            print("!"*60 + "\n")
    
    # Run simulation
    if not args.quiet:
        print(f"\nRunning simulation for {args.ticks} ticks...")
    
    metrics_history = sim.run(args.ticks, stop_test_tick=args.stop_test)
    
    # Generate summary
    summary = generate_summary_statistics(metrics_history)
    
    if not args.quiet:
        print("\n" + "="*50)
        print("SIMULATION COMPLETE")
        print("="*50)
        print(f"Final tick: {summary['final_tick']}")
        print(f"Organizers: {summary['final_organizer_count']}")
        print(f"Mobilized: {summary['final_mobilized_count']}")
        print(f"Reach: {summary['final_reach']:.1%}")
        print(f"DSI: {summary['final_dsi']:.3f}")
        print(f"LWR: {summary['final_lwr']:.3f}")
    
    if args.verbose:
        print("\nDetailed Metrics:")
        print(f"  Total Energy: {summary['final_total_energy']:.2f}")
        print(f"  Max Energy: {summary['max_total_energy']:.2f}")
        print(f"  Clustering: {summary['final_clustering']:.3f}")
        print(f"  Efficiency: {summary['final_efficiency']:.3f}")
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        logger = SimulationLogger(output_dir)
        logger.log_simulation_run(metrics_history)
        filepath = logger.save_csv(args.output)
        if not args.quiet:
            print(f"\nMetrics saved to: {filepath}")
    
    if args.save_state:
        filepath = JSONPersistence.save_state(sim, output_dir / args.save_state)
        if not args.quiet:
            print(f"State saved to: {filepath}")


def run_parameter_sweep(args: argparse.Namespace) -> None:
    """Run a parameter sweep experiment."""
    if not args.sweep_values:
        print("Error: --sweep-values required for parameter sweep", file=sys.stderr)
        sys.exit(1)
    
    # Parse sweep values
    try:
        sweep_values = [float(v.strip()) for v in args.sweep_values.split(',')]
    except ValueError:
        print("Error: Invalid sweep values. Use comma-separated numbers.", file=sys.stderr)
        sys.exit(1)
    
    # Build base parameters
    base_params = {
        'n_departments': args.departments,
        'agents_per_dept': args.agents_per_dept,
        'p_in': args.p_in,
        'p_out': args.p_out,
        'seed_size': args.seed_size,
        'alpha': args.alpha,
        'beta': args.beta,
        'delta': args.delta,
        'omega': args.omega,
        'mobilizing_threshold': args.mobilizing_threshold,
        'persistence_threshold': args.persistence,
        'outreach_energy': args.outreach_energy,
        'broadcast_energy': args.broadcast_energy,
        'contagion_power': args.contagion_power,
        'seed': args.seed,
    }
    
    if not args.quiet:
        print(f"Running parameter sweep for '{args.sweep}'")
        print(f"  Values: {sweep_values}")
        print(f"  Replications: {args.replications}")
        print(f"  Ticks: {args.ticks}")
    
    # Run sweep
    runner = ExperimentRunner(args.output_dir)
    results = runner.run_parameter_sweep(
        base_params=base_params,
        sweep_param=args.sweep,
        sweep_values=sweep_values,
        n_ticks=args.ticks,
        n_replications=args.replications,
        stop_test_tick=args.stop_test,
    )
    
    # Save and display results
    config = {
        'base_params': base_params,
        'sweep_param': args.sweep,
        'sweep_values': sweep_values,
        'n_ticks': args.ticks,
        'n_replications': args.replications,
    }
    
    paths = runner.save_experiment_results(results, f"sweep_{args.sweep}", config)
    
    if not args.quiet:
        print("\n" + "="*50)
        print("PARAMETER SWEEP COMPLETE")
        print("="*50)
        
        # Show summary by sweep value
        summary = results.groupby('sweep_value').agg({
            'organizer_count': ['mean', 'std'],
            'reach': ['mean', 'std'],
            'dsi': ['mean', 'std'],
        }).round(3)
        
        print(f"\nResults by {args.sweep}:")
        print(summary.to_string())
        
        print(f"\nResults saved to: {paths['results']}")


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.sweep and not args.sweep_values:
        parser.error("--sweep-values is required when using --sweep")
    
    try:
        if args.sweep:
            run_parameter_sweep(args)
        else:
            run_single_simulation(args)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
