"""
Union ABM Experiment Runner Module

This module handles batch experiment execution with parameter sweeps,
parallel processing, and result aggregation.
"""

import itertools
import yaml
import csv
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pandas as pd
import numpy as np

from .engine import UnionSim, AgentState


@dataclass
class ExperimentConfig:
    """
    Configuration for a batch experiment.
    
    Attributes:
        name: Experiment name
        description: Description of the experiment
        parameter_ranges: Dict of parameter names to lists of values to sweep
        fixed_parameters: Dict of parameter names to fixed values
        n_trials: Number of trials (different seeds) per parameter combination
        n_ticks: Number of simulation ticks to run
        base_seed: Starting seed for reproducibility
        output_dir: Directory to save results
        snapshot_config: Optional config for SVG snapshots
    """
    name: str = "experiment"
    description: str = ""
    parameter_ranges: Dict[str, List[Any]] = field(default_factory=dict)
    fixed_parameters: Dict[str, Any] = field(default_factory=dict)
    n_trials: int = 5
    n_ticks: int = 100
    base_seed: int = 42
    output_dir: str = "output/experiments"
    snapshot_config: Optional[Dict] = None  # {"params": {...}, "seed": int, "steps": [0, 25, 50, 75, 100]}
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load experiment configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save experiment configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def get_parameter_grid(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from ranges.
        
        Returns:
            List of parameter dictionaries, one per combination
        """
        if not self.parameter_ranges:
            return [self.fixed_parameters.copy()]
        
        # Get keys and value lists
        keys = list(self.parameter_ranges.keys())
        value_lists = [self.parameter_ranges[k] for k in keys]
        
        # Generate all combinations
        combinations = list(itertools.product(*value_lists))
        
        # Create parameter dicts
        param_dicts = []
        for combo in combinations:
            params = self.fixed_parameters.copy()
            for key, value in zip(keys, combo):
                params[key] = value
            param_dicts.append(params)
        
        return param_dicts
    
    def get_total_simulations(self) -> int:
        """Calculate total number of simulations to run."""
        n_combinations = len(self.get_parameter_grid())
        return n_combinations * self.n_trials
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary information."""
        grid = self.get_parameter_grid()
        return {
            'name': self.name,
            'description': self.description,
            'n_parameter_combinations': len(grid),
            'n_trials_per_combination': self.n_trials,
            'total_simulations': self.get_total_simulations(),
            'n_ticks': self.n_ticks,
            'parameter_ranges': self.parameter_ranges,
            'fixed_parameters': self.fixed_parameters,
        }


def run_single_simulation(args: Tuple) -> Dict[str, Any]:
    """
    Run a single simulation with given parameters.
    
    This function is designed to be called in parallel.
    
    Args:
        args: Tuple of (params_dict, seed, n_ticks, param_combo_id, trial_id)
    
    Returns:
        Dictionary with simulation results including all metrics by tick
    """
    params, seed, n_ticks, combo_id, trial_id = args
    
    # Create simulation with specific seed
    sim_params = params.copy()
    sim_params['seed'] = seed
    
    try:
        sim = UnionSim(**sim_params)
        metrics_history = sim.run(n_ticks)
        
        # Add metadata to each tick's metrics
        results = []
        for metrics in metrics_history:
            result = {
                'combo_id': combo_id,
                'trial_id': trial_id,
                'seed': seed,
                **{f'param_{k}': v for k, v in params.items()},
                **metrics
            }
            results.append(result)
        
        return {
            'success': True,
            'combo_id': combo_id,
            'trial_id': trial_id,
            'seed': seed,
            'params': params,
            'results': results,
            'final_metrics': metrics_history[-1] if metrics_history else None
        }
    except Exception as e:
        return {
            'success': False,
            'combo_id': combo_id,
            'trial_id': trial_id,
            'seed': seed,
            'params': params,
            'error': str(e)
        }


class ExperimentRunner:
    """
    Runs batch experiments with parameter sweeps and parallel execution.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[Dict] = []
        self.all_tick_results: List[Dict] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    def run(self, n_workers: Optional[int] = None, progress_callback=None) -> pd.DataFrame:
        """
        Run all simulations in the experiment.
        
        Args:
            n_workers: Number of parallel workers. Defaults to CPU count - 1
            progress_callback: Optional callback function(completed, total) for progress updates
        
        Returns:
            DataFrame with all results
        """
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        
        self.start_time = datetime.now()
        
        # Generate all simulation tasks
        param_grid = self.config.get_parameter_grid()
        tasks = []
        
        for combo_id, params in enumerate(param_grid):
            for trial_id in range(self.config.n_trials):
                seed = self.config.base_seed + combo_id * 1000 + trial_id
                tasks.append((params, seed, self.config.n_ticks, combo_id, trial_id))
        
        total_tasks = len(tasks)
        completed = 0
        
        # Run simulations in parallel
        self.results = []
        self.all_tick_results = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_single_simulation, task): task for task in tasks}
            
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
                
                if result['success']:
                    self.all_tick_results.extend(result['results'])
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_tasks)
        
        self.end_time = datetime.now()
        
        return self.get_results_dataframe()
    
    def run_sequential(self, progress_callback=None) -> pd.DataFrame:
        """
        Run all simulations sequentially (useful for debugging or Streamlit).
        
        Args:
            progress_callback: Optional callback function(completed, total) for progress updates
        
        Returns:
            DataFrame with all results
        """
        self.start_time = datetime.now()
        
        # Generate all simulation tasks
        param_grid = self.config.get_parameter_grid()
        tasks = []
        
        for combo_id, params in enumerate(param_grid):
            for trial_id in range(self.config.n_trials):
                seed = self.config.base_seed + combo_id * 1000 + trial_id
                tasks.append((params, seed, self.config.n_ticks, combo_id, trial_id))
        
        total_tasks = len(tasks)
        
        # Run simulations sequentially
        self.results = []
        self.all_tick_results = []
        
        for i, task in enumerate(tasks):
            result = run_single_simulation(task)
            self.results.append(result)
            
            if result['success']:
                self.all_tick_results.extend(result['results'])
            
            if progress_callback:
                progress_callback(i + 1, total_tasks)
        
        self.end_time = datetime.now()
        
        return self.get_results_dataframe()
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        if not self.all_tick_results:
            return pd.DataFrame()
        return pd.DataFrame(self.all_tick_results)
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Get summary statistics aggregated by parameter combination.
        
        Returns DataFrame with one row per parameter combination,
        showing mean, std, min, max of key metrics across trials.
        """
        df = self.get_results_dataframe()
        if df.empty:
            return pd.DataFrame()
        
        # Get final tick for each simulation
        final_ticks = df.groupby(['combo_id', 'trial_id']).last().reset_index()
        
        # Identify parameter columns
        param_cols = [c for c in final_ticks.columns if c.startswith('param_')]
        
        # Aggregate by parameter combination
        agg_funcs = {
            'reach': ['mean', 'std', 'min', 'max'],
            'total_energy': ['mean', 'std', 'min', 'max'],
            'organizer_count': ['mean', 'std', 'min', 'max'],
            'mobilized_count': ['mean', 'std', 'min', 'max'],
            'dsi': ['mean', 'std'],
            'lwr': ['mean', 'std'],
        }
        
        # Group by combo_id and parameter columns
        group_cols = ['combo_id'] + param_cols
        summary = final_ticks.groupby(group_cols).agg(agg_funcs)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        
        return summary.reset_index()
    
    def save_results_csv(self, filepath: Optional[str] = None) -> str:
        """
        Save all tick-level results to CSV.
        
        Args:
            filepath: Output path. Defaults to output_dir/experiment_name/results.csv
        
        Returns:
            Path to saved file
        """
        if filepath is None:
            # Create experiment-specific folder
            exp_dir = Path(self.config.output_dir) / self.config.name
            exp_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = exp_dir / f"{self.config.name}_{timestamp}_results.csv"
        
        df = self.get_results_dataframe()
        df.to_csv(filepath, index=False)
        return str(filepath)
    
    def save_config_yaml(self, filepath: Optional[str] = None) -> str:
        """
        Save experiment configuration to YAML.
        
        Args:
            filepath: Output path. Defaults to output_dir/experiment_name/config.yaml
        
        Returns:
            Path to saved file
        """
        if filepath is None:
            # Create experiment-specific folder
            exp_dir = Path(self.config.output_dir) / self.config.name
            exp_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = exp_dir / f"{self.config.name}_{timestamp}_config.yaml"
        
        self.config.to_yaml(filepath)
        return str(filepath)
    
    def get_execution_time(self) -> Optional[float]:
        """Get execution time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


def save_network_snapshots_svg(
    params: Dict[str, Any],
    seed: int,
    steps: List[int],
    output_dir: str,
    n_ticks: Optional[int] = None,
    experiment_name: Optional[str] = None
) -> List[str]:
    """
    Run a simulation and save network snapshots at specified steps as SVG.
    
    Args:
        params: Simulation parameters
        seed: Random seed
        steps: List of tick numbers to capture
        output_dir: Base directory to save SVG files
        n_ticks: Total ticks to run (defaults to max of steps)
        experiment_name: Optional experiment name to create subfolder
    
    Returns:
        List of paths to saved SVG files
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    
    # Determine number of ticks needed
    if n_ticks is None:
        n_ticks = max(steps) if steps else 100
    
    # Create simulation
    sim_params = params.copy()
    sim_params['seed'] = seed
    sim = UnionSim(**sim_params)
    
    # Create output directory (with experiment subfolder if provided)
    if experiment_name:
        output_path = Path(output_dir) / experiment_name / "snapshots"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Compute fixed layout
    pos = nx.spring_layout(sim.graph, seed=42, k=2/np.sqrt(len(sim.graph.nodes())), iterations=100)
    
    # Color mapping
    state_colors = {
        AgentState.PASSIVE: '#3498db',      # Blue
        AgentState.MOBILIZED: '#f39c12',    # Orange
        AgentState.ORGANIZER: '#e74c3c',    # Red
    }
    
    saved_files = []
    
    # Capture initial state if 0 in steps
    if 0 in steps:
        filepath = _save_snapshot_svg(sim, pos, state_colors, output_path, 0, params, seed)
        saved_files.append(filepath)
    
    # Run simulation and capture at specified steps
    for tick in range(1, n_ticks + 1):
        sim.step()
        if tick in steps:
            filepath = _save_snapshot_svg(sim, pos, state_colors, output_path, tick, params, seed)
            saved_files.append(filepath)
    
    return saved_files


def _save_snapshot_svg(
    sim: UnionSim,
    pos: Dict,
    state_colors: Dict,
    output_path: Path,
    tick: int,
    params: Dict,
    seed: int
) -> str:
    """Save a single network snapshot as SVG."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Get node colors and sizes based on state
    node_colors = [state_colors[sim.agents[n].state] for n in sim.graph.nodes()]
    node_sizes = []
    for n in sim.graph.nodes():
        if sim.agents[n].state == AgentState.ORGANIZER:
            node_sizes.append(300)
        elif sim.agents[n].state == AgentState.MOBILIZED:
            node_sizes.append(200)
        else:
            node_sizes.append(100)
    
    # Draw network
    nx.draw_networkx_edges(sim.graph, pos, edge_color='#cccccc', alpha=0.5, width=0.5, ax=ax)
    nx.draw_networkx_nodes(sim.graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#3498db', label=f'Passive ({len(sim.get_passive())})'),
        mpatches.Patch(color='#f39c12', label=f'Mobilized ({len(sim.get_mobilized())})'),
        mpatches.Patch(color='#e74c3c', label=f'Organizer ({len(sim.get_organizers())})'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Title with key params
    key_params = ['alpha', 'beta', 'delta', 'p_in', 'p_out']
    param_str = ', '.join([f"{k}={params.get(k, 'N/A')}" for k in key_params if k in params])
    ax.set_title(f"Tick {tick} | Seed {seed}\n{param_str}", fontsize=12)
    
    ax.axis('off')
    
    # Save as SVG
    filename = f"network_tick{tick:04d}_seed{seed}.svg"
    filepath = output_path / filename
    fig.savefig(filepath, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    return str(filepath)


def create_sample_config() -> ExperimentConfig:
    """Create a sample experiment configuration."""
    return ExperimentConfig(
        name="alpha_sweep_experiment",
        description="Investigating the effect of organizing strategy (alpha) on campaign success",
        parameter_ranges={
            'alpha': [0.0, 0.5, 1.0],
            'contagion_power': [1.0, 2.0],
        },
        fixed_parameters={
            'n_departments': 5,
            'agents_per_dept': 20,
            'p_in': 0.4,
            'p_out': 0.05,
            'seed_size': 3,
            'beta': 0.3,
            'delta': 0.01,
            'omega': 0.8,
            'mobilizing_threshold': 0.3,
            'persistence_threshold': 3,
            'outreach_energy': 0.5,
            'broadcast_energy': 0.01,
            'broadcast_reach_ratio': 0.75,
            'organizer_potential_ratio': 0.2,
        },
        n_trials=5,
        n_ticks=100,
        base_seed=42,
        output_dir="output/experiments"
    )
