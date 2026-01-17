"""
Union ABM Analytics Module

This module handles data persistence, logging, and metric calculations
for the union organizing simulation.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import networkx as nx

from .engine import UnionSim, AgentState


class SimulationLogger:
    """
    Logger for tracking simulation metrics over time.
    
    Handles CSV logging of per-tick metrics and provides methods
    for data export and analysis.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the simulation logger.
        
        Args:
            output_dir: Directory for output files. Defaults to current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[Dict] = []
        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_metrics(self, metrics: Dict) -> None:
        """
        Log metrics for a single tick.
        
        Args:
            metrics: Dictionary of metrics from UnionSim.get_metrics()
        """
        # Add timestamp
        metrics_with_time = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(metrics_with_time)
    
    def log_simulation_run(self, metrics_history: List[Dict]) -> None:
        """
        Log metrics from a complete simulation run.
        
        Args:
            metrics_history: List of metric dictionaries from simulation
        """
        for metrics in metrics_history:
            self.log_metrics(metrics)
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert logged metrics to a pandas DataFrame.
        
        Returns:
            DataFrame with all logged metrics
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics_history)
    
    def save_csv(self, filename: Optional[str] = None) -> Path:
        """
        Save metrics to a CSV file.
        
        Args:
            filename: Output filename. Defaults to run_id based name.
        
        Returns:
            Path to the saved CSV file
        """
        if filename is None:
            filename = f"union_sim_{self.run_id}.csv"
        
        filepath = self.output_dir / filename
        df = self.get_dataframe()
        
        if not df.empty:
            df.to_csv(filepath, index=False)
        
        return filepath
    
    def append_to_csv(self, filepath: Union[str, Path], metrics: Dict) -> None:
        """
        Append a single metrics row to an existing CSV file.
        
        Args:
            filepath: Path to CSV file
            metrics: Metrics dictionary to append
        """
        filepath = Path(filepath)
        file_exists = filepath.exists()
        
        metrics_with_time = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_with_time.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_with_time)
    
    def clear(self) -> None:
        """Clear all logged metrics."""
        self.metrics_history = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


class JSONPersistence:
    """
    Handles saving and loading simulation state to/from JSON.
    
    Enables reproducible experiments by preserving complete simulation state.
    """
    
    @staticmethod
    def save_state(
        sim: UnionSim, 
        filepath: Union[str, Path],
        include_history: bool = False
    ) -> Path:
        """
        Save simulation state to a JSON file.
        
        Args:
            sim: UnionSim instance to save
            filepath: Output file path
            include_history: Whether to include agent energy/state history
        
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = sim.get_state_dict()
        
        # Optionally include history
        if include_history:
            state_dict['agent_history'] = {
                aid: {
                    'energy_history': agent.energy_history,
                    'state_history': [s.value for s in agent.state_history]
                }
                for aid, agent in sim.agents.items()
            }
        
        # Add metadata
        state_dict['metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'version': '1.0',
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        return filepath
    
    @staticmethod
    def load_state(filepath: Union[str, Path]) -> UnionSim:
        """
        Load simulation state from a JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            Reconstructed UnionSim instance
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        # Remove metadata before reconstruction
        state_dict.pop('metadata', None)
        state_dict.pop('agent_history', None)
        
        return UnionSim.from_state_dict(state_dict)
    
    @staticmethod
    def save_experiment_config(
        config: Dict,
        filepath: Union[str, Path]
    ) -> Path:
        """
        Save experiment configuration to JSON.
        
        Args:
            config: Dictionary of experiment parameters
            filepath: Output file path
        
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config['metadata'] = {
            'created_at': datetime.now().isoformat(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        return filepath


class MetricsCalculator:
    """
    Advanced metrics calculations for the Union ABM.
    
    Provides methods for calculating fitness metrics and structural
    analysis of the organizing network.
    """
    
    @staticmethod
    def calculate_structural_resilience(
        metrics_before_stop: List[Dict],
        metrics_after_stop: List[Dict],
        metric_name: str = 'total_energy'
    ) -> Dict:
        """
        Calculate structural resilience after stop test.
        
        Measures how well the network maintains its state after
        organizer actions cease.
        
        Args:
            metrics_before_stop: Metrics history before stop test
            metrics_after_stop: Metrics history after stop test
            metric_name: Which metric to analyze
        
        Returns:
            Dictionary with resilience metrics
        """
        if not metrics_after_stop:
            return {'resilience_score': None, 'decay_rate': None}
        
        # Get metric values
        before_values = [m[metric_name] for m in metrics_before_stop]
        after_values = [m[metric_name] for m in metrics_after_stop]
        
        # Peak value before stop
        peak_value = max(before_values) if before_values else 0
        
        # Final value after stop test
        final_value = after_values[-1] if after_values else 0
        
        # Value at stop time
        stop_value = before_values[-1] if before_values else 0
        
        # Calculate resilience score (0-1, how much was retained)
        if stop_value > 0:
            resilience_score = final_value / stop_value
        else:
            resilience_score = 0
        
        # Calculate decay rate (exponential decay fit)
        if len(after_values) > 1:
            try:
                # Simple linear decay rate approximation
                decay_per_tick = (stop_value - final_value) / len(after_values)
                decay_rate = decay_per_tick / stop_value if stop_value > 0 else 0
            except:
                decay_rate = None
        else:
            decay_rate = None
        
        return {
            'resilience_score': resilience_score,
            'decay_rate': decay_rate,
            'peak_value': peak_value,
            'stop_value': stop_value,
            'final_value': final_value,
            'ticks_after_stop': len(after_values),
        }
    
    @staticmethod
    def calculate_mobilization_velocity(
        metrics_history: List[Dict],
        window: int = 10
    ) -> List[float]:
        """
        Calculate the rate of change in mobilization over time.
        
        Args:
            metrics_history: List of metrics dictionaries
            window: Rolling window size for smoothing
        
        Returns:
            List of velocity values
        """
        if len(metrics_history) < 2:
            return []
        
        mobilized_counts = [m['mobilized_count'] for m in metrics_history]
        velocities = []
        
        for i in range(1, len(mobilized_counts)):
            velocity = mobilized_counts[i] - mobilized_counts[i-1]
            velocities.append(velocity)
        
        # Apply rolling average smoothing
        if len(velocities) >= window:
            smoothed = pd.Series(velocities).rolling(window).mean().tolist()
            return smoothed
        
        return velocities
    
    @staticmethod
    def calculate_network_metrics(graph: nx.Graph) -> Dict:
        """
        Calculate comprehensive network structure metrics.
        
        Args:
            graph: NetworkX graph
        
        Returns:
            Dictionary of network metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['n_nodes'] = graph.number_of_nodes()
        metrics['n_edges'] = graph.number_of_edges()
        metrics['density'] = nx.density(graph)
        
        # Connectivity
        if graph.number_of_nodes() > 0:
            metrics['avg_degree'] = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        else:
            metrics['avg_degree'] = 0
        
        # Clustering
        metrics['avg_clustering'] = nx.average_clustering(graph)
        
        # Path-based metrics (only for connected graphs)
        if nx.is_connected(graph):
            metrics['diameter'] = nx.diameter(graph)
            metrics['avg_path_length'] = nx.average_shortest_path_length(graph)
        else:
            # For disconnected graphs, use largest component
            largest_cc = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc)
            if len(largest_cc) > 1:
                metrics['diameter'] = nx.diameter(subgraph)
                metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph)
            else:
                metrics['diameter'] = 0
                metrics['avg_path_length'] = 0
        
        # Efficiency
        metrics['global_efficiency'] = nx.global_efficiency(graph)
        metrics['local_efficiency'] = nx.local_efficiency(graph)
        
        return metrics
    
    @staticmethod
    def calculate_department_analysis(sim: UnionSim) -> Dict:
        """
        Analyze organizing success by department.
        
        Args:
            sim: UnionSim instance
        
        Returns:
            Dictionary with per-department analysis
        """
        dept_stats = {}
        
        for dept_id in range(sim.n_departments):
            # Get agents in this department
            dept_agents = [
                agent for agent in sim.agents.values()
                if agent.department_id == dept_id
            ]
            
            if not dept_agents:
                continue
            
            n_agents = len(dept_agents)
            energies = [a.energy for a in dept_agents]
            
            # Count by state
            n_organizers = sum(1 for a in dept_agents if a.state == AgentState.ORGANIZER)
            n_mobilized = sum(1 for a in dept_agents if a.state == AgentState.MOBILIZED)
            n_passive = sum(1 for a in dept_agents if a.state == AgentState.PASSIVE)
            
            dept_stats[dept_id] = {
                'n_agents': n_agents,
                'n_organizers': n_organizers,
                'n_mobilized': n_mobilized,
                'n_passive': n_passive,
                'organizer_pct': n_organizers / n_agents,
                'mobilized_pct': n_mobilized / n_agents,
                'passive_pct': n_passive / n_agents,
                'mean_energy': np.mean(energies),
                'std_energy': np.std(energies),
                'min_energy': min(energies),
                'max_energy': max(energies),
            }
        
        return dept_stats


class ExperimentRunner:
    """
    Utility class for running batch experiments.
    
    Handles parameter sweeps and multiple simulation runs
    for statistical analysis.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "experiments"):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Base directory for experiment outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_parameter_sweep(
        self,
        base_params: Dict,
        sweep_param: str,
        sweep_values: List,
        n_ticks: int = 100,
        n_replications: int = 5,
        stop_test_tick: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run a parameter sweep experiment.
        
        Args:
            base_params: Base simulation parameters
            sweep_param: Parameter to vary
            sweep_values: Values to test for sweep_param
            n_ticks: Number of ticks per simulation
            n_replications: Number of replications per parameter value
            stop_test_tick: Optional tick to activate stop test
        
        Returns:
            DataFrame with results from all runs
        """
        all_results = []
        
        for value in sweep_values:
            for rep in range(n_replications):
                # Set up parameters
                params = base_params.copy()
                params[sweep_param] = value
                params['seed'] = base_params.get('seed', 0) + rep
                
                # Run simulation
                sim = UnionSim(**params)
                metrics_history = sim.run(n_ticks, stop_test_tick)
                
                # Get final metrics
                final_metrics = metrics_history[-1]
                final_metrics['sweep_param'] = sweep_param
                final_metrics['sweep_value'] = value
                final_metrics['replication'] = rep
                
                all_results.append(final_metrics)
        
        return pd.DataFrame(all_results)
    
    def run_alpha_comparison(
        self,
        base_params: Dict,
        alpha_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        n_ticks: int = 100,
        n_replications: int = 10,
    ) -> pd.DataFrame:
        """
        Compare organizing (high alpha) vs mobilizing (low alpha) strategies.
        
        Args:
            base_params: Base simulation parameters
            alpha_values: Alpha values to test
            n_ticks: Number of ticks per simulation
            n_replications: Number of replications
        
        Returns:
            DataFrame with comparison results
        """
        return self.run_parameter_sweep(
            base_params=base_params,
            sweep_param='alpha',
            sweep_values=alpha_values,
            n_ticks=n_ticks,
            n_replications=n_replications,
        )
    
    def save_experiment_results(
        self,
        results: pd.DataFrame,
        experiment_name: str,
        config: Optional[Dict] = None,
    ) -> Dict[str, Path]:
        """
        Save experiment results and configuration.
        
        Args:
            results: DataFrame of results
            experiment_name: Name for this experiment
            config: Optional configuration dictionary
        
        Returns:
            Dictionary of saved file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Save results CSV
        results_path = exp_dir / "results.csv"
        results.to_csv(results_path, index=False)
        paths['results'] = results_path
        
        # Save configuration
        if config:
            config_path = exp_dir / "config.json"
            JSONPersistence.save_experiment_config(config, config_path)
            paths['config'] = config_path
        
        return paths


def generate_summary_statistics(metrics_history: List[Dict]) -> Dict:
    """
    Generate summary statistics from a simulation run.
    
    Args:
        metrics_history: List of metrics dictionaries from simulation
    
    Returns:
        Dictionary of summary statistics
    """
    if not metrics_history:
        return {}
    
    df = pd.DataFrame(metrics_history)
    
    summary = {
        'n_ticks': len(metrics_history),
        'final_tick': metrics_history[-1]['tick'],
        
        # Energy statistics
        'final_total_energy': metrics_history[-1]['total_energy'],
        'max_total_energy': df['total_energy'].max(),
        'mean_total_energy': df['total_energy'].mean(),
        
        # State counts
        'final_organizer_count': metrics_history[-1]['organizer_count'],
        'final_mobilized_count': metrics_history[-1]['mobilized_count'],
        'final_passive_count': metrics_history[-1]['passive_count'],
        'max_organizer_count': df['organizer_count'].max(),
        
        # Fitness metrics
        'final_reach': metrics_history[-1]['reach'],
        'final_dsi': metrics_history[-1]['dsi'],
        'final_lwr': metrics_history[-1]['lwr'],
        
        # Network metrics
        'final_clustering': metrics_history[-1]['clustering_coefficient'],
        'final_efficiency': metrics_history[-1]['global_efficiency'],
        
        # Time series characteristics
        'tick_to_50pct_mobilized': _find_threshold_tick(df, 'reach', 0.5),
        'tick_to_max_organizers': df['organizer_count'].idxmax(),
    }
    
    return summary


def _find_threshold_tick(df: pd.DataFrame, column: str, threshold: float) -> Optional[int]:
    """Find the first tick where a metric exceeds a threshold."""
    above_threshold = df[df[column] >= threshold]
    if len(above_threshold) > 0:
        return above_threshold.iloc[0]['tick']
    return None
