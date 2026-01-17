"""
Union ABM Streamlit Dashboard

Interactive GUI for running and visualizing the Union organizing simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tempfile
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO, StringIO
from typing import Optional, Tuple
import os

from union_abm.engine import UnionSim, AgentState
from union_abm.analytics import (
    SimulationLogger, 
    JSONPersistence, 
    MetricsCalculator,
    generate_summary_statistics
)
from union_abm.experiment import (
    ExperimentConfig,
    ExperimentRunner,
    save_network_snapshots_svg,
    create_sample_config
)
import yaml

# Page configuration
st.set_page_config(
    page_title="Union ABM Simulator",
    page_icon="✊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'sim' not in st.session_state:
        st.session_state.sim = None
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'logger' not in st.session_state:
        st.session_state.logger = SimulationLogger()
    if 'playback_tick' not in st.session_state:
        st.session_state.playback_tick = 0
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'playback_speed' not in st.session_state:
        st.session_state.playback_speed = 0.5
    if 'network_layout' not in st.session_state:
        st.session_state.network_layout = None
    # Experiment state
    if 'experiment_config' not in st.session_state:
        st.session_state.experiment_config = None
    if 'experiment_results' not in st.session_state:
        st.session_state.experiment_results = None
    if 'experiment_running' not in st.session_state:
        st.session_state.experiment_running = False
    if 'loaded_experiment_path' not in st.session_state:
        st.session_state.loaded_experiment_path = None


def create_sidebar():
    """Create the sidebar with parameter controls."""
    st.sidebar.title("⚙️ Simulation Parameters")
    
    st.sidebar.header("Network Structure")
    n_departments = st.sidebar.slider(
        "Number of Departments (K)", 
        min_value=2, max_value=10, value=5,
        help="Number of departments/blocks in the Stochastic Block Model"
    )
    agents_per_dept = st.sidebar.slider(
        "Agents per Department", 
        min_value=5, max_value=50, value=20,
        help="Number of agents in each department"
    )
    p_in = st.sidebar.slider(
        "P_in (Intra-department connectivity)", 
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Probability of connection within a department"
    )
    p_out = st.sidebar.slider(
        "P_out (Inter-department connectivity)", 
        min_value=0.001, max_value=0.2, value=0.001, step=0.001, format="%.3f",
        help="Probability of connection between departments"
    )
    
    st.sidebar.header("Initial Conditions")
    seed_size = st.sidebar.slider(
        "Seed Team Size", 
        min_value=1, max_value=10, value=3,
        help="Number of initial organizers in the seed clique"
    )
    organizer_potential_ratio = st.sidebar.slider(
        "Organizer Potential Ratio",
        min_value=0.1, max_value=1.0, value=0.3, step=0.05,
        help="Fraction of agents whose ideological range allows becoming organizer (e.g., 0.3 = 1 in 3.3 can become organizers)"
    )
    
    st.sidebar.header("Strategy Parameters")
    alpha = st.sidebar.slider(
        "α (Organizing vs Mobilizing)", 
        min_value=0.0, max_value=1.0, value=0.8, step=0.05,
        help="α=1: Pure 1-to-1 organizing, α=0: Pure broadcast mobilizing"
    )
    
    st.sidebar.header("Energy Dynamics")
    beta = st.sidebar.slider(
        "β social reinforcement weight", 
        min_value=0.0, max_value=3.0, value=0.3, step=0.05,
        help="Strength of social influence"
    )
    contagion_power = st.sidebar.slider(
        "Complex contagion exponent (p)",
        min_value=0.0, max_value=5.0, value=2.0, step=0.1,
        help="Exponent for the complex contagion term β(Ē)^p, the greater the more social reinforcement is needed"
    )
    delta = st.sidebar.slider(
        "δ (Energy Decay)", 
        min_value=0.0, max_value=0.4, value=0.01, step=0.001,
        help="Rate of energy decay when neighbors have low energy"
    )
    
    st.sidebar.header("Thresholds & Gates")
    mobilizing_threshold = st.sidebar.slider(
        "Mobilizing Threshold",
        min_value=-0.5, max_value=1.0, value=0.6, step=0.05,
        help="Energy threshold to transition from Passive to Mobilized"
    )
    omega = st.sidebar.slider(
        "Ω (Organizer Threshold)", 
        min_value=0.0, max_value=1.0, value=0.8, step=0.05,
        help="Energy threshold required to become an organizer"
    )
    persistence_threshold = st.sidebar.slider(
        "Persistence Gate (X)", 
        min_value=1, max_value=20, value=3,
        help="Ticks above Ω while targeted needed to become organizer"
    )
    
    st.sidebar.header("Energy Boosts")
    outreach_energy = st.sidebar.slider(
        "1-to-1 Energy Boost", 
        min_value=0.0, max_value=0.5, value=0.45, step=0.01,
        help="Energy added by 1-to-1 organizing"
    )
    broadcast_energy = st.sidebar.slider(
        "Broadcast Energy Boost", 
        min_value=0.0, max_value=0.2, value=0.01, step=0.005,
        help="Energy added by broadcast mobilizing"
    )
    broadcast_reach_ratio = st.sidebar.slider(
        "Broadcast Reach Ratio",
        min_value=0.01, max_value=1.0, value=0.3, step=0.01,
        help="Fraction of all agents reached by each broadcast action"
    )
    
    st.sidebar.header("Simulation Settings")
    random_seed = st.sidebar.number_input(
        "Random Seed", 
        min_value=0, max_value=99999, value=42,
        help="Seed for reproducibility"
    )
    
    # Add parameter balance check info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚖️ Parameter Balance")
    with st.sidebar.expander("Check Balance", expanded=False):
        st.markdown("""
        For fair strategy comparison:
        - **Strategy balance**: Total energy from 1-to-1 should ≈ broadcast
        - **Decay balance**: Decay should be in similar range to outreach
        
        Initialize simulation to see balance validation.
        """)
    
    return {
        'n_departments': n_departments,
        'agents_per_dept': agents_per_dept,
        'p_in': p_in,
        'p_out': p_out,
        'seed_size': seed_size,
        'alpha': alpha,
        'beta': beta,
        'delta': delta,
        'omega': omega,
        'mobilizing_threshold': mobilizing_threshold,
        'persistence_threshold': persistence_threshold,
        'outreach_energy': outreach_energy,
        'broadcast_energy': broadcast_energy,
        'broadcast_reach_ratio': broadcast_reach_ratio,
        'contagion_power': contagion_power,
        'organizer_potential_ratio': organizer_potential_ratio,
        'seed': random_seed,
    }


def get_cached_layout(sim: UnionSim, force_recompute: bool = False) -> dict:
    """
    Get or compute cached layout positions using Fruchterman-Reingold algorithm.
    
    Uses session state to cache layout for consistent visualization across ticks.
    The layout persists across simulation runs until a new simulation is initialized.
    
    Args:
        sim: The simulation object
        force_recompute: If True, recompute layout even if cached
    
    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    # Try to use session state caching (only works within Streamlit context)
    try:
        # Check if we have a cached layout and it matches current graph nodes
        if (not force_recompute and 
            st.session_state.network_layout is not None and
            set(st.session_state.network_layout.keys()) == set(sim.graph.nodes())):
            return st.session_state.network_layout
        
        # Compute new layout using Fruchterman-Reingold
        pos = nx.spring_layout(sim.graph, seed=42, k=2/np.sqrt(len(sim.graph.nodes())), iterations=100)
        
        # Cache in session state
        st.session_state.network_layout = pos
        
        return pos
    except AttributeError:
        # Running outside Streamlit context, just compute layout without caching
        return nx.spring_layout(sim.graph, seed=42, k=2/np.sqrt(len(sim.graph.nodes())), iterations=100)


def create_network_visualization(sim: UnionSim, color_by: str = "state") -> plt.Figure:
    """
    Create a fast network visualization using matplotlib with Fruchterman-Reingold layout.
    
    Args:
        sim: The simulation object
        color_by: "state" to color by agent state, "energy" to color by energy level
    
    Returns matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Get layout positions using Fruchterman-Reingold
    pos = get_cached_layout(sim)
    
    # Prepare node sizes
    node_sizes = [100 + max(0, sim.agents[node].energy) * 300 for node in sim.graph.nodes()]
    
    if color_by == "energy":
        # Color by energy level using a colormap
        energies = [sim.agents[node].energy for node in sim.graph.nodes()]
        
        # Draw edges first (underneath nodes)
        nx.draw_networkx_edges(
            sim.graph, pos, 
            edge_color='#cccccc', 
            alpha=0.5, 
            width=0.5,
            ax=ax
        )
        
        # Draw nodes with energy colormap
        nodes = nx.draw_networkx_nodes(
            sim.graph, pos,
            node_color=energies,
            node_size=node_sizes,
            cmap=plt.cm.RdYlGn,  # Red (low) -> Yellow -> Green (high)
            vmin=-1.0,
            vmax=1.0,
            alpha=0.9,
            ax=ax
        )
        
        # Add colorbar
        cbar = plt.colorbar(nodes, ax=ax, shrink=0.8, label='Energy Level')
        cbar.ax.tick_params(labelsize=8)
        
        # Labels: show state letters when colored by energy
        state_letters = {
            AgentState.PASSIVE: "P",
            AgentState.MOBILIZED: "M",
            AgentState.ORGANIZER: "O",
        }
        labels = {node: state_letters[sim.agents[node].state] for node in sim.graph.nodes()}
        
    else:
        # Color by state (default)
        state_colors = {
            AgentState.PASSIVE: "#3498db",      # Blue
            AgentState.MOBILIZED: "#f39c12",    # Orange
            AgentState.ORGANIZER: "#e74c3c",    # Red
        }
        node_colors = [state_colors[sim.agents[node].state] for node in sim.graph.nodes()]
        
        # Draw edges first (underneath nodes)
        nx.draw_networkx_edges(
            sim.graph, pos, 
            edge_color='#cccccc', 
            alpha=0.5, 
            width=0.5,
            ax=ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            sim.graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )
        
        # Add legend for states
        legend_elements = [
            mpatches.Patch(color='#e74c3c', label='Organizer'),
            mpatches.Patch(color='#f39c12', label='Mobilized'),
            mpatches.Patch(color='#3498db', label='Passive'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        # Labels: show energy level when colored by state
        labels = {node: f"{sim.agents[node].energy:.2f}" for node in sim.graph.nodes()}
    
    # Draw node labels
    nx.draw_networkx_labels(
        sim.graph, pos,
        labels=labels,
        font_size=6,
        font_color='black',
        ax=ax
    )
    
    # Count states for title
    n_org = len([a for a in sim.agents.values() if a.state == AgentState.ORGANIZER])
    n_mob = len([a for a in sim.agents.values() if a.state == AgentState.MOBILIZED])
    n_pass = len([a for a in sim.agents.values() if a.state == AgentState.PASSIVE])
    
    color_label = "by Energy" if color_by == "energy" else "by State"
    ax.set_title(f"Tick {sim.tick} | Org: {n_org} | Mob: {n_mob} | Pass: {n_pass} (Colored {color_label})", fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_network_from_snapshot(sim: UnionSim, snapshot: dict, color_by: str = "state") -> plt.Figure:
    """
    Create network visualization from a historical snapshot using matplotlib.
    
    Args:
        sim: The simulation (for graph structure)
        snapshot: Dictionary with agent states at a specific tick
        color_by: "state" to color by agent state, "energy" to color by energy level
        
    Returns matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Get layout positions using Fruchterman-Reingold (consistent across snapshots)
    pos = get_cached_layout(sim)
    
    state_names = {0: "Passive", 1: "Mobilized", 2: "Organizer"}
    
    # Build temporary graph from snapshot edges
    temp_graph = nx.Graph()
    temp_graph.add_nodes_from(sim.graph.nodes())
    temp_graph.add_edges_from(snapshot['edges'])
    
    # Prepare node sizes from snapshot
    node_sizes = []
    energies = []
    for node in temp_graph.nodes():
        agent_data = snapshot['agents'].get(str(node), snapshot['agents'].get(node))
        if agent_data:
            node_sizes.append(100 + max(0, agent_data['energy']) * 300)
            energies.append(agent_data['energy'])
        else:
            node_sizes.append(100)
            energies.append(0)
    
    # Draw edges first
    nx.draw_networkx_edges(
        temp_graph, pos, 
        edge_color='#cccccc', 
        alpha=0.5, 
        width=0.5,
        ax=ax
    )
    
    if color_by == "energy":
        # Draw nodes with energy colormap
        nodes = nx.draw_networkx_nodes(
            temp_graph, pos,
            node_color=energies,
            node_size=node_sizes,
            cmap=plt.cm.RdYlGn,  # Red (low) -> Yellow -> Green (high)
            vmin=-0.5,
            vmax=1.0,
            alpha=0.9,
            ax=ax
        )
        
        # Add colorbar
        cbar = plt.colorbar(nodes, ax=ax, shrink=0.8, label='Energy Level')
        cbar.ax.tick_params(labelsize=8)
        
        # Labels: show state letters when colored by energy
        state_letters = {0: "P", 1: "M", 2: "O"}  # Passive, Mobilized, Organizer
        labels = {}
        for node in temp_graph.nodes():
            agent_data = snapshot['agents'].get(str(node), snapshot['agents'].get(node))
            if agent_data:
                labels[node] = state_letters[agent_data['state']]
            else:
                labels[node] = "?"
    else:
        # Color by state (default)
        state_colors = {
            0: "#3498db",      # PASSIVE - Blue
            1: "#f39c12",      # MOBILIZED - Orange
            2: "#e74c3c",      # ORGANIZER - Red
        }
        
        node_colors = []
        for node in temp_graph.nodes():
            agent_data = snapshot['agents'].get(str(node), snapshot['agents'].get(node))
            if agent_data:
                node_colors.append(state_colors[agent_data['state']])
            else:
                node_colors.append("#cccccc")
        
        # Draw nodes
        nx.draw_networkx_nodes(
            temp_graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#e74c3c', label='Organizer'),
            mpatches.Patch(color='#f39c12', label='Mobilized'),
            mpatches.Patch(color='#3498db', label='Passive'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        # Labels: show energy level when colored by state
        labels = {}
        for node in temp_graph.nodes():
            agent_data = snapshot['agents'].get(str(node), snapshot['agents'].get(node))
            if agent_data:
                labels[node] = f"{agent_data['energy']:.2f}"
            else:
                labels[node] = "?"
    
    # Draw node labels
    nx.draw_networkx_labels(
        temp_graph, pos,
        labels=labels,
        font_size=6,
        font_color='black',
        ax=ax
    )
    
    # Count states for title
    state_counts = {0: 0, 1: 0, 2: 0}
    for agent_data in snapshot['agents'].values():
        state_counts[agent_data['state']] += 1
    
    color_label = "by Energy" if color_by == "energy" else "by State"
    ax.set_title(f"Tick {snapshot['tick']} | Org: {state_counts[2]} | Mob: {state_counts[1]} | Pass: {state_counts[0]} (Colored {color_label})", fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def save_network_as_svg(sim: UnionSim, snapshot: dict, filepath: str) -> None:
    """
    Save network visualization as SVG file.
    
    Args:
        sim: The simulation (for graph structure)
        snapshot: Dictionary with agent states at a specific tick
        filepath: Path to save the SVG file
    """
    fig = create_network_from_snapshot(sim, snapshot)
    fig.savefig(filepath, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_plotly_network_from_snapshot(sim: UnionSim, snapshot: dict) -> go.Figure:
    """
    Create a Plotly network visualization from a historical snapshot.
    
    Uses consistent layout positions across all snapshots for animation.
    """
    # Get layout positions (consistent seed for stable layout)
    pos = nx.spring_layout(sim.graph, seed=42, k=2)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in snapshot['edges']:
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    state_colors = {
        0: '#3498db',  # PASSIVE
        1: '#f39c12',  # MOBILIZED
        2: '#e74c3c',  # ORGANIZER
    }
    state_names = {0: "Passive", 1: "Mobilized", 2: "Organizer"}
    
    for node in sim.graph.nodes():
        if node in pos:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            agent_data = snapshot['agents'].get(str(node), snapshot['agents'].get(node))
            if agent_data:
                node_colors.append(state_colors[agent_data['state']])
                node_sizes.append(10 + max(0, agent_data['energy']) * 20)
                node_text.append(
                    f"Agent {node}<br>"
                    f"Dept: {agent_data['department_id']}<br>"
                    f"Energy: {agent_data['energy']:.2f}<br>"
                    f"State: {state_names[agent_data['state']]}"
                )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color='white')
        )
    )
    
    # Count states for title
    state_counts = {0: 0, 1: 0, 2: 0}
    for agent_data in snapshot['agents'].values():
        state_counts[agent_data['state']] += 1
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Tick {snapshot['tick']} | 🔴 Org: {state_counts[2]} | 🟠 Mob: {state_counts[1]} | 🔵 Pass: {state_counts[0]}",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=40, b=0),
        )
    )
    
    return fig


def create_plotly_network(sim: UnionSim) -> go.Figure:
    """
    Create a Plotly network visualization.
    
    Alternative to Pyvis for static or export purposes.
    """
    # Get layout positions
    pos = nx.spring_layout(sim.graph, seed=42, k=2)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in sim.graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    state_colors = {
        AgentState.PASSIVE: '#3498db',
        AgentState.MOBILIZED: '#f39c12',
        AgentState.ORGANIZER: '#e74c3c',
    }
    
    for node in sim.graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        agent = sim.agents[node]
        node_colors.append(state_colors[agent.state])
        node_sizes.append(10 + max(0, agent.energy) * 20)
        node_text.append(
            f"Agent {node}<br>"
            f"Dept: {agent.department_id}<br>"
            f"Energy: {agent.energy:.2f}<br>"
            f"State: {agent.state.name}"
        )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color='white')
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Network Visualization',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=40, b=0),
        )
    )
    
    return fig


def create_metrics_charts(metrics_history: list) -> dict:
    """Create Plotly charts for all metrics."""
    if not metrics_history:
        return {}
    
    df = pd.DataFrame(metrics_history)
    charts = {}
    
    # Total Energy over time
    charts['energy'] = px.line(
        df, x='tick', y=['total_energy', 'mean_energy'],
        title='Energy Dynamics',
        labels={'value': 'Energy', 'tick': 'Tick'},
    )
    charts['energy'].update_layout(legend_title_text='Metric')
    
    # Energy Gradient (change per tick)
    if 'energy_gradient' in df.columns:
        charts['energy_gradient'] = px.bar(
            df, x='tick', y='energy_gradient',
            title='Energy Gradient (ΔE per tick)',
            labels={'energy_gradient': 'Energy Change', 'tick': 'Tick'},
            color='energy_gradient',
            color_continuous_scale=['red', 'white', 'green'],
            color_continuous_midpoint=0,
        )
        charts['energy_gradient'].update_layout(
            yaxis_title='ΔE (Energy Added/Subtracted)',
            xaxis_title='Tick'
        )
    
    # Energy Components Breakdown
    if 'energy_from_outreach' in df.columns and 'energy_from_contagion' in df.columns:
        charts['energy_components'] = px.bar(
            df, x='tick', 
            y=['energy_from_outreach', 'energy_from_contagion', 'energy_from_decay'],
            title='Energy Components (Contribution to ΔE)',
            labels={'value': 'Energy Contribution', 'tick': 'Tick', 'variable': 'Component'},
            color_discrete_map={
                'energy_from_outreach': '#3498db',  # Blue for outreach
                'energy_from_contagion': '#2ecc71',  # Green for contagion
                'energy_from_decay': '#e74c3c'  # Red for decay
            }
        )
        charts['energy_components'].update_layout(
            barmode='relative',
            yaxis_title='Energy Contribution',
            xaxis_title='Tick',
            legend_title_text='Component'
        )
    
    # State counts over time
    charts['states'] = px.area(
        df, x='tick', 
        y=['organizer_count', 'mobilized_count', 'passive_count'],
        title='Agent States Over Time',
        labels={'value': 'Count', 'tick': 'Tick'},
        color_discrete_sequence=['#e74c3c', '#f39c12', '#3498db']
    )
    charts['states'].update_layout(legend_title_text='State')
    
    # Fitness metrics
    fitness_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Reach', 'DSI', 'LWR', 'Network Metrics')
    )
    
    # Reach
    fitness_fig.add_trace(
        go.Scatter(x=df['tick'], y=df['reach'], mode='lines', name='Reach'),
        row=1, col=1
    )
    
    # DSI
    fitness_fig.add_trace(
        go.Scatter(x=df['tick'], y=df['dsi'], mode='lines', name='DSI'),
        row=1, col=2
    )
    
    # LWR (handle inf values)
    lwr_values = df['lwr'].replace([np.inf, -np.inf], np.nan)
    fitness_fig.add_trace(
        go.Scatter(x=df['tick'], y=lwr_values, mode='lines', name='LWR'),
        row=2, col=1
    )
    
    # Network metrics
    fitness_fig.add_trace(
        go.Scatter(x=df['tick'], y=df['clustering_coefficient'], 
                   mode='lines', name='Clustering'),
        row=2, col=2
    )
    fitness_fig.add_trace(
        go.Scatter(x=df['tick'], y=df['global_efficiency'], 
                   mode='lines', name='Efficiency'),
        row=2, col=2
    )
    
    fitness_fig.update_layout(height=500, title_text="Fitness Metrics")
    charts['fitness'] = fitness_fig
    
    return charts


def download_button(data, filename, button_text, mime_type):
    """Create a download button for data."""
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False)
    elif isinstance(data, dict):
        import json
        data = json.dumps(data, indent=2)
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{button_text}</a>'
    return href


def create_experiment_comparison_charts(df: pd.DataFrame, group_by: str) -> dict:
    """
    Create comparison charts for experiment results grouped by a parameter.
    
    Args:
        df: DataFrame with experiment results
        group_by: Parameter column to group by (e.g., 'param_alpha')
    
    Returns:
        Dictionary of Plotly figures
    """
    charts = {}
    
    if df.empty or group_by not in df.columns:
        return charts
    
    # Get unique values for grouping
    unique_values = sorted(df[group_by].unique())
    
    # Energy over time by group
    fig_energy = go.Figure()
    for val in unique_values:
        subset = df[df[group_by] == val]
        # Average across trials
        avg_data = subset.groupby('tick').agg({
            'total_energy': ['mean', 'std'],
            'mean_energy': ['mean', 'std']
        }).reset_index()
        avg_data.columns = ['tick', 'total_energy_mean', 'total_energy_std', 
                           'mean_energy_mean', 'mean_energy_std']
        
        fig_energy.add_trace(go.Scatter(
            x=avg_data['tick'],
            y=avg_data['total_energy_mean'],
            name=f'{group_by.replace("param_", "")}={val}',
            mode='lines',
        ))
        # Add confidence band
        fig_energy.add_trace(go.Scatter(
            x=list(avg_data['tick']) + list(avg_data['tick'][::-1]),
            y=list(avg_data['total_energy_mean'] + avg_data['total_energy_std']) + 
              list((avg_data['total_energy_mean'] - avg_data['total_energy_std'])[::-1]),
            fill='toself',
            fillcolor='rgba(0,100,200,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig_energy.update_layout(
        title=f'Total Energy by {group_by.replace("param_", "")}',
        xaxis_title='Tick',
        yaxis_title='Total Energy',
        legend_title=group_by.replace("param_", "")
    )
    charts['energy_comparison'] = fig_energy
    
    # Reach over time
    fig_reach = go.Figure()
    for val in unique_values:
        subset = df[df[group_by] == val]
        avg_data = subset.groupby('tick')['reach'].agg(['mean', 'std']).reset_index()
        
        fig_reach.add_trace(go.Scatter(
            x=avg_data['tick'],
            y=avg_data['mean'],
            name=f'{group_by.replace("param_", "")}={val}',
            mode='lines',
        ))
    
    fig_reach.update_layout(
        title=f'Reach by {group_by.replace("param_", "")}',
        xaxis_title='Tick',
        yaxis_title='Reach (fraction non-passive)',
        legend_title=group_by.replace("param_", "")
    )
    charts['reach_comparison'] = fig_reach
    
    # Organizer count over time
    fig_org = go.Figure()
    for val in unique_values:
        subset = df[df[group_by] == val]
        avg_data = subset.groupby('tick')['organizer_count'].agg(['mean', 'std']).reset_index()
        
        fig_org.add_trace(go.Scatter(
            x=avg_data['tick'],
            y=avg_data['mean'],
            name=f'{group_by.replace("param_", "")}={val}',
            mode='lines',
        ))
    
    fig_org.update_layout(
        title=f'Organizer Count by {group_by.replace("param_", "")}',
        xaxis_title='Tick',
        yaxis_title='Number of Organizers',
        legend_title=group_by.replace("param_", "")
    )
    charts['organizer_comparison'] = fig_org
    
    # Energy components if available
    if 'energy_from_outreach' in df.columns:
        fig_components = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Outreach', 'Contagion', 'Decay')
        )
        
        for i, component in enumerate(['energy_from_outreach', 'energy_from_contagion', 'energy_from_decay']):
            for val in unique_values:
                subset = df[df[group_by] == val]
                avg_data = subset.groupby('tick')[component].mean().reset_index()
                
                fig_components.add_trace(
                    go.Scatter(
                        x=avg_data['tick'],
                        y=avg_data[component],
                        name=f'{group_by.replace("param_", "")}={val}',
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=1, col=i+1
                )
        
        fig_components.update_layout(
            title=f'Energy Components by {group_by.replace("param_", "")}',
            height=400
        )
        charts['components_comparison'] = fig_components
    
    # Final state box plots
    final_ticks = df.groupby(['combo_id', 'trial_id', group_by]).last().reset_index()
    
    fig_box = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Final Reach', 'Final Organizers', 'Final Energy')
    )
    
    for col_idx, metric in enumerate(['reach', 'organizer_count', 'total_energy']):
        fig_box.add_trace(
            go.Box(
                x=final_ticks[group_by].astype(str),
                y=final_ticks[metric],
                name=metric
            ),
            row=1, col=col_idx+1
        )
    
    fig_box.update_layout(
        title=f'Final Metrics Distribution by {group_by.replace("param_", "")}',
        showlegend=False,
        height=400
    )
    charts['final_metrics_box'] = fig_box
    
    return charts


def load_experiment_from_folder(folder_path: str) -> Tuple[Optional[pd.DataFrame], Optional[ExperimentConfig], Optional[str]]:
    """
    Load experiment results and configuration from a folder.
    
    Args:
        folder_path: Path to the experiment folder
        
    Returns:
        Tuple of (results_df, config, yaml_path) or (None, None, None) if loading fails
    """
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        return None, None, None
    
    # Find CSV files
    csv_files = list(folder.glob("*_results.csv"))
    if not csv_files:
        return None, None, None
    
    # Use most recent CSV
    csv_file = sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    
    # Find YAML config files
    yaml_files = list(folder.glob("*_config.yaml")) + list(folder.glob("*_config.yml"))
    yaml_file = None
    if yaml_files:
        yaml_file = sorted(yaml_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    
    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        
        # Load config if available
        config = None
        if yaml_file:
            with open(yaml_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
                config = ExperimentConfig(**yaml_data)
        
        return df, config, str(yaml_file) if yaml_file else None
    except Exception as e:
        st.error(f"Error loading experiment: {e}")
        return None, None, None


def render_experiment_tab():
    """Render the experiment configuration and results tab."""
    st.header("🧪 Batch Experiments")
    
    # Sub-tabs for different experiment stages
    exp_tab0, exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
        "📂 Load", "📝 Configure", "🚀 Run", "📊 Results", "💾 Export"
    ])
    
    with exp_tab0:
        st.subheader("Load Existing Experiment")
        st.markdown("Load previously run experiments from the experiments folder.")
        
        # Get available experiment folders
        experiments_dir = Path("output/experiments")
        if experiments_dir.exists():
            # Find all directories
            exp_folders = [d for d in experiments_dir.iterdir() if d.is_dir()]
            
            if exp_folders:
                # Sort by modification time, most recent first
                exp_folders = sorted(exp_folders, key=lambda x: x.stat().st_mtime, reverse=True)
                
                st.markdown(f"**Found {len(exp_folders)} experiment(s):**")
                
                # Create selection
                folder_options = {f.name: str(f) for f in exp_folders}
                selected_folder_name = st.selectbox(
                    "Select experiment to load",
                    options=list(folder_options.keys()),
                    help="Choose an experiment folder to view results"
                )
                
                selected_folder_path = folder_options[selected_folder_name]
                
                # Show folder info
                folder_path_obj = Path(selected_folder_path)
                csv_files = list(folder_path_obj.glob("*_results.csv"))
                yaml_files = list(folder_path_obj.glob("*_config.yaml")) + list(folder_path_obj.glob("*_config.yml"))
                snapshot_dir = folder_path_obj / "snapshots"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CSV Files", len(csv_files))
                with col2:
                    st.metric("Config Files", len(yaml_files))
                with col3:
                    if snapshot_dir.exists():
                        svg_files = list(snapshot_dir.glob("*.svg"))
                        st.metric("Snapshots", len(svg_files))
                    else:
                        st.metric("Snapshots", 0)
                
                # Load button
                if st.button("📂 Load Experiment", type="primary"):
                    with st.spinner("Loading experiment..."):
                        df, config, yaml_path = load_experiment_from_folder(selected_folder_path)
                        
                        if df is not None:
                            st.session_state.experiment_results = df
                            st.session_state.experiment_config = config
                            st.session_state.loaded_experiment_path = selected_folder_path
                            
                            # Create a mock runner object for compatibility
                            if config:
                                st.session_state.experiment_runner = ExperimentRunner(config)
                                st.session_state.experiment_runner.results = df
                            
                            st.success(f"✅ Loaded experiment from `{selected_folder_name}`")
                            st.info(f"📊 Loaded {len(df)} rows of data")
                            
                            if config:
                                summary = config.get_summary()
                                st.info(f"🔬 {summary['n_parameter_combinations']} parameter combinations × {summary['n_trials_per_combination']} trials")
                            
                            st.balloons()
                        else:
                            st.error("Failed to load experiment. Make sure the folder contains valid results CSV.")
                
                # Show files in folder
                with st.expander("📁 View Folder Contents"):
                    all_files = list(folder_path_obj.rglob("*"))
                    files_only = [f for f in all_files if f.is_file()]
                    
                    for f in sorted(files_only):
                        rel_path = f.relative_to(folder_path_obj)
                        size_kb = f.stat().st_size / 1024
                        st.text(f"📄 {rel_path} ({size_kb:.1f} KB)")
            else:
                st.info("No experiment folders found yet. Run an experiment first!")
        else:
            st.info("No experiments directory found. Run an experiment to create it.")
    
    with exp_tab1:
        st.subheader("Experiment Configuration")
        
        # Add prominent parameter balance guidance
        with st.expander("💡 Recommended Parameter Presets for Alpha Comparison", expanded=True):
            st.markdown("""
            ### Balanced Configuration for Comparing Organizing vs Mobilizing
            
            When comparing **extreme strategies** (α=0 pure mobilizing vs α=1 pure organizing), use these balanced parameters:
            
            **Recommended Starting Values:**
            - **Outreach Energy**: `0.45` (1-to-1 organizing boost)
            - **Broadcast Energy**: `0.01` (balanced for typical network sizes)
            - **Broadcast Reach**: `0.3` (30% of all agents reached)
            - **Delta (Decay)**: `0.02` (balanced: ~0.4× outreach energy)
            - **Beta (Contagion)**: `0.3` (moderate peer influence)
            - **Omega (Organizer Threshold)**: `0.8` (meaningful threshold)
            - **Persistence Threshold**: `3` ticks (balanced transition speed)
            - **P_in**: `0.2` (intra-department connectivity)
            - **P_out**: `0.001` (inter-department connectivity)
            - **Contagion Power**: `2.0` (complex contagion exponent)
            - **Organizer Potential**: `0.3` (30% can become organizers)
            - **Mobilizing Threshold**: `0.6` (energy threshold for mobilization)
            
            **Why These Values?**
            - **Broadcast Energy**: Ensures α=0 and α=1 strategies receive similar initial energy per tick (formula: outreach / (reach × total_agents))
            - **Delta**: Counters ~40% of organizing energy, creating realistic attrition without overwhelming outreach
            - **Omega at 0.7**: High enough that mobilized agents can't accumulate energy infinitely, low enough that organizers are achievable
            
            **Formula Reminder:**
            ```
            broadcast_energy = outreach_energy / (broadcast_reach × total_agents)
            delta = 0.2 to 1.0 × outreach_energy  (for typical α values)
            ```
            
            Copy these values to the Fixed Parameters section below!
            """)
        
        # Experiment metadata
        col1, col2 = st.columns(2)
        with col1:
            exp_name = st.text_input("Experiment Name", value="my_experiment", 
                                     help="Name for this experiment")
        with col2:
            exp_description = st.text_area("Description", height=68,
                                          help="Describe the purpose of this experiment")
        
        st.markdown("---")
        
        # Parameter ranges
        st.subheader("🔄 Parameter Ranges (to sweep)")
        st.markdown("Select parameters to vary and specify values to test.")
        
        # Available parameters for sweeping
        sweep_params = {
            'alpha': {'min': 0.0, 'max': 1.0, 'default': [0.0, 0.5, 1.0], 'step': 0.1},
            'beta': {'min': 0.0, 'max': 1.0, 'default': [0.1, 0.3, 0.5], 'step': 0.1},
            'delta': {'min': 0.001, 'max': 0.1, 'default': [0.01, 0.02, 0.05], 'step': 0.001},
            'contagion_power': {'min': 0.5, 'max': 5.0, 'default': [1.0, 2.0, 3.0], 'step': 0.5},
            'p_in': {'min': 0.1, 'max': 0.9, 'default': [0.3, 0.4, 0.5], 'step': 0.05},
            'p_out': {'min': 0.01, 'max': 0.2, 'default': [0.05, 0.1], 'step': 0.01},
            'omega': {'min': 0.5, 'max': 1.0, 'default': [0.7, 0.8, 0.9], 'step': 0.05},
            'persistence_threshold': {'min': 1, 'max': 10, 'default': [2, 3, 5], 'step': 1},
        }
        
        # Multi-select for which parameters to sweep
        selected_sweep_params = st.multiselect(
            "Select parameters to sweep",
            options=list(sweep_params.keys()),
            default=['alpha'],
            help="Choose which parameters to vary across simulations"
        )
        
        # Input fields for each selected parameter's values
        parameter_ranges = {}
        for param in selected_sweep_params:
            config = sweep_params[param]
            st.markdown(f"**{param}**")
            values_str = st.text_input(
                f"Values for {param} (comma-separated)",
                value=", ".join(map(str, config['default'])),
                key=f"sweep_{param}"
            )
            try:
                if param == 'persistence_threshold':
                    values = [int(v.strip()) for v in values_str.split(',')]
                else:
                    values = [float(v.strip()) for v in values_str.split(',')]
                parameter_ranges[param] = values
            except ValueError:
                st.error(f"Invalid values for {param}")
        
        st.markdown("---")
        
        # Fixed parameters
        st.subheader("📌 Fixed Parameters")
        
        # Parameter Balance Warning Box
        st.info("**⚠️ Parameter Balance is Critical**")
        st.markdown("""
        For valid comparisons between organizing (α=1) and mobilizing (α=0) strategies, parameters must be balanced:
        
        **Strategy Energy Balance:**
        - Organizing energy per tick: `outreach_energy × n_organizers`
        - Mobilizing energy per tick: `broadcast_energy × n_organizers × broadcast_reach × total_agents`
        - These should be within 3:1 ratio
        
        **Recommended Formula:** `broadcast_energy ≈ outreach_energy / (broadcast_reach × total_agents)`
        
        """)
        
        # Calculate suggested broadcast_energy
        with st.expander("🔧 Auto-Calculate Balanced Broadcast Energy"):
            calc_col1, calc_col2, calc_col3 = st.columns(3)
            with calc_col1:
                temp_outreach = st.number_input("Your 1-1 Energy", value=0.45, step=0.05, key="calc_outreach")
            with calc_col2:
                temp_reach = st.number_input("Your Broadcast Reach", value=0.3, step=0.05, key="calc_reach")
            with calc_col3:
                temp_total_agents = st.number_input("Total Agents", value=100, min_value=10, key="calc_agents")
            
            suggested_broadcast = temp_outreach / (temp_reach * temp_total_agents)
            st.success(f"**Suggested Balanced Broadcast Energy: {suggested_broadcast:.4f}**")
            st.caption(f"This ensures organizing (α=1) and mobilizing (α=0) have similar total energy input per tick.")
        
        st.markdown("---")
        
        fixed_col1, fixed_col2, fixed_col3 = st.columns(3)
        
        with fixed_col1:
            n_departments = st.number_input("Departments", min_value=2, max_value=10, value=5)
            agents_per_dept = st.number_input("Agents/Dept", min_value=5, max_value=50, value=20)
            seed_size = st.number_input("Seed Size", min_value=1, max_value=10, value=3)
        
        with fixed_col2:
            outreach_energy = st.number_input("Outreach Energy", min_value=0.0, max_value=1.0, value=0.45, step=0.05,
                                             help="Energy boost for 1-to-1 organizing conversations")
            broadcast_energy = st.number_input("Broadcast Energy", min_value=0.0, max_value=0.2, value=0.01, step=0.001,
                                              help="Energy per agent from broadcasts. Use calculator above for balanced value!")
            broadcast_reach = st.number_input("Broadcast Reach", min_value=0.01, max_value=1.0, value=0.3, step=0.05,
                                             help="Fraction of neighbors reached by broadcast")
        
        with fixed_col3:
            mobilizing_threshold = st.number_input("Mobilizing Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
            organizer_potential = st.number_input("Organizer Potential", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        
        # Show balance check
        if outreach_energy > 0 and broadcast_energy > 0:
            # Calculate total network size
            total_agents = n_departments * agents_per_dept
            
            organizing_energy = outreach_energy * seed_size
            mobilizing_energy = broadcast_energy * seed_size * broadcast_reach * total_agents
            ratio = organizing_energy / mobilizing_energy if mobilizing_energy > 0 else float('inf')
            
            balance_col1, balance_col2, balance_col3 = st.columns(3)
            with balance_col1:
                st.metric("Organizing Energy/Tick", f"{organizing_energy:.3f}")
            with balance_col2:
                st.metric("Mobilizing Energy/Tick", f"{mobilizing_energy:.3f}")
            with balance_col3:
                if 0.33 < ratio < 3.0:
                    st.metric("Energy Ratio", f"{ratio:.2f}x", delta="✓ Balanced", delta_color="normal")
                else:
                    st.metric("Energy Ratio", f"{ratio:.2f}x", delta="⚠️ Imbalanced", delta_color="inverse")
            
            if ratio < 0.33 or ratio > 3.0:
                st.warning(f"⚠️ Energy imbalance detected! Ratio is {ratio:.2f}:1 (should be between 0.33-3.0). "
                          f"Adjust broadcast_energy to ~{outreach_energy / (broadcast_reach * total_agents):.4f} for balance.")
        
        st.markdown("---")
        st.subheader("📌 Additional Parameters (Not Swept)")
        st.caption("Set values for parameters not included in the sweep above.")
        
        add_col1, add_col2, add_col3, add_col4 = st.columns(4)
        
        with add_col1:
            st.markdown("**Energy Dynamics**")
            beta_fixed = st.number_input("Beta (Contagion)", min_value=0.0, max_value=3.0, value=0.3, step=0.05,
                                        help="Complex contagion strength (peer influence)",
                                        key="beta_fixed") if 'beta' not in selected_sweep_params else 0.3
            contagion_power_fixed = st.number_input("Contagion Power (p)", min_value=0.0, max_value=9.0, value=2.0, step=0.1,
                                                   help="Exponent for complex contagion β(Ē)^p",
                                                   key="contagion_power_fixed") if 'contagion_power' not in selected_sweep_params else 2.0
            delta_fixed = st.number_input("Delta (Decay)", min_value=0.001, max_value=0.4, value=0.02, step=0.001,
                                         help="Energy decay rate. Recommended: 0.2-1.0× outreach energy",
                                         key="delta_fixed") if 'delta' not in selected_sweep_params else 0.02
        
        with add_col2:
            st.markdown("**Thresholds**")
            mobilizing_threshold_fixed = st.number_input("Mobilizing Threshold", min_value=-0.5, max_value=1.0, value=0.6, step=0.05,
                                                        help="Energy threshold: Passive → Mobilized",
                                                        key="mobilizing_threshold_fixed") if 'mobilizing_threshold' not in selected_sweep_params else mobilizing_threshold
            omega_fixed = st.number_input("Omega (Organizer Threshold)", min_value=0.5, max_value=1.0, value=0.7, step=0.05,
                                         help="Energy threshold to become organizer (also max for non-organizers)",
                                         key="omega_fixed") if 'omega' not in selected_sweep_params else 0.7
            persistence_fixed = st.number_input("Persistence Threshold", min_value=1, max_value=20, value=3,
                                               help="Consecutive ticks needed for state transitions",
                                               key="persistence_fixed") if 'persistence_threshold' not in selected_sweep_params else 3
        
        with add_col3:
            st.markdown("**Network Structure**")
            p_in_fixed = st.number_input("P_in (Intra-dept)", min_value=0.1, max_value=0.9, value=0.2, step=0.05,
                                        help="Connection probability within departments",
                                        key="p_in_fixed") if 'p_in' not in selected_sweep_params else 0.2
            p_out_fixed = st.number_input("P_out (Inter-dept)", min_value=0.0001, max_value=0.2, value=0.001, step=0.001, format="%.4f",
                                         help="Connection probability between departments",
                                         key="p_out_fixed") if 'p_out' not in selected_sweep_params else 0.001
            n_departments_fixed = n_departments  # Already set above
            agents_per_dept_fixed = agents_per_dept  # Already set above
        
        with add_col4:
            st.markdown("**Initial Conditions**")
            seed_size_fixed = seed_size  # Already set above
            organizer_potential_fixed = organizer_potential  # Already set above
            st.markdown("**Strategy**")
            alpha_fixed = st.number_input("Alpha (Strategy)", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                         help="α=1: Pure organizing, α=0: Pure mobilizing",
                                         key="alpha_fixed") if 'alpha' not in selected_sweep_params else 0.5
        
        # Check decay balance
        if 'delta' not in selected_sweep_params and 'alpha' in selected_sweep_params:
            # Check for typical alpha values
            alpha_values = parameter_ranges.get('alpha', [0.5])
            avg_alpha = sum(alpha_values) / len(alpha_values)
            decay_target_min = 0.1 * outreach_energy * avg_alpha
            decay_target_max = 1.5 * outreach_energy * avg_alpha
            
            if delta_fixed < decay_target_min or delta_fixed > decay_target_max:
                st.warning(f"⚠️ **Decay Balance Warning**: For your alpha range, delta should be between "
                          f"{decay_target_min:.4f} and {decay_target_max:.4f} (currently {delta_fixed:.4f}). "
                          f"Too low = unrealistic energy accumulation. Too high = energy can't sustain.")
        
        # Build fixed parameters dict (excluding those being swept)
        fixed_parameters = {
            'n_departments': n_departments,
            'agents_per_dept': agents_per_dept,
            'seed_size': seed_size,
            'outreach_energy': outreach_energy,
            'broadcast_energy': broadcast_energy,
            'broadcast_reach_ratio': broadcast_reach,
        }
        
        # Add non-swept parameters with proper conditional values
        if 'alpha' not in selected_sweep_params:
            fixed_parameters['alpha'] = alpha_fixed
        if 'beta' not in selected_sweep_params:
            fixed_parameters['beta'] = beta_fixed
        if 'delta' not in selected_sweep_params:
            fixed_parameters['delta'] = delta_fixed
        if 'omega' not in selected_sweep_params:
            fixed_parameters['omega'] = omega_fixed
        if 'persistence_threshold' not in selected_sweep_params:
            fixed_parameters['persistence_threshold'] = persistence_fixed
        if 'p_in' not in selected_sweep_params:
            fixed_parameters['p_in'] = p_in_fixed
        if 'p_out' not in selected_sweep_params:
            fixed_parameters['p_out'] = p_out_fixed
        if 'contagion_power' not in selected_sweep_params:
            fixed_parameters['contagion_power'] = contagion_power_fixed
        if 'mobilizing_threshold' not in selected_sweep_params:
            fixed_parameters['mobilizing_threshold'] = mobilizing_threshold_fixed
        if 'organizer_potential_ratio' not in selected_sweep_params:
            fixed_parameters['organizer_potential_ratio'] = organizer_potential
        
        st.markdown("---")
        
        # Execution settings
        st.subheader("⚙️ Execution Settings")
        exec_col1, exec_col2, exec_col3 = st.columns(3)
        
        with exec_col1:
            n_trials = st.number_input("Trials per combination", min_value=1, max_value=50, value=5,
                                       help="Number of runs with different seeds for each parameter combination")
        with exec_col2:
            n_ticks = st.number_input("Ticks per simulation", min_value=10, max_value=500, value=100)
        with exec_col3:
            base_seed = st.number_input("Base seed", min_value=0, max_value=99999, value=42)
        
        # Create config and show summary
        if parameter_ranges:
            config = ExperimentConfig(
                name=exp_name,
                description=exp_description,
                parameter_ranges=parameter_ranges,
                fixed_parameters=fixed_parameters,
                n_trials=n_trials,
                n_ticks=n_ticks,
                base_seed=base_seed
            )
            
            st.session_state.experiment_config = config
            
            # Show summary
            st.markdown("---")
            st.subheader("📊 Experiment Summary")
            
            summary = config.get_summary()
            
            sum_col1, sum_col2, sum_col3 = st.columns(3)
            with sum_col1:
                st.metric("Parameter Combinations", summary['n_parameter_combinations'])
            with sum_col2:
                st.metric("Total Simulations", summary['total_simulations'])
            with sum_col3:
                estimated_time = summary['total_simulations'] * n_ticks * 0.01  # rough estimate
                st.metric("Est. Time (seq)", f"{estimated_time:.0f}s")
            
            # Show parameter grid preview
            with st.expander("Preview Parameter Grid"):
                grid = config.get_parameter_grid()
                grid_preview = pd.DataFrame(grid[:10])  # Show first 10
                st.dataframe(grid_preview)
                if len(grid) > 10:
                    st.caption(f"... and {len(grid) - 10} more combinations")
        else:
            st.warning("Please select at least one parameter to sweep.")
    
    with exp_tab2:
        st.subheader("Run Experiment")
        
        if st.session_state.experiment_config is None:
            st.warning("Please configure an experiment first in the Configure tab.")
        else:
            config = st.session_state.experiment_config
            summary = config.get_summary()
            
            st.info(f"Ready to run **{summary['total_simulations']}** simulations "
                   f"({summary['n_parameter_combinations']} combinations × {summary['n_trials_per_combination']} trials)")
            
            # Run options
            run_col1, run_col2 = st.columns(2)
            with run_col1:
                use_parallel = st.checkbox("Use parallel execution", value=False,
                                          help="Run simulations in parallel (faster but may not show progress)")
            with run_col2:
                if use_parallel:
                    import multiprocessing as mp
                    n_workers = st.slider("Number of workers", 1, mp.cpu_count(), max(1, mp.cpu_count() - 1))
            
            if st.button("🚀 Run Experiment", type="primary"):
                st.session_state.experiment_running = True
                
                runner = ExperimentRunner(config)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(completed, total):
                    progress_bar.progress(completed / total)
                    status_text.text(f"Running simulation {completed}/{total}...")
                
                with st.spinner("Running experiment..."):
                    start_time = datetime.now()
                    
                    if use_parallel:
                        # Note: Parallel may not update progress smoothly
                        df = runner.run(n_workers=n_workers, progress_callback=update_progress)
                    else:
                        df = runner.run_sequential(progress_callback=update_progress)
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                
                st.session_state.experiment_results = df
                st.session_state.experiment_runner = runner
                st.session_state.experiment_running = False
                
                # Auto-save results and config to experiment folder
                csv_path = runner.save_results_csv()
                yaml_path = runner.save_config_yaml()
                
                status_text.text("")
                progress_bar.progress(1.0)
                
                st.success(f"✅ Experiment completed in {duration:.1f} seconds!")
                st.info(f"📁 Results saved to: `{Path(csv_path).parent}`")
                st.balloons()
    
    with exp_tab3:
        st.subheader("Experiment Results")
        
        if st.session_state.experiment_results is None:
            st.info("No experiment results yet. Run an experiment first.")
        else:
            df = st.session_state.experiment_results
            config = st.session_state.experiment_config
            
            # Get parameter columns for grouping
            param_cols = [c for c in df.columns if c.startswith('param_')]
            
            if param_cols:
                # Group by selector
                group_by = st.selectbox(
                    "Group results by parameter",
                    options=param_cols,
                    index=param_cols.index('param_alpha') if 'param_alpha' in param_cols else 0,
                    format_func=lambda x: x.replace('param_', '')
                )
                # Generate comparison charts
                charts = create_experiment_comparison_charts(df, group_by)
                
                if charts:
                    # Display charts
                    st.plotly_chart(charts.get('energy_comparison'), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'reach_comparison' in charts:
                            st.plotly_chart(charts['reach_comparison'], use_container_width=True)
                    with col2:
                        if 'organizer_comparison' in charts:
                            st.plotly_chart(charts['organizer_comparison'], use_container_width=True)
                    
                    if 'components_comparison' in charts:
                        st.plotly_chart(charts['components_comparison'], use_container_width=True)
                    
                    if 'final_metrics_box' in charts:
                        st.plotly_chart(charts['final_metrics_box'], use_container_width=True)
                
                # Multi-parameter heatmap if multiple params
                if len(param_cols) >= 2:
                    st.markdown("---")
                    st.subheader("Parameter Interaction Heatmap")
                    
                    hm_col1, hm_col2, hm_col3 = st.columns(3)
                    with hm_col1:
                        hm_x = st.selectbox("X-axis parameter", param_cols, key="hm_x")
                    with hm_col2:
                        hm_y = st.selectbox("Y-axis parameter", 
                                           [p for p in param_cols if p != hm_x], 
                                           key="hm_y")
                    with hm_col3:
                        hm_metric = st.selectbox("Metric", 
                                                ['reach', 'organizer_count', 'total_energy'],
                                                key="hm_metric")
                    
                    # Get final tick values
                    final_df = df.groupby(['combo_id', 'trial_id'] + param_cols).last().reset_index()
                    
                    # Aggregate by the two parameters
                    pivot_data = final_df.groupby([hm_x, hm_y])[hm_metric].mean().unstack()
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=pivot_data.values,
                        x=[str(c) for c in pivot_data.columns],
                        y=[str(i) for i in pivot_data.index],
                        colorscale='Viridis',
                        colorbar_title=hm_metric
                    ))
                    fig_heatmap.update_layout(
                        title=f'{hm_metric} by {hm_x.replace("param_", "")} and {hm_y.replace("param_", "")}',
                        xaxis_title=hm_y.replace("param_", ""),
                        yaxis_title=hm_x.replace("param_", "")
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Raw data view
            with st.expander("View Raw Data"):
                st.dataframe(df.head(100))
                st.caption(f"Showing first 100 of {len(df)} rows")
    
    with exp_tab4:
        st.subheader("Export Results")
        
        if st.session_state.experiment_results is None:
            st.info("No experiment results to export. Run an experiment first.")
        else:
            df = st.session_state.experiment_results
            config = st.session_state.experiment_config
            
            # Show experiment folder structure
            exp_folder = Path(config.output_dir) / config.name
            st.info(f"📁 **Experiment Folder**: `{exp_folder}`")
            
            if exp_folder.exists():
                st.markdown("**Current files:**")
                files = list(exp_folder.rglob("*"))
                files = [f for f in files if f.is_file()]
                for f in sorted(files)[:10]:  # Show first 10
                    st.markdown(f"- `{f.relative_to(exp_folder)}`")
                if len(files) > 10:
                    st.caption(f"... and {len(files) - 10} more files")
            
            st.markdown("---")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.markdown("### 📄 CSV Export")
                st.markdown("Export all results with parameters, seeds, and metrics by tick.")
                st.caption("Note: CSV and YAML are automatically saved when experiment completes.")
                
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv_data,
                    file_name=f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.csv",
                    mime="text/csv"
                )
                
                # Summary CSV
                st.markdown("---")
                st.markdown("**Summary Statistics (Final Tick)**")
                if hasattr(st.session_state, 'experiment_runner'):
                    summary_df = st.session_state.experiment_runner.get_summary_dataframe()
                    if not summary_df.empty:
                        summary_csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Summary CSV",
                            data=summary_csv,
                            file_name=f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_summary.csv",
                            mime="text/csv"
                        )
            
            with export_col2:
                st.markdown("### 📋 YAML Config Export")
                st.markdown("Save experiment configuration for CLI reproduction.")
                
                from dataclasses import asdict
                yaml_data = yaml.dump(asdict(config), default_flow_style=False, sort_keys=False)
                st.download_button(
                    label="📥 Download Config YAML",
                    data=yaml_data,
                    file_name=f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_config.yaml",
                    mime="text/yaml"
                )
                
                st.markdown("---")
                st.markdown("**Load Config**")
                uploaded_yaml = st.file_uploader("Upload YAML config", type=['yaml', 'yml'])
                if uploaded_yaml is not None:
                    try:
                        yaml_content = yaml.safe_load(uploaded_yaml)
                        st.session_state.experiment_config = ExperimentConfig(**yaml_content)
                        st.success("Configuration loaded! Go to Configure tab to review.")
                    except Exception as e:
                        st.error(f"Error loading config: {e}")
            
            # SVG Snapshot Export
            st.markdown("---")
            st.markdown("### 🖼️ Network Snapshot Export (SVG)")
            st.markdown("Generate SVG snapshots of network evolution for a specific parameter combination.")
            
            if config is not None:
                param_grid = config.get_parameter_grid()
                
                svg_col1, svg_col2 = st.columns(2)
                
                with svg_col1:
                    combo_idx = st.selectbox(
                        "Parameter combination",
                        range(len(param_grid)),
                        format_func=lambda i: str({k: v for k, v in param_grid[i].items() 
                                                   if k in config.parameter_ranges})
                    )
                    trial_idx = st.number_input("Trial index", min_value=0, 
                                               max_value=config.n_trials - 1, value=0)
                
                with svg_col2:
                    steps_str = st.text_input("Steps to capture (comma-separated)", 
                                             value="0, 25, 50, 75, 100")
                    try:
                        steps = [int(s.strip()) for s in steps_str.split(',')]
                    except ValueError:
                        steps = [0, 25, 50, 75, 100]
                        st.warning("Invalid steps, using defaults")
                
                if st.button("🖼️ Generate SVG Snapshots"):
                    params = param_grid[combo_idx]
                    seed = config.base_seed + combo_idx * 1000 + trial_idx
                    
                    with st.spinner("Generating SVG snapshots..."):
                        import zipfile
                        from io import BytesIO
                        
                        # Create temp directory for SVGs
                        import tempfile
                        with tempfile.TemporaryDirectory() as tmpdir:
                            svg_files = save_network_snapshots_svg(
                                params=params,
                                seed=seed,
                                steps=steps,
                                output_dir=tmpdir,
                                n_ticks=config.n_ticks
                            )
                            
                            # Create zip file
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                for svg_path in svg_files:
                                    zf.write(svg_path, Path(svg_path).name)
                            
                            zip_buffer.seek(0)
                            
                            st.download_button(
                                label=f"📥 Download SVG Archive ({len(svg_files)} files)",
                                data=zip_buffer.getvalue(),
                                file_name=f"network_snapshots_combo{combo_idx}_trial{trial_idx}.zip",
                                mime="application/zip"
                            )
                    
                    st.success(f"Generated {len(svg_files)} SVG files!")


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("✊ Union ABM Simulator")
    st.markdown("**Agent-Based Model of Labor Organizing Dynamics**")
    
    # Add prominent info box about parameter balance
    with st.expander("ℹ️ About This Model & Parameter Balance", expanded=False):
        st.markdown("""
        ### Model Features
        
        This simulation implements union organizing dynamics based on the following key features:
        
        - **Symmetric State Transitions**: Agents can become organizers AND be demoted back to mobilized with persistence gates
        - **Energy Gradient Tracking**: Monitor net energy flow decomposed into outreach, contagion, and decay components
        - **Sociologically Realistic Decay**: Decay depends only on engaged neighbors (mobilized + organizers), not passive agents
        - **Broadcast Scaling**: Broadcast energy scales with number of organizers, creating realistic capacity constraints
        - **Energy Clipping**: Non-organizers cannot exceed the organizer threshold (ω)
        
        ### ⚠️ Critical: Parameter Balance
        
        **For scientifically valid comparisons between organizing (α=1) and mobilizing (α=0) strategies, 
        total energy input must be balanced.**
        
        **Strategy Energy Balance**:
        - Organizing: `outreach_energy × n_organizers` per tick
        - Mobilizing: `broadcast_energy × n_organizers × avg_neighbors × broadcast_reach` per tick
        
        These should be within 3:1 ratio. Formula:
        ```
        broadcast_energy ≈ outreach_energy / (avg_neighbors × broadcast_reach)
        ```
        
        **Decay Balance**:
        Decay rate should counter 10-150% of outreach energy:
        ```
        0.1 × (outreach_energy × α) < delta < 1.5 × (outreach_energy × α)
        ```
        
        **The model automatically warns you** if parameters are imbalanced when you initialize a simulation.
        
        ### Why This Matters
        
        Without balanced parameters, one strategy may appear more effective simply because it has more resources,
        not because it's inherently superior. This would invalidate comparisons and lead to incorrect conclusions
        about organizing vs mobilizing effectiveness.
        """)
    
    # Main mode selector
    mode = st.radio(
        "Mode",
        ["🎮 Single Simulation", "🧪 Batch Experiments"],
        horizontal=True,
        help="Choose between running a single simulation or batch experiments with parameter sweeps"
    )
    
    st.markdown("---")
    
    if mode == "🧪 Batch Experiments":
        render_experiment_tab()
        # Footer for experiment mode
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: #888;'>"
            "Union ABM v1.0 | Built with Streamlit"
            "</p>",
            unsafe_allow_html=True
        )
        return  # Exit main() early for experiment mode
    
    # === Single Simulation Mode ===
    # Get parameters from sidebar
    params = create_sidebar()
    
    # Main control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Initialize Simulation", type="primary"):
            st.session_state.sim = UnionSim(**params)
            st.session_state.metrics_history = []
            st.session_state.logger.clear()
            st.session_state.network_layout = None  # Clear cached layout for new simulation
            
            # Validate parameter balance
            validation = st.session_state.sim.validate_parameter_balance()
            if not validation['balanced']:
                st.warning("⚠️ **Parameter Balance Warning**")
                for warning in validation['warnings']:
                    st.write(warning)
                if validation['recommendations']:
                    st.info("**Recommendations:**")
                    for rec in validation['recommendations']:
                        st.write(rec)
            else:
                st.success("Simulation initialized with balanced parameters!")
    
    with col2:
        n_steps = st.number_input("Steps to run", min_value=1, max_value=500, value=50)
    
    with col3:
        if st.button("▶️ Run Simulation"):
            if st.session_state.sim is not None:
                with st.spinner(f"Running {n_steps} steps..."):
                    new_metrics = st.session_state.sim.run(n_steps)
                    st.session_state.metrics_history.extend(new_metrics)
                    st.session_state.logger.log_simulation_run(new_metrics)
                st.success(f"Completed {n_steps} steps!")
            else:
                st.error("Please initialize the simulation first.")
    
    with col4:
        stop_test = st.checkbox("🛑 Activate Stop Test")
        if stop_test and st.session_state.sim is not None:
            st.session_state.sim.activate_stop_test()
            st.warning("Stop Test Active - Organizers disabled")
        elif st.session_state.sim is not None:
            st.session_state.sim.deactivate_stop_test()
    
    # Display current state
    if st.session_state.sim is not None:
        sim = st.session_state.sim
        
        # Current metrics
        st.header("📊 Current State")
        metrics = sim.get_metrics()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Tick", metrics['tick'])
        with col2:
            st.metric("Total Energy", f"{metrics['total_energy']:.2f}")
        with col3:
            st.metric("Organizers", metrics['organizer_count'])
        with col4:
            st.metric("Mobilized", metrics['mobilized_count'])
        with col5:
            st.metric("Reach", f"{metrics['reach']:.1%}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("DSI", f"{metrics['dsi']:.3f}")
        with col2:
            lwr_display = f"{metrics['lwr']:.3f}" if metrics['lwr'] != float('inf') else "∞"
            st.metric("LWR", lwr_display)
        with col3:
            st.metric("Clustering", f"{metrics['clustering_coefficient']:.3f}")
        with col4:
            st.metric("Efficiency", f"{metrics['global_efficiency']:.3f}")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🕸️ Network", "🎬 Evolution Playback", "📈 Charts", "📋 Data", "💾 Export"
        ])
        
        with tab1:
            st.subheader("Network Visualization (Current State)")
            
            # Color mode selector
            color_mode = st.radio(
                "Color nodes by:",
                ["State", "Energy"],
                horizontal=True,
                key="network_color_mode",
                help="State: Blue=Passive, Orange=Mobilized, Red=Organizer | Energy: Red=Low, Green=High"
            )
            color_by = "energy" if color_mode == "Energy" else "state"
            
            # Legend based on color mode
            if color_by == "state":
                st.markdown("""
                **Legend:** 🔴 Organizer | 🟠 Mobilized | 🔵 Passive | Node size indicates energy level.
                """)
            else:
                st.markdown("""
                **Legend:** 🔴 Low Energy (-0.5) → 🟡 Medium → 🟢 High Energy (1.0) | Node size indicates energy level.
                """)
            
            # Network visualization using matplotlib (fast Fruchterman-Reingold layout)
            fig = create_network_visualization(sim, color_by=color_by)
            st.pyplot(fig)
            plt.close(fig)
        
        with tab2:
            st.subheader("🎬 Network Evolution Playback")
            
            snapshots = sim.get_all_snapshots()
            n_snapshots = len(snapshots)
            
            if n_snapshots > 0:
                # Color mode selector for playback
                playback_color_mode = st.radio(
                    "Color nodes by:",
                    ["State", "Energy"],
                    horizontal=True,
                    key="playback_color_mode",
                    help="State: Blue=Passive, Orange=Mobilized, Red=Organizer | Energy: Red=Low, Green=High"
                )
                playback_color_by = "energy" if playback_color_mode == "Energy" else "state"
                
                if playback_color_by == "state":
                    st.markdown("""
                    **Explore the network evolution step by step or watch it animate!**
                    
                    🔴 Organizer | 🟠 Mobilized | 🔵 Passive
                    """)
                else:
                    st.markdown("""
                    **Explore the network evolution step by step or watch it animate!**
                    
                    🔴 Low Energy (-0.5) → 🟡 Medium → 🟢 High Energy (1.0)
                    """)
                
                # Playback controls
                st.markdown("### ⏯️ Playback Controls")
                
                ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 2])
                
                with ctrl_col1:
                    if st.button("⏮️ Start"):
                        st.session_state.playback_tick = 0
                
                with ctrl_col2:
                    if st.button("⬅️ Previous"):
                        st.session_state.playback_tick = max(0, st.session_state.playback_tick - 1)
                
                with ctrl_col3:
                    if st.button("➡️ Next"):
                        st.session_state.playback_tick = min(n_snapshots - 1, st.session_state.playback_tick + 1)
                
                with ctrl_col4:
                    if st.button("⏭️ End"):
                        st.session_state.playback_tick = n_snapshots - 1
                
                # Tick slider (only show if more than 1 snapshot)
                if n_snapshots > 1:
                    selected_tick = st.slider(
                        "Select Tick",
                        min_value=0,
                        max_value=n_snapshots - 1,
                        value=min(st.session_state.playback_tick, n_snapshots - 1),
                        key="tick_slider",
                        help="Drag to explore network state at any tick"
                    )
                    st.session_state.playback_tick = selected_tick
                else:
                    selected_tick = 0
                    st.session_state.playback_tick = 0
                    st.info("Run more simulation steps to enable tick selection.")
                
                # Animation section
                st.markdown("### 🎥 Animation")
                
                anim_col1, anim_col2, anim_col3 = st.columns([1, 1, 2])
                
                with anim_col1:
                    playback_speed = st.selectbox(
                        "Speed",
                        options=[0.1, 0.25, 0.5, 1.0, 2.0],
                        index=2,
                        format_func=lambda x: f"{x}x"
                    )
                
                with anim_col2:
                    auto_play = st.checkbox("▶️ Auto-play", value=False)
                
                # Auto-play logic
                if auto_play:
                    import time
                    placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    for i in range(st.session_state.playback_tick, n_snapshots):
                        snapshot = snapshots[i]
                        fig = create_network_from_snapshot(sim, snapshot, color_by=playback_color_by)
                        placeholder.pyplot(fig)
                        plt.close(fig)
                        progress_bar.progress((i + 1) / n_snapshots)
                        time.sleep(1.0 / playback_speed)
                        
                        # Check if we should stop
                        if i == n_snapshots - 1:
                            st.session_state.playback_tick = 0
                            break
                    
                    st.success("✅ Playback complete!")
                else:
                    # Static view of selected tick
                    snapshot = snapshots[selected_tick]
                    
                    # Show snapshot metrics
                    state_counts = {0: 0, 1: 0, 2: 0}
                    total_energy = 0
                    for agent_data in snapshot['agents'].values():
                        state_counts[agent_data['state']] += 1
                        total_energy += agent_data['energy']
                    
                    metric_cols = st.columns(5)
                    with metric_cols[0]:
                        st.metric("Tick", snapshot['tick'])
                    with metric_cols[1]:
                        st.metric("Total Energy", f"{total_energy:.2f}")
                    with metric_cols[2]:
                        st.metric("Organizers", state_counts[2])
                    with metric_cols[3]:
                        st.metric("Mobilized", state_counts[1])
                    with metric_cols[4]:
                        st.metric("Passive", state_counts[0])
                    
                    # Visualization using matplotlib (fast Fruchterman-Reingold)
                    fig = create_network_from_snapshot(sim, snapshot, color_by=playback_color_by)
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Run the simulation to see network evolution.")
        
        with tab3:
            st.subheader("Metrics Over Time")
            
            if st.session_state.metrics_history:
                charts = create_metrics_charts(st.session_state.metrics_history)
                
                # Energy chart
                st.plotly_chart(charts['energy'], use_container_width=True)
                
                # Energy gradient chart
                if 'energy_gradient' in charts:
                    st.plotly_chart(charts['energy_gradient'], use_container_width=True)
                
                # Energy components breakdown chart
                if 'energy_components' in charts:
                    st.plotly_chart(charts['energy_components'], use_container_width=True)
                
                # States chart
                st.plotly_chart(charts['states'], use_container_width=True)
                
                # Fitness metrics
                st.plotly_chart(charts['fitness'], use_container_width=True)
            else:
                st.info("Run the simulation to see metrics over time.")
        
        with tab4:
            st.subheader("Simulation Data")
            
            # Summary statistics
            if st.session_state.metrics_history:
                summary = generate_summary_statistics(st.session_state.metrics_history)
                
                st.write("**Summary Statistics:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"- Total ticks: {summary.get('n_ticks', 0)}")
                    st.write(f"- Final organizers: {summary.get('final_organizer_count', 0)}")
                    st.write(f"- Final reach: {summary.get('final_reach', 0):.1%}")
                with col2:
                    st.write(f"- Max energy: {summary.get('max_total_energy', 0):.2f}")
                    st.write(f"- Final DSI: {summary.get('final_dsi', 0):.3f}")
                    st.write(f"- Final LWR: {summary.get('final_lwr', 0):.3f}")
                
                # Department analysis
                st.write("**Department Analysis:**")
                dept_analysis = MetricsCalculator.calculate_department_analysis(sim)
                dept_df = pd.DataFrame(dept_analysis).T
                dept_df.index.name = 'Department'
                st.dataframe(dept_df.style.format({
                    'mean_energy': '{:.3f}',
                    'organizer_pct': '{:.1%}',
                    'mobilized_pct': '{:.1%}',
                    'passive_pct': '{:.1%}',
                }))
                
                # Raw metrics table
                st.write("**Metrics History:**")
                metrics_df = pd.DataFrame(st.session_state.metrics_history)
                st.dataframe(metrics_df.tail(20))
        
        with tab5:
            st.subheader("Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**CSV Export**")
                if st.session_state.metrics_history:
                    metrics_df = pd.DataFrame(st.session_state.metrics_history)
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Metrics CSV",
                        data=csv,
                        file_name=f"union_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.write("**JSON State Export**")
                state_dict = sim.get_state_dict()
                import json
                json_str = json.dumps(state_dict, indent=2)
                st.download_button(
                    label="📥 Download State JSON",
                    data=json_str,
                    file_name=f"union_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col3:
                st.write("**Chart Export**")
                if st.session_state.metrics_history:
                    charts = create_metrics_charts(st.session_state.metrics_history)
                    
                    # Export energy chart as PNG (requires kaleido)
                    if 'energy' in charts:
                        try:
                            img_bytes = charts['energy'].to_image(format="png", scale=2)
                            st.download_button(
                                label="📥 Download Energy Chart (PNG)",
                                data=img_bytes,
                                file_name="energy_chart.png",
                                mime="image/png"
                            )
                        except ValueError:
                            st.warning("PNG export requires kaleido. Install with: `pip install kaleido`")
                            # Fallback: offer HTML export
                            html_str = charts['energy'].to_html()
                            st.download_button(
                                label="📥 Download Energy Chart (HTML)",
                                data=html_str,
                                file_name="energy_chart.html",
                                mime="text/html"
                            )
            
            # SVG Network Evolution Export
            st.write("---")
            st.write("**📊 Network Evolution SVG Export**")
            st.markdown("Export network snapshots as SVG files at regular intervals.")
            
            snapshots = sim.get_all_snapshots()
            n_snapshots = len(snapshots)
            
            if n_snapshots > 1:
                svg_col1, svg_col2 = st.columns(2)
                
                with svg_col1:
                    svg_interval = st.number_input(
                        "Export every X ticks",
                        min_value=1,
                        max_value=max(1, n_snapshots),
                        value=min(10, n_snapshots),
                        help="Save an SVG image every X simulation ticks"
                    )
                
                with svg_col2:
                    if st.button("🖼️ Generate SVG Archive"):
                        import zipfile
                        from io import BytesIO
                        
                        # Create a zip file in memory
                        zip_buffer = BytesIO()
                        
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            exported_count = 0
                            for i, snapshot in enumerate(snapshots):
                                if i % svg_interval == 0:
                                    status_text.text(f"Generating SVG for tick {snapshot['tick']}...")
                                    
                                    # Create SVG in memory
                                    fig = create_network_from_snapshot(sim, snapshot)
                                    svg_buffer = BytesIO()
                                    fig.savefig(svg_buffer, format='svg', bbox_inches='tight', facecolor='white')
                                    plt.close(fig)
                                    
                                    svg_buffer.seek(0)
                                    zip_file.writestr(
                                        f"network_tick_{snapshot['tick']:04d}.svg",
                                        svg_buffer.getvalue()
                                    )
                                    exported_count += 1
                                
                                progress_bar.progress((i + 1) / n_snapshots)
                            
                            status_text.text(f"✅ Generated {exported_count} SVG files!")
                        
                        zip_buffer.seek(0)
                        
                        st.download_button(
                            label=f"📥 Download SVG Archive ({exported_count} files)",
                            data=zip_buffer.getvalue(),
                            file_name=f"network_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
            else:
                st.info("Run the simulation for more ticks to export network evolution SVGs.")
            
            # Load state
            st.write("---")
            st.write("**Load Saved State**")
            uploaded_file = st.file_uploader("Upload JSON state file", type=['json'])
            if uploaded_file is not None:
                try:
                    import json
                    state_dict = json.load(uploaded_file)
                    st.session_state.sim = UnionSim.from_state_dict(state_dict)
                    st.session_state.metrics_history = []
                    st.success("State loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading state: {e}")
    
    else:
        # No simulation initialized
        st.info("👆 Configure parameters in the sidebar and click 'Initialize Simulation' to begin.")
        
        # Show example/demo section
        st.header("📚 About This Model")
        
        st.markdown("""
        This Agent-Based Model simulates **labor organizing dynamics** based on 
        labor organizing theory.
        
        ### Key Concepts:
        
        - **Organizing (α → 1)**: Deep 1-to-1 relationship building that creates 
          committed organizers who can recruit others
        - **Mobilizing (α → 0)**: Broadcast-style communication that activates 
          existing supporters but builds less structural power
        
        ### Agent States:
        - **Passive** 🔵: Not yet engaged
        - **Mobilized** 🟠: Engaged but not committed to organizing others
        - **Organizer** 🔴: Actively recruiting and building the network
        
        ### Key Metrics:
        - **DSI (Department Spread Index)**: How evenly organizers are distributed
        - **LWR (Leader-Worker Ratio)**: Ratio of organizers to mobilized workers
        - **Reach**: Fraction of non-passive agents
        - **Structural Resilience**: How well gains persist after organizing stops
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>"
        "Union ABM v1.0 | Built with Streamlit"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
