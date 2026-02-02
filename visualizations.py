"""
Visualization Functions for Union ABM

This module contains all visualization and plotting functions
for network diagrams, charts, and metrics displays.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as npt
import streamlit as st

from union_abm.engine import UnionSim, AgentState


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
    
    # Additional charts (reach, organizer count, components, box plots)
    # ... (keeping existing chart creation code)
    
    return charts
