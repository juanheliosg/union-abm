"""
UI Components for Union ABM Streamlit Dashboard

This module contains reusable UI components and widgets
for the Streamlit application.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


def show_parameter_balance_info():
    """Display parameter balance information box."""
    with st.expander("ℹ️ About This Model & Parameter Balance", expanded=False):
        st.markdown("""
        ### Model Features
        
        This simulation implements Jane McAlevey's organizing theory with:
        
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


def show_balance_calculator():
    """Display auto-calculate balanced broadcast energy widget."""
    with st.expander("🔧 Auto-Calculate Balanced Broadcast Energy"):
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        with calc_col1:
            temp_outreach = st.number_input("Your Outreach Energy", value=0.5, step=0.05, key="calc_outreach")
        with calc_col2:
            temp_avg_neighbors = st.number_input("Estimated Avg Neighbors", value=15.0, step=1.0, key="calc_neighbors",
                                                 help="Typically 10-20 for SBM networks")
        with calc_col3:
            temp_reach = st.number_input("Your Broadcast Reach", value=0.75, step=0.05, key="calc_reach")
        
        suggested_broadcast = temp_outreach / (temp_avg_neighbors * temp_reach)
        st.success(f"**Suggested Balanced Broadcast Energy: {suggested_broadcast:.4f}**")
        st.caption(f"This ensures organizing (α=1) and mobilizing (α=0) have similar total energy input.")


def show_preset_recommendations():
    """Display recommended parameter presets."""
    with st.expander("💡 Recommended Parameter Presets for Alpha Comparison", expanded=True):
        st.markdown("""
        ### Balanced Configuration for Comparing Organizing vs Mobilizing
        
        When comparing **extreme strategies** (α=0 pure mobilizing vs α=1 pure organizing), use these balanced parameters:
        
        **Recommended Starting Values:**
        - **Outreach Energy**: `0.5` (1-to-1 organizing boost)
        - **Broadcast Energy**: `0.044` (calculated: 0.5 / (15 × 0.75) for balance)
        - **Broadcast Reach**: `0.75` (75% of neighbors reached)
        - **Delta (Decay)**: `0.02` (balanced: ~0.4× outreach energy)
        - **Beta (Contagion)**: `0.3` (moderate peer influence)
        - **Omega (Organizer Threshold)**: `0.7` (meaningful threshold)
        - **Persistence Threshold**: `3` ticks (balanced transition speed)
        
        **Why These Values?**
        - **Broadcast Energy**: Ensures α=0 and α=1 strategies receive similar total energy per tick
        - **Delta**: Counters ~40% of organizing energy, creating realistic attrition without overwhelming outreach
        - **Omega at 0.7**: High enough that mobilized agents can't accumulate energy infinitely, low enough that organizers are achievable
        
        **Formula Reminder:**
        ```
        broadcast_energy = outreach_energy / (avg_neighbors × broadcast_reach)
        delta = 0.2 to 1.0 × outreach_energy  (for typical α values)
        ```
        
        Copy these values to the Fixed Parameters section below!
        """)


def show_balance_check(outreach_energy: float, broadcast_energy: float, 
                       broadcast_reach: float, agents_per_dept: int, n_departments: int):
    """
    Display real-time balance checking metrics.
    
    Args:
        outreach_energy: 1-to-1 organizing energy boost
        broadcast_energy: Broadcast energy per agent
        broadcast_reach: Fraction of neighbors reached
        agents_per_dept: Number of agents per department
        n_departments: Number of departments
    """
    if outreach_energy > 0 and broadcast_energy > 0:
        # Estimate average neighbors for typical SBM (using default values)
        p_in_est = 0.4  # Will be overridden by user input in Additional Parameters section
        p_out_est = 0.05
        avg_neighbors_est = agents_per_dept * p_in_est + (n_departments - 1) * agents_per_dept * p_out_est
        
        organizing_energy = outreach_energy
        mobilizing_energy = broadcast_energy * avg_neighbors_est * broadcast_reach
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
                      f"Adjust broadcast_energy to ~{outreach_energy / (avg_neighbors_est * broadcast_reach):.4f} for balance.")


def show_decay_balance_warning(delta_fixed: float, outreach_energy: float, 
                               parameter_ranges: Dict, selected_sweep_params: List[str]):
    """Display decay balance warning if applicable."""
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


def show_experiment_folder_browser(experiments_dir: Path) -> tuple:
    """
    Display experiment folder browser and return selected folder info.
    
    Returns:
        Tuple of (selected_folder_path, folder_name) or (None, None) if no selection
    """
    if not experiments_dir.exists():
        st.info("No experiments directory found. Run an experiment to create it.")
        return None, None
    
    # Find all directories
    exp_folders = [d for d in experiments_dir.iterdir() if d.is_dir()]
    
    if not exp_folders:
        st.info("No experiment folders found yet. Run an experiment first!")
        return None, None
    
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
    
    return selected_folder_path, selected_folder_name


def show_folder_contents(folder_path: Path):
    """Display contents of a folder in an expander."""
    with st.expander("📁 View Folder Contents"):
        all_files = list(folder_path.rglob("*"))
        files_only = [f for f in all_files if f.is_file()]
        
        for f in sorted(files_only):
            rel_path = f.relative_to(folder_path)
            size_kb = f.stat().st_size / 1024
            st.text(f"📄 {rel_path} ({size_kb:.1f} KB)")
