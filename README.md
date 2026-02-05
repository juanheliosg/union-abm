# Union ABM: Agent-Based Model of Labor Organizing

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An Agent-Based Model (ABM) simulating labor organizing dynamics based on **Jane McAlevey's organizing theory**, distinguishing between *Organizing* (deep relationship building) and *Mobilizing* (broadcast activation) strategies. The model features energy dynamics with **organizer demotion mechanics**, **energy gradient tracking**, and **batch experiment capabilities** for parameter sweeps.

## 📚 Theoretical Background

### McAlevey's Framework

Jane McAlevey's work distinguishes between two approaches to building worker power that we have modelized as follows:

1. **Organizing**: Deep, 1-to-1 relationship building that develops new leaders from the base. Creates structural power that persists even when external support is withdrawn.

2. **Mobilizing**: Activating existing supporters through broadcasts and campaigns. More visible but creates less durable structural change.

### Model Implementation

This ABM operationalizes McAlevey's concepts through:

- **Network Structure**: A Stochastic Block Model (SBM) representing workplace departments with higher internal connectivity
- **Agent States**: Workers transition between Passive → Mobilized → Organizer (with symmetric demotion mechanics)
- **Energy Dynamics**: Complex contagion with social influence, strategic decay, and energy gradient tracking
- **Strategy Parameter (α)**: Controls the organizing-mobilizing mix with scaled broadcast energy
- **Persistence Gate**: Sustained energy thresholds required for both promotion AND demotion of organizers
- **Parameter Validation**: Built-in checks for strategy balance and decay calibration

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project
cd organizing-network-modeling

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Streamlit Dashboard

```bash
streamlit run app.py
```

This opens an interactive web interface with two modes:

#### 🎮 **Single Simulation Mode**
- Configure all simulation parameters via sidebar
- Real-time network visualization with Fruchterman-Reingold layout
- Live energy gradient charts showing ΔE components
- Monitor fitness metrics with interactive Plotly charts
- Evolution playback slider to review any historical tick
- **Automatic parameter balance warnings** when initializing
- Export data (CSV/JSON) and network snapshots (SVG)

#### 🧪 **Batch Experiments Mode**
Five-tab interface for comprehensive parameter studies:

1. **Load Tab**: Browse and load previously run experiments from organized folders
2. **Configure Tab**: Set up parameter sweeps with automatic experiment size calculation
3. **Run Tab**: Execute simulations (sequential or parallel) with real-time progress
4. **Results Tab**: Interactive comparison charts grouped by parameters, heatmaps for multi-parameter analysis
5. **Export Tab**: Download results CSV, config YAML, network snapshots as SVG/ZIP

All experiment artifacts auto-save to organized folders: `output/experiments/{experiment_name}/`

### Running from Command Line

```bash
# Basic run with parameter validation
python main.py --ticks 100 --alpha 0.8

# With specific parameters
python main.py --ticks 200 --alpha 1.0 --departments 5 --agents-per-dept 20

# Batch experiments from YAML config
python run_experiment.py my_experiment.yaml

# Generate sample YAML config
python run_experiment.py --generate-sample > my_experiment.yaml

# Run with network snapshots at specific ticks
python run_experiment.py my_experiment.yaml --snapshots --snapshot-steps "0,50,100"

# Use parallel execution (faster for large experiments)
python run_experiment.py my_experiment.yaml --workers 4
```

This runs 60 simulations (3 alphas × 2 betas × 10 trials) and saves results to `output/experiments/alpha_sweep_experiment/`.

## 📖 Model Documentation

### Agent States

| State | Description | Visualization |
|-------|-------------|---------------|
| **Passive** | Not engaged with organizing | 🔵 Blue |
| **Mobilized** | Engaged but not recruiting | 🟠 Orange |
| **Organizer** | Actively building the network | 🔴 Red |

### Energy Dynamics

Each agent has an energy value $E_i \in [-1.0, 1.0]$ that evolves according to:

$$E_i(t+1) = \text{clip}\left(E_i(t) + \Delta E_{\text{outreach}} + \beta (\bar{E}_{\text{engaged}})^p - \delta(1 - \bar{E}_{\text{engaged}}), E_{\text{min}}, E_{\text{max}}\right)$$

Where:
- $\Delta E_{\text{outreach}}$: Energy boost from being targeted by an organizer
- $\beta (\bar{E}_{\text{engaged}})^p$: Complex contagion (positive influence from high-energy neighbors)
- $\delta(1 - \bar{E}_{\text{engaged}})$: Decay (stronger when **engaged** neighbors have low energy)
- $\bar{E}_{\text{engaged}}$: Mean energy of mobilized/organizer neighbors (passive neighbors excluded)
- $E_{\text{max}}$: For organizers = 1.0, for mobilized/passive = $\omega$ (organizer threshold)

**Key Implementation Details**:

1. **Engaged-Only Decay**: Decay term uses $\bar{E}_{\text{engaged}}$ instead of all neighbors, preventing passive agents from artificially suppressing energy. This is more sociologically realistic—disengagement spreads from actively disengaging people, not from those who were never engaged.

2. **Energy Clipping by State**: 
   - Organizers can accumulate energy up to 1.0
   - Non-organizers are capped at ω (typically 0.7-0.8)
   - Prevents "super-mobilized" agents who never become organizers

3. **Energy Gradient**: Track total network energy change per tick:

$$\Delta E_{\text{total}} = \sum_i \left( \Delta E_{\text{outreach},i} + \beta (\bar{E}_{\text{engaged},i})^p - \delta(1 - \bar{E}_{\text{engaged},i}) \right)$$



### Persistence Gate (Symmetric Transitions)

**Becoming an Organizer**:
An agent becomes an **Organizer** only if ALL conditions are met for $X$ consecutive ticks:
1. Energy exceeds threshold: $E_i > \Omega$ (omega)
2. Being actively targeted by an existing Organizer
3. Maintains both conditions for `persistence_threshold` consecutive ticks

**Losing Organizer Status (Demotion)**:
Symmetrically, an organizer becomes **Mobilized** only if:
1. Energy falls below threshold: $E_i \leq \Omega$
2. Maintains low energy for `persistence_threshold` consecutive ticks
3. Counter resets if energy rises above ω

This models the reality that:
- Developing leaders requires sustained attention, not just momentary enthusiasm
- Organizing skills and structural position provide resilience against temporary setbacks
- True disengagement takes time—leaders don't collapse at the first difficulty
- Both transitions have inertia, creating hysteresis in the system

### Strategy Parameter (α)

The parameter $\alpha \in [0, 1]$ controls the organizing-mobilizing mix:

- **α = 1.0**: Pure Organizing
  - 1-to-1 conversations with targeted workers
  - Energy boost: `outreach_energy` to specific individuals
  - Builds clique connections among organizers
  - Total energy input: `outreach_energy × n_organizers` per tick
  
- **α = 0.0**: Pure Mobilizing
  - Broadcast communication to network neighbors
  - Energy boost: `broadcast_energy × n_organizers` distributed to reach% of neighbors
  - No structural changes to the network
  - Total energy input: `broadcast_energy × n_organizers × avg_neighbors × broadcast_reach` per tick

- **0 < α < 1**: Mixed strategy
  - Probability α of organizing action
  - Probability (1-α) of mobilizing action

### ⚠️ **CRITICAL: Parameter Balance**

For valid comparisons between strategies, **total energy input in initial state must be balanced**:

**Strategy Energy Balance**:
```
Organizing initial energy ≈ Mobilizing initial energy

outreach_energy × n_organizers ≈ broadcast_energy × n_organizers × avg_neighbors × broadcast_reach

Therefore: broadcast_energy ≈ outreach_energy / (avg_neighbors × broadcast_reach)
```

**Example**: If `outreach_energy = 0.5`, `avg_neighbors ≈ 15`, and `broadcast_reach = 0.75`:
```
broadcast_energy ≈ 0.5 / (15 × 0.75) ≈ 0.044
```

**Model provides automatic warnings** if:
- Strategy energy ratio exceeds 3:1 (one strategy has >3x energy input)
- Decay rate is outside 0.1-1.5× of outreach energy (too weak or overwhelming)

**Decay Balance**:
Decay should counter ~10-150% of outreach energy to create realistic dynamics:
```
0.1 × (outreach_energy × α) < delta < 1.5 × (outreach_energy × α)
```

If decay is too strong, no strategy can sustain energy. If too weak, energy accumulates unrealistically.

## ⚙️ Parameters Reference

### Network Structure

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `n_departments` | `--departments` | 5 | Number of departments (SBM blocks) |
| `agents_per_dept` | `--agents-per-dept` | 20 | Workers per department |
| `p_in` | `--p-in` | 0.4 | Intra-department connection probability |
| `p_out` | `--p-out` | 0.05 | Inter-department connection probability |
| `seed_size` | `--seed-size` | 3 | Initial organizer count |

### Strategy & Energy

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `alpha` | `--alpha` | 0.8 | Organizing vs mobilizing mix (0=pure mobilize, 1=pure organize) |
| `delta` | `--delta` | 0.01 | Energy decay rate (should balance with outreach: 0.1-1.5× outreach_energy) |
| `omega` | `--omega` | 0.7 | Organizer energy threshold (also max energy for non-organizers) |
| `persistence_threshold` | `--persistence` | 3 | Consecutive ticks required for state transitions (promotion AND demotion) |
| `outreach_energy` | `--outreach-energy` | 0.5 | 1-to-1 organizing energy boost |
| `broadcast_energy` | `--broadcast-energy` | 0.01 | Broadcast energy per agent (balance: ≈ outreach / (neighbors × reach)) |
| `broadcast_reach_ratio` | `--broadcast-reach` | 0.75 | Fraction of neighbors reached by broadcast |
| `contagion_power` | `--contagion-power` | 1.0 | Exponent p in contagion term β(Ē)^p |
| `beta` | `--beta` | 0.3 | Complex contagion strength, higher means more social reinforcement is needed (peer influence coefficient) |


**Parameter Tuning Guidelines**:

1. **Start with `alpha` sweep** to compare strategies: [0.0, 0.5, 1.0]
2. **Balance `broadcast_energy`** using formula: `outreach_energy / (avg_neighbors × broadcast_reach)`
3. **Tune `delta`** to 0.1-1.5× of effective outreach: `0.1 × outreach_energy × alpha < delta < 1.5 × outreach_energy × alpha`
4. **Set `omega`** high enough (0.7-0.8) to create meaningful threshold between mobilized and organizer states
5. **Adjust `persistence_threshold`** (2-5) to control transition speed—higher values = more stable states


## 📚 References

- McAlevey, J. (2016). *No Shortcuts: Organizing for Power in the New Gilded Age*. Oxford University Press.
- Centola, D. (2018). *How Behavior Spreads: The Science of Complex Contagions*. Princeton University Press.

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
