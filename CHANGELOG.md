# Changelog - Union ABM Simulator

## 2026-01-17 - UI Updates

### Added Features

#### 1. Load Existing Experiments
- **New Tab**: Added "📂 Load" tab to the Batch Experiments interface
- **Functionality**: 
  - Browse and select from previously run experiments in `output/experiments/`
  - View experiment folder details (CSV files, config files, snapshots)
  - Load experiment results and configuration into the Results and Export tabs
  - Displays folder contents with file sizes
- **Location**: First tab in the Batch Experiments mode

#### 2. Streamlit Light Theme
- **Removed**: Custom CSS styling from `assets/styles.css`
- **Changed to**: Native Streamlit light theme via `.streamlit/config.toml`
- **Benefits**: 
  - Cleaner, more consistent UI
  - Better accessibility
  - Reduced maintenance overhead
  - Native Streamlit look and feel

### Technical Changes

#### app.py
- Removed `load_css()` function and all custom CSS loading
- Added `load_experiment_from_folder()` function to load experiment data from folders
- Added `import yaml` and `from typing import Optional, Tuple`
- Updated session state initialization with `loaded_experiment_path`
- Reorganized experiment tabs: Load → Configure → Run → Results → Export

#### .streamlit/config.toml
- Fixed CORS configuration (`enableCORS = true`)
- Theme already configured for light mode:
  - `base = "light"`
  - Primary color: `#1f77b4` (blue)
  - Background: `#ffffff` (white)
  - Secondary background: `#f0f2f6` (light gray)
  - Text color: `#262730` (dark gray)

### Usage

#### Loading Existing Experiments:
1. Go to "🧪 Batch Experiments" mode
2. Click on "📂 Load" tab
3. Select an experiment from the dropdown
4. Click "📂 Load Experiment" button
5. Navigate to "📊 Results" or "💾 Export" tabs to view/export data

#### Theme:
The app now uses Streamlit's native light theme. No custom styling is applied.
