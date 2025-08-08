"""
CHO Cell Perfusion Bioreactor Simulation - Interactive Analysis UI

This is the main entry point for the bioreactor simulation interface.
The application is now modularized into separate components for better
maintainability and code organization.

Key Features:
- Interactive parameter modification
- Timeline navigation and fork creation
- Real-time visualization with multiple plots
- CSV export with complete simulation data
- Fork comparison with parameter change tracking

Architecture:
- streamlit_app.py: Main application entry point
- components/: UI components (controls, visualization, etc.)
- utils/: Utility functions and helpers
"""

import streamlit as st

# Import modular components
from components import (
    initialize_session_state,
    render_control_panel,
    execute_simulation,
    render_visualization_layout,
    render_parameter_tabs
)

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(page_title="Bioreactor Simulation", layout="wide")
st.title("CHO Cell Perfusion Bioreactor Simulation - Interactive Analysis")

# ==================== SESSION STATE INITIALIZATION ====================
# Initialize all session state variables
initialize_session_state()

# ==================== SIDEBAR: PARAMETER CONFIGURATION ====================
# Render parameter tabs in sidebar
with st.sidebar:
    base_parameters = render_parameter_tabs()

# ==================== MAIN INTERFACE ====================

# Render control panel and check if simulation should start
start_simulation = render_control_panel(base_parameters)

# Execute simulation if requested
if start_simulation or st.session_state.is_running_simulation:
    success = execute_simulation(base_parameters)
    if success:
        st.session_state.is_running_simulation = False
        st.rerun()

# Render visualization layout
render_visualization_layout()