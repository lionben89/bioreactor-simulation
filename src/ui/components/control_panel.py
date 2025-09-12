"""
Control Panel Module

This module provides the main control interface for the bioreactor simulation,
including simulation controls, timeline navigation, and data export functionality.

Key Features:
- Initial simulation setup
- Fork creation controls
- Timeline navigation
- Export functionality
- Reset capabilities
"""

import streamlit as st
import copy
from .session_state import reset_simulation_state, get_max_simulation_time, can_continue_simulation
from .data_export import render_export_button


def render_control_panel(base_parameters):
    """
    Render the main control panel with all simulation controls.
    
    Args:
        base_parameters (dict): Current simulation parameters from sidebar
    
    Returns:
        bool: True if a simulation should be started, False otherwise
    """
    st.markdown("### Simulation Controls")
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
    
    start_simulation = False
    
    # Column 1: Initial simulation setup
    with col1:
        # Initial simulation button
        if st.button("🚀 Run Initial Simulation", type="primary", disabled=st.session_state.is_running_simulation):
            st.session_state.is_running_simulation = True
            st.session_state.base_parameters = copy.deepcopy(base_parameters)
            start_simulation = True
    
    # Column 2: Fork creation
    with col2:
        # Determine if user can continue simulation from current point
        can_continue = can_continue_simulation(base_parameters)
        
        # Continue simulation button (creates fork with modified parameters)
        if st.button("▶️ Continue from Current Point", disabled=not can_continue or st.session_state.is_running_simulation):
            if can_continue:
                st.session_state.is_running_simulation = True
                start_simulation = True
    
    # Column 3: Timeline navigation
    with col3:
        render_timeline_slider()
    
    # Column 4: Data export
    with col4:
        render_export_button()
    
    # Column 5: Reset functionality
    with col5:
        render_reset_button()
    
    return start_simulation


def render_timeline_slider():
    """
    Render the timeline navigation slider.
    
    This slider allows users to scrub through the simulation timeline
    and jump to specific time points for analysis or fork creation.
    """
    # Timeline scrubber - only show if we have simulation data
    if st.session_state.simulation_segments:
        # Find maximum time across all segments
        max_time = get_max_simulation_time()
        
        # Timeline slider for navigation
        st.session_state.current_time_point = st.slider(
            "Timeline Position (hours)", 
            0.0, 
            max_time, 
            st.session_state.current_time_point,
            step=st.session_state.base_parameters['time_step'],
            key="timeline_slider"
        )


def render_reset_button():
    """
    Render the reset all functionality.
    
    This button clears all simulation data and returns the interface
    to its initial state for starting a new analysis session.
    """
    # Reset all simulations and return to initial state
    if st.button("🔄 Reset All", type="secondary"):
        reset_simulation_state()
        st.rerun()
