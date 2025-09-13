"""
Session State Management Module

This module handles initialization and management of all Streamlit session state
variables used throughout the bioreactor simulation interface.

Key Features:
- Centralized session state initialization
- State validation and defaults
- Helper functions for state management
"""

import streamlit as st


def initialize_session_state():
    """
    Initialize all session state variables with default values.
    
    This function ensures all required session state variables exist
    with appropriate default values for the simulation interface.
    """
    # Core simulation data storage
    if 'simulation_segments' not in st.session_state:
        st.session_state.simulation_segments = []  # List of all simulation segments (original + forks)
    
    # Timeline and navigation state
    if 'current_time_point' not in st.session_state:
        st.session_state.current_time_point = 0.0  # Current position on timeline slider
    
    # Parameter management
    if 'parameters_at_timepoint' not in st.session_state:
        st.session_state.parameters_at_timepoint = None  # Parameters at selected time point
    
    if 'base_parameters' not in st.session_state:
        st.session_state.base_parameters = None  # Base parameters for current session
    
    # Simulation control state
    if 'is_running_simulation' not in st.session_state:
        st.session_state.is_running_simulation = False  # Flag to prevent concurrent simulations
    
    # DoE session state
    if 'doe_design_matrix' not in st.session_state:
        st.session_state.doe_design_matrix = None  # Generated DoE design matrix
    
    if 'doe_design_properties' not in st.session_state:
        st.session_state.doe_design_properties = None  # DoE design properties and metadata
    
    if 'doe_batch_results' not in st.session_state:
        st.session_state.doe_batch_results = None  # Batch simulation results from DoE experiments
    
    if 'doe_selected_params' not in st.session_state:
        st.session_state.doe_selected_params = {}  # DoE parameter selections


def reset_simulation_state():
    """
    Reset all simulation-related session state to initial values.
    
    This function clears all simulation data and returns the interface
    to its initial state, ready for a new simulation session.
    """
    st.session_state.simulation_segments = []
    st.session_state.current_time_point = 0.0
    st.session_state.parameters_at_timepoint = None
    st.session_state.base_parameters = None
    st.session_state.is_running_simulation = False
    
    # Reset legend pagination
    if 'legend_page' in st.session_state:
        st.session_state.legend_page = 0
    if 'legend_prev_segment_count' in st.session_state:
        st.session_state.legend_prev_segment_count = 0
    
    # Reset DoE state when simulations are reset
    st.session_state.doe_design_matrix = None
    st.session_state.doe_design_properties = None
    st.session_state.doe_batch_results = None
    st.session_state.doe_selected_params = {}


def get_max_simulation_time():
    """
    Get the maximum time point across all simulation segments.
    
    Returns:
        float: Maximum time in hours across all segments, or 0.0 if no segments exist
    """
    if not st.session_state.simulation_segments:
        return 0.0
    
    return max([max([s['time'] for s in seg['states']]) for seg in st.session_state.simulation_segments])

def has_simulation_data():
    """
    Check if any simulation data exists in session state.
    
    Returns:
        bool: True if simulation segments exist, False otherwise
    """
    return len(st.session_state.simulation_segments) > 0


def can_continue_simulation(base_parameters):
    """
    Check if simulation can be continued from current time point.
    
    Args:
        base_parameters (dict): Current simulation parameters
        
    Returns:
        bool: True if simulation can continue, False otherwise
    """
    return (has_simulation_data() and 
            st.session_state.current_time_point < base_parameters.get('total_hours', 100))


def clear_doe_state():
    """Clear all DoE-related session state variables."""
    st.session_state.doe_design_matrix = None
    st.session_state.doe_design_properties = None
    st.session_state.doe_batch_results = None
    st.session_state.doe_selected_params = {}
