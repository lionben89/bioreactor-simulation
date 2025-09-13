"""
DoE Parameter Selection Component

This module provides functionality for selecting and configuring parameters
for Design of Experiments studies in the bioreactor simulation.
"""

import streamlit as st
from typing import Dict, Any

def get_simulation_parameters():
    """
    Get all available simulation parameters with their metadata.
    
    Returns:
        dict: Dictionary of parameter metadata including defaults, descriptions, and categories
    """
    return {
        
        # Process Control
        "glucose_threshold": {
            "default": 4.0,
            "description": "Glucose threshold for perfusion control (g/L)",
            "category": "Pump Control",
            "min_factor": 0.1,
            "max_factor": 10.0,
            "units": "g/L"
        },
        "perfusion_rate_high": {
            "default": 1.0, 
            "description": "High perfusion rate (vvd)",
            "category": "Pump Control", 
            "min_factor": 0.1,
            "max_factor": 5.0,
            "units": "vvd"
        },
        "glucose_feed_conc": {
            "default": 8.0,
            "description": "Glucose concentration in feed medium (g/L)",
            "category": "Feeding",
            "min_factor": 0.1,
            "max_factor": 9.0,
            "units": "g/L"
        },
        "glutamine_feed_conc": {
            "default": 5.0,
            "description": "Glutamine concentration in feed medium (g/L)",
            "category": "Feeding",
            "min_factor": 0.1,
            "max_factor": 9.0,
            "units": "g/L"
        },
        
        # Environmental Parameters
        "culture_ph": {
            "default": 6.5,
            "description": "Current culture pH",
            "category": "Environmental",
            "min_factor": 0.8,
            "max_factor": 1.3,
            "units": "pH"
        },
        "culture_temp": {
            "default": 37.0,
            "description": "Current culture temperature (°C)",
            "category": "Environmental",
            "min_factor": 0.8,
            "max_factor": 1.2,
            "units": "°C"
        },
        "dissolved_oxygen_percent": {
            "default": 100.0,
            "description": "Dissolved oxygen level (%)",
            "category": "Environmental",
            "min_factor": 0.5,
            "max_factor": 1.0,
            "units": "%"
        },
    }

def render_doe_parameter_selection() -> Dict[str, Any]:
    """
    Render the DoE parameter selection interface in the sidebar.
    
    Returns:
        dict: Dictionary of selected parameters with their ranges and metadata
    """
    st.sidebar.markdown("### 📋 Parameter Selection")
    st.sidebar.markdown("Select parameters to vary in your DoE study:")
    
    # Initialize DoE parameter selection in session state if not exists
    if 'doe_selected_params' not in st.session_state:
        st.session_state.doe_selected_params = {}
    
    # Get all available parameters
    all_params = get_simulation_parameters()
    
    # Group parameters by category
    categories = {}
    for param_name, param_info in all_params.items():
        category = param_info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(param_name)
    
    # Initialize selected parameters dictionary
    selected_params = {}
    
    # Render parameter selection by category
    for category, param_names in categories.items():
        with st.sidebar.expander(category, expanded=False):
            for param_name in param_names:
                param_info = all_params[param_name]
                
                # Check if parameter was previously selected
                previously_selected = param_name in st.session_state.doe_selected_params
                
                # Checkbox for parameter selection
                is_selected = st.checkbox(
                    f"{param_name}",
                    value=previously_selected,
                    key=f"doe_select_{param_name}",
                    help=f"{param_info['description']} (Default: {param_info['default']} {param_info['units']})"
                )
                
                if is_selected:
                    # Calculate default min/max based on factors
                    default_val = param_info["default"]
                    default_min = default_val * param_info["min_factor"]
                    default_max = default_val * param_info["max_factor"]
                    
                    # Get previous values if they exist
                    prev_min = default_min
                    prev_max = default_max
                    if param_name in st.session_state.doe_selected_params:
                        prev_min = st.session_state.doe_selected_params[param_name].get('min', default_min)
                        prev_max = st.session_state.doe_selected_params[param_name].get('max', default_max)
                    
                    # Create two columns for min/max inputs
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        min_val = st.number_input(
                            "Min",
                            value=prev_min,
                            key=f"doe_min_{param_name}",
                            format="%.6f",
                            help=f"Minimum value for {param_name}"
                        )
                    
                    with col2:
                        max_val = st.number_input(
                            "Max",
                            value=prev_max,
                            key=f"doe_max_{param_name}",
                            format="%.6f",
                            help=f"Maximum value for {param_name}"
                        )
                    
                    # Validate range
                    if min_val >= max_val:
                        st.error(f"Min must be < Max for {param_name}")
                    else:
                        # Store selected parameter in both local dict and session state
                        param_data = {
                            "min": min_val,
                            "max": max_val,
                            "default": default_val,
                            "description": param_info["description"],
                            "units": param_info["units"],
                            "category": param_info["category"]
                        }
                        selected_params[param_name] = param_data
                        st.session_state.doe_selected_params[param_name] = param_data
                else:
                    # Remove from session state if deselected
                    if param_name in st.session_state.doe_selected_params:
                        del st.session_state.doe_selected_params[param_name]
    
    # Update session state with current selection
    st.session_state.doe_selected_params = selected_params
    
    # Show selection summary
    if selected_params:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Selection Summary")
        st.sidebar.success(f"**{len(selected_params)} parameters selected**")
        
        # Show estimated study sizes for different DoE types
        n_factors = len(selected_params)
        
        st.sidebar.markdown("**Estimated Study Sizes:**")
        st.sidebar.write(f"• Full Factorial (2 levels): **{2**n_factors} runs**")
        st.sidebar.write(f"• Full Factorial (3 levels): **{3**n_factors} runs**")
        
        if n_factors >= 3:
            st.sidebar.write(f"• Central Composite Design: **{2**n_factors + 2*n_factors + 1} runs**")
        
        if n_factors >= 4:
            # Plackett-Burman needs multiples of 4
            pb_runs = ((n_factors // 4) + 1) * 4
            st.sidebar.write(f"• Plackett-Burman: **{pb_runs} runs**")
    
    else:
        st.sidebar.info("👆 Select parameters above to design your DoE study")
    
    return selected_params

def validate_parameter_selection(selected_params: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate the selected parameters for DoE study.
    
    Args:
        selected_params: Dictionary of selected parameters with ranges
        
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    if not selected_params:
        errors.append("No parameters selected. Please select at least one parameter.")
        return False, errors
    
    if len(selected_params) > 10:
        errors.append("Too many parameters selected. DoE studies with >10 factors become impractical.")
    
    # Check for range validity
    for param_name, param_data in selected_params.items():
        if param_data["min"] >= param_data["max"]:
            errors.append(f"Invalid range for {param_name}: min >= max")
        
        # Check for reasonable ranges
        range_ratio = param_data["max"] / param_data["min"]
        if range_ratio > 100:
            errors.append(f"Range for {param_name} is very wide (ratio: {range_ratio:.1f}). Consider narrowing the range.")
    
    is_valid = len(errors) == 0
    return is_valid, errors
