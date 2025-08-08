"""
Helper Utilities Module

This module provides utility functions used throughout the simulation interface
for parameter comparison, data formatting, and other common operations.

Key Features:
- Parameter change detection and formatting
- Y-axis label generation
- Data validation utilities
"""


def get_parameter_changes(segment, original_params):
    """
    Compare segment parameters with original parameters to identify changes.
    
    Args:
        segment (dict): Simulation segment containing parameters
        original_params (dict): Original baseline parameters
        
    Returns:
        list: List of strings describing parameter changes in format "param: old→new"
    """
    changes = []
    for key, value in segment['parameters'].items():
        if key in original_params and original_params[key] != value:
            changes.append(f"{key}: {original_params[key]:.3f}→{value:.3f}")
    return changes


def get_y_axis_label(variable_name):
    """
    Generate appropriate Y-axis label based on variable type.
    
    Args:
        variable_name (str): Name of the measurement variable
        
    Returns:
        str: Formatted string for Y-axis label with appropriate units
    """
    if 'cell_density' in variable_name:
        return "Cell Density (cells/mL)"
    elif variable_name == 'pump_active':
        return "Pump State"
    elif 'product' in variable_name:
        return "Product Concentration (g/L)"
    else:
        return "Concentration (g/L)"


def format_variable_name(variable_name):
    """
    Format variable name for display purposes.
    
    Args:
        variable_name (str): Raw variable name (e.g., 'viable_cell_density')
        
    Returns:
        str: Formatted name for display (e.g., 'Viable Cell Density')
    """
    # Special formatting for product variables
    if variable_name == 'product_concentration':
        return "Product (In Reactor)"
    elif variable_name == 'aggregated_product':
        return "Product (Total Produced)"
    else:
        return variable_name.replace('_', ' ').title()


def validate_parameters(parameters):
    """
    Validate simulation parameters for required fields and ranges.
    
    Args:
        parameters (dict): Parameter dictionary to validate
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    required_fields = [
        'time_step', 'total_hours', 'max_growth_rate', 'death_rate',
        'glucose_uptake_rate', 'glutamine_uptake_rate'
    ]
    
    errors = []
    
    # Check for required fields
    for field in required_fields:
        if field not in parameters:
            errors.append(f"Missing required parameter: {field}")
    
    # Check for positive values where required
    positive_fields = ['time_step', 'total_hours', 'max_growth_rate']
    for field in positive_fields:
        if field in parameters and parameters[field] <= 0:
            errors.append(f"Parameter {field} must be positive")
    
    return len(errors) == 0, errors
