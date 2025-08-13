"""
Visualization Module

This module handles all visualization functionality including individual plots,
grid layout, legends, and interactive features for the simulation interface.

Key Features:
- Individual plot creation for each measurement variable
- Grid layout management
- Interactive legends with fork comparison
- Parameter change visualization in hover tooltips
"""

import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to Python path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import get_parameter_changes, get_y_axis_label, format_variable_name, apply_measurement_noise
from utils.helpers import get_parameter_changes, get_y_axis_label, format_variable_name


def render_visualization_layout():
    """
    Render the complete visualization layout with plots and legend.
    
    This function creates the main visualization section including:
    - Individual plots in a 4x2 grid layout
    - Comprehensive legend showing all segments
    - Parameter change information for forks
    """
    if st.session_state.simulation_segments:
        # Get original parameters for fork comparison
        original_params = st.session_state.simulation_segments[0]['parameters']
        
        st.markdown("### Simulation Results")
        
        # Render individual plots in grid
        render_plots_grid(original_params)
        
        # Render comprehensive legend
        render_comprehensive_legend(original_params)
    else:
        # Render help section for initial state
        render_help_section()


def render_plots_grid(original_params):
    """
    Render individual plots in a 3x3 grid layout.
    
    Args:
        original_params (dict): Original simulation parameters for comparison
    """
    # Define measurement variables for individual plots
    plot_variables = [
        'viable_cell_density',    # Cell viability measurements
        'dead_cell_density', 
        'product_concentration',   # Product in reactor
        'aggregated_product',     # Total product produced
        'glucose_concentration',   # Nutrient concentrations
        'glutamine_concentration',
        'lactate_concentration',   # Metabolite concentrations
        'ammonia_concentration',
        'pump_active'             # Process control state
    ]
    
    # Create 3x3 grid layout for individual plots
    for row in range(3):
        cols = st.columns(3)
        for col in range(3):
            var_index = row * 3 + col
            if var_index < len(plot_variables):
                variable = plot_variables[var_index]
                
                with cols[col]:
                    # Create individual plot with legend enabled
                    fig = create_individual_plot(
                        variable, 
                        st.session_state.simulation_segments,
                        st.session_state.current_time_point,
                        original_params,
                        show_legend=True  # Show legend on every plot
                    )
                    st.plotly_chart(fig, use_container_width=True)


def create_individual_plot(variable_name, segments, current_time_point, original_params, show_legend=False):
    """
    Create individual Plotly chart for a specific measurement variable.
    
    Args:
        variable_name (str): Name of the variable to plot (e.g., 'viable_cell_density')
        segments (list): List of simulation segments (original + forks)
        current_time_point (float): Current timeline position for vertical line
        original_params (dict): Original parameters for comparison
        show_legend (bool): Whether to display legend on this plot
        
    Returns:
        plotly.graph_objects.Figure: Configured Plotly figure for the variable
    """
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Add trace for each simulation segment
    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        line_style = 'dash' if segment['is_fork'] else 'solid'
        segment_name = f"Fork {segment['segment_id']}" if segment['is_fork'] else "Original"
        
        # Apply measurement noise to states for visualization
        measurement_noise = segment['parameters'].get('measurement_noise', 0.0)
        noisy_states = apply_measurement_noise(segment['states'], measurement_noise)
        
        # Extract time series data from noisy states
        time_points = [s['time'] for s in noisy_states]
        y_data = [s[variable_name] for s in noisy_states]
        
        # Generate parameter change information for hover
        param_changes = get_parameter_changes(segment, original_params) if segment['is_fork'] else []
        changes_text = f"<br>Changed: {', '.join(param_changes)}" if param_changes else ""
        
        # Add noise indicator to hover if noise is applied
        noise_text = f"<br>Noise: {measurement_noise*100:.1f}%" if measurement_noise > 0 else ""
        
        # Add trace to figure
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=y_data,
                name=segment_name,
                line={'color': color, 'dash': line_style},
                showlegend=show_legend,
                hovertemplate=f"<b>{format_variable_name(variable_name)}</b><br>" +
                            "Time: %{x:.1f} hours<br>" +
                            "Value: %{y:.3f}<br>" +
                            f"Segment: {segment_name}" +
                            changes_text +
                            noise_text +
                            "<extra></extra>"
            )
        )
    
    # Add vertical line indicating current time position
    fig.add_vline(
        x=current_time_point,
        line_dash="dot",
        line_color="black",
        opacity=0.7
    )
    
    # Configure plot layout
    fig.update_layout(
        title=format_variable_name(variable_name),
        xaxis_title="Time (hours)",
        yaxis_title=get_y_axis_label(variable_name),
        height=300,
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40},
        legend={
            'orientation': "h",
            'yanchor': "top",
            'y': -0.2,
            'xanchor': "center",
            'x': 0.5
        } if show_legend else None
    )
    
    return fig


def render_comprehensive_legend(original_params):
    """
    Render comprehensive legend showing all segments with parameter changes.
    
    Args:
        original_params (dict): Original simulation parameters for comparison
    """
    st.markdown("### Legend")
    legend_cols = st.columns(len(st.session_state.simulation_segments))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Create legend entry for each simulation segment
    for i, segment in enumerate(st.session_state.simulation_segments):
        color = colors[i % len(colors)]
        segment_name = f"Fork {segment['segment_id']}" if segment['is_fork'] else "Original"
        
        with legend_cols[i]:
            # Create visual line indicator for segment
            mini_fig = go.Figure()
            mini_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 0],
                    mode='lines',
                    line={'color': color, 'dash': 'dash' if segment['is_fork'] else 'solid', 'width': 3},
                    showlegend=False
                )
            )
            mini_fig.update_layout(
                height=50,
                margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
                xaxis={'visible': False},
                yaxis={'visible': False},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(mini_fig, use_container_width=True)
            st.markdown(f"**{segment_name}**")
            
            # Display parameter changes for fork segments
            if segment['is_fork']:
                param_changes = get_parameter_changes(segment, original_params)
                if param_changes:
                    st.markdown("*Changed parameters:*")
                    # Show first 3 parameter changes
                    for change in param_changes[:3]:
                        st.markdown(f"• {change}")
                    # Indicate if more changes exist
                    if len(param_changes) > 3:
                        st.markdown(f"• +{len(param_changes)-3} more...")


def render_help_section():
    """
    Render the help section for initial state when no simulation data exists.
    """
    st.info("Run the initial simulation to begin analysis")
    st.markdown("""
    ### How to use this interface:
    1. **Set parameters** in the sidebar and click "Run Initial Simulation"
    2. **Use the timeline slider** to navigate to any time point
    3. **Modify parameters** in the sidebar at your chosen time point
    4. **Click "Continue from Current Point"** to create a fork with new parameters
    5. **Use "Reset All"** to clear all simulations and start over
    6. **Click "Export CSV"** to download all simulation data
    
    The graphs will show:
    - **Solid lines**: Original simulation path
    - **Dashed lines**: Fork simulations with modified parameters
    - **Vertical dotted line**: Current timeline position
    """)
