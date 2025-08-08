"""
Data Export Module

This module handles CSV export functionality for simulation data,
including proper handling of fork segments and parameter changes.

Key Features:
- Comprehensive CSV export with all simulation data
- Proper parameter handling for fork segments
- Timestamped file naming
- Original parameter values before fork points
"""

import streamlit as st
import pandas as pd
from datetime import datetime


def render_export_button():
    """
    Render the CSV export button and handle export functionality.
    
    This function creates a download button that exports all simulation
    data including original and fork segments with proper parameter handling.
    """
    # CSV export functionality
    if st.session_state.simulation_segments:
        if st.button("📁 Export CSV"):
            # Generate CSV data and create download
            csv_data = generate_csv_data()
            df = pd.DataFrame(csv_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bioreactor_simulation_{timestamp}.csv"
            
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False),
                file_name=filename,
                mime="text/csv"
            )


def generate_csv_data():
    """
    Generate comprehensive CSV data from all simulation segments.
    
    This function creates a complete dataset that includes:
    - Original simulation data
    - Fork simulation data
    - Proper parameter values at each time point
    - Flags indicating data source (original vs fork)
    
    Returns:
        list: List of dictionaries containing all simulation data
    """
    csv_data = []
    original_params = st.session_state.simulation_segments[0]['parameters'] if st.session_state.simulation_segments else {}
    
    for segment in st.session_state.simulation_segments:
        segment_name = f"Fork {segment['segment_id']}" if segment['is_fork'] else "Original"
        
        # For fork segments, include original data up to fork point with ORIGINAL parameters
        if segment['is_fork']:
            original_segment = st.session_state.simulation_segments[0]  # Original is always first
            for state in original_segment['states']:
                if state['time'] <= segment['start_time']:
                    row = {
                        'segment_id': segment['segment_id'],
                        'segment_name': segment_name,
                        'is_fork': segment['is_fork'],
                        'time': state['time'],
                        'fork_original_data': True  # Flag for pre-fork original data
                    }
                    # Add state variables from original timeline
                    row.update(state)
                    # Add ORIGINAL parameters (before fork changes)
                    row.update(original_params)
                    csv_data.append(row)
        
        # Add the segment's own data with correct parameters
        for state in segment['states']:
            row = {
                'segment_id': segment['segment_id'],
                'segment_name': segment_name,
                'is_fork': segment['is_fork'],
                'time': state['time'],
                'fork_original_data': False  # This is actual fork/original data
            }
            # Add state variables
            row.update(state)
            # Add appropriate parameters (original for original, modified for forks)
            row.update(segment['parameters'])
            csv_data.append(row)
    
    return csv_data
