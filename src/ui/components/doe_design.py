"""
DoE Design Generation Component

This module provides functionality for generating Design of Experiments matrices
using pyDOE3 package.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import io
import time
from datetime import datetime
from pyDOE3 import *

def generate_doe_design(selected_params: Dict[str, Any], doe_type: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate DoE design matrix using pyDOE3.
    
    Args:
        selected_params: Dictionary of selected parameters with ranges
        doe_type: Type of DoE design to generate
        **kwargs: Additional parameters (levels, n_samples, etc.)
    
    Returns:
        tuple: (design_matrix_dataframe, design_properties_dict)
    """
    
    # Convert selected parameters to factors format
    factors = {}
    for param_name, param_info in selected_params.items():
        factors[param_name] = [param_info['min'], param_info['max']]
    
    n_factors = len(factors)
    factor_names = list(factors.keys())
    
    if doe_type == "Full Factorial":
        levels = kwargs.get('levels', 2)
        # Use pyDOE3 fullfact function
        design_matrix = fullfact([levels] * n_factors)
        
        # Convert to DataFrame with proper column names and scaling
        design_data = {}
        for i, factor_name in enumerate(factor_names):
            min_val, max_val = factors[factor_name]
            # Map design matrix values (0, 1, ..., levels-1) to actual ranges
            factor_values = []
            for val in design_matrix[:, i]:
                mapped_val = min_val + (val / (levels - 1)) * (max_val - min_val)
                factor_values.append(mapped_val)
            design_data[factor_name] = factor_values
        design_matrix = pd.DataFrame(design_data)
        
        properties = {
            "type": "Full Factorial",
            "factors": n_factors,
            "levels": levels,
            "runs": len(design_matrix),
            "resolution": "Full",
            "orthogonal": True,
            "balanced": True
        }
    
    elif doe_type == "Central Composite Design":
        # Use pyDOE3 ccdesign function
        design_matrix = ccdesign(n_factors)
        
        # Convert to DataFrame with proper column names and scaling
        design_data = {}
        for i, factor_name in enumerate(factor_names):
            min_val, max_val = factors[factor_name]
            center = (min_val + max_val) / 2
            range_val = (max_val - min_val) / 2
            # Scale from coded (-1, 0, 1) to actual values
            factor_values = [center + val * range_val for val in design_matrix[:, i]]
            design_data[factor_name] = factor_values
        design_matrix = pd.DataFrame(design_data)
        
        properties = {
            "type": "Central Composite Design",
            "factors": n_factors,
            "runs": len(design_matrix),
            "resolution": "Full",
            "orthogonal": True,
            "rotatable": True
        }
    
    elif doe_type == "Plackett-Burman":
        # Use pyDOE3 pbdesign function
        design_matrix = pbdesign(n_factors)
        
        # Convert to DataFrame with proper column names and scaling
        design_data = {}
        for i, factor_name in enumerate(factor_names):
            min_val, max_val = factors[factor_name]
            # Map from coded (-1, 1) to actual values
            factor_values = []
            for val in design_matrix[:, i]:
                if val == -1:
                    factor_values.append(min_val)
                else:
                    factor_values.append(max_val)
            design_data[factor_name] = factor_values
        design_matrix = pd.DataFrame(design_data)
        
        properties = {
            "type": "Plackett-Burman",
            "factors": n_factors,
            "runs": len(design_matrix),
            "resolution": "III",
            "orthogonal": True,
            "balanced": True
        }
    
    elif doe_type == "Box-Behnken Design":
        # Use pyDOE3 bbdesign function
        design_matrix = bbdesign(n_factors)
        
        # Convert to DataFrame with proper column names and scaling
        design_data = {}
        for i, factor_name in enumerate(factor_names):
            min_val, max_val = factors[factor_name]
            center = (min_val + max_val) / 2
            range_val = (max_val - min_val) / 2
            # Scale from coded (-1, 0, 1) to actual values
            factor_values = [center + val * range_val for val in design_matrix[:, i]]
            design_data[factor_name] = factor_values
        design_matrix = pd.DataFrame(design_data)
        
        properties = {
            "type": "Box-Behnken Design",
            "factors": n_factors,
            "runs": len(design_matrix),
            "resolution": "Full",
            "orthogonal": True,
            "balanced": True
        }
    
    elif doe_type == "Latin Hypercube Sampling":
        n_samples = kwargs.get('n_samples', 50)
        # Use pyDOE3 lhs function
        design_matrix = lhs(n_factors, samples=n_samples)
        
        # Convert to DataFrame with proper column names and scaling
        design_data = {}
        for i, factor_name in enumerate(factor_names):
            min_val, max_val = factors[factor_name]
            # Scale from [0,1] to actual range
            factor_values = [min_val + val * (max_val - min_val) for val in design_matrix[:, i]]
            design_data[factor_name] = factor_values
        design_matrix = pd.DataFrame(design_data)
        
        properties = {
            "type": "Latin Hypercube Sampling",
            "factors": n_factors,
            "runs": len(design_matrix),
            "space_filling": True,
            "orthogonal": False,
            "balanced": False
        }
    
    else:
        raise ValueError(f"Unsupported DoE type: {doe_type}")
    
    # Add run numbers and predicted fork numbers
    design_matrix.insert(0, 'Run', range(1, len(design_matrix) + 1))
    
    # Calculate predicted fork numbers based on current simulation state
    if 'simulation_segments' in st.session_state and st.session_state.simulation_segments:
        starting_fork_id = max([seg['segment_id'] for seg in st.session_state.simulation_segments]) + 1
    else:
        starting_fork_id = 1  # Start with 1 if no segments exist yet
    
    predicted_fork_numbers = range(starting_fork_id, starting_fork_id + len(design_matrix))
    design_matrix.insert(1, 'Fork', predicted_fork_numbers)
    
    # Add design efficiency metrics
    properties["efficiency"] = calculate_design_efficiency(design_matrix, factors)
    
    return design_matrix, properties

def calculate_design_efficiency(design_matrix: pd.DataFrame, factors: Dict[str, list]) -> Dict[str, float]:
    """
    Calculate advanced design efficiency metrics similar to DoEgen's approach.
    
    This function computes sophisticated design properties including:
    - True D-efficiency using determinant of X'X matrix
    - Condition number for design stability
    - Maximum correlation for orthogonality assessment
    - Space coverage across factor ranges
    - G-efficiency for prediction variance
    """
    
    # Remove Run and Fork columns for calculations
    factor_data = design_matrix.drop(['Run', 'Fork'], axis=1, errors='ignore')
    
    if factor_data.empty:
        return {
            "space_coverage": 0.0,
            "d_efficiency": 0.0,
            "runs_per_factor": 0.0,
            "condition_number": float('inf'),
            "max_correlation": 1.0,
            "g_efficiency": 0.0,
            "orthogonal": False,
            "well_conditioned": False
        }
    
    n_runs = len(factor_data)
    n_factors = len(factors)
    
    # Convert to numpy array for matrix operations
    X = factor_data.values
    
    # 1. SPACE COVERAGE CALCULATION
    # How well the design covers the specified parameter ranges
    space_coverage = 0.0
    for factor_name in factors.keys():
        if factor_name in factor_data.columns:
            min_val, max_val = factors[factor_name]
            factor_range = max_val - min_val
            actual_range = factor_data[factor_name].max() - factor_data[factor_name].min()
            space_coverage += actual_range / factor_range if factor_range > 0 else 1.0
    
    space_coverage = space_coverage / n_factors if n_factors > 0 else 0.0
    
    # 2. TRUE D-EFFICIENCY CALCULATION
    # Based on determinant of information matrix X'X
    d_efficiency = 0.0
    condition_number = float('inf')
    max_correlation = 1.0
    g_efficiency = 0.0
    
    try:
        # Standardize the design matrix (mean=0, std=1)
        X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
        
        # Calculate information matrix X'X
        XtX = X_std.T @ X_std
        
        # D-efficiency: (det(X'X))^(1/p) / n
        det_XtX = np.linalg.det(XtX)
        if det_XtX > 0:
            d_efficiency = (det_XtX ** (1/n_factors)) / n_runs
        
        # Condition number for design stability
        condition_number = np.linalg.cond(XtX)
        
        # G-efficiency: 1 / max(diagonal of (X'X)^-1 * X)
        try:
            XtX_inv = np.linalg.inv(XtX)
            variance_ratios = np.diagonal(X_std @ XtX_inv @ X_std.T)
            max_variance = np.max(variance_ratios)
            g_efficiency = 1.0 / max_variance if max_variance > 0 else 0.0
        except:
            g_efficiency = 0.0
        
    except (np.linalg.LinAlgError, ValueError):
        # Handle singular matrices or other numerical issues
        d_efficiency = 0.0
        condition_number = float('inf')
        g_efficiency = 0.0
    
    # 3. CORRELATION ANALYSIS
    # Maximum absolute correlation between factors
    try:
        correlation_matrix = np.corrcoef(X.T)
        # Get off-diagonal elements (correlations between different factors)
        correlation_matrix_no_diag = correlation_matrix.copy()
        np.fill_diagonal(correlation_matrix_no_diag, 0)
        max_correlation = np.max(np.abs(correlation_matrix_no_diag))
    except:
        max_correlation = 1.0
    
    # 4. DESIGN QUALITY INDICATORS
    orthogonal = max_correlation < 0.1  # Factors are nearly orthogonal
    well_conditioned = condition_number < 100  # Design matrix is well-conditioned
    
    # 5. NORMALIZE METRICS
    # Ensure all metrics are in reasonable ranges
    d_efficiency = min(1.0, max(0.0, d_efficiency))
    g_efficiency = min(1.0, max(0.0, g_efficiency))
    space_coverage = min(1.0, max(0.0, space_coverage))
    max_correlation = min(1.0, max(0.0, max_correlation))
    
    return {
        "space_coverage": space_coverage,
        "d_efficiency": d_efficiency,
        "g_efficiency": g_efficiency,
        "runs_per_factor": n_runs / n_factors if n_factors > 0 else 0,
        "condition_number": condition_number,
        "max_correlation": max_correlation,
        "orthogonal": orthogonal,
        "well_conditioned": well_conditioned
    }

def render_design_matrix(design_matrix: pd.DataFrame, properties: Dict[str, Any]) -> None:
    """Render the design matrix with interactive features"""
    
    if design_matrix.empty:
        st.error("No design matrix to display")
        return
    
    # Design properties summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", properties.get("runs", 0))
    
    with col2:
        st.metric("Factors", properties.get("factors", 0))
    
    with col3:
        efficiency = properties.get("efficiency", {})
        space_cov = efficiency.get("space_coverage", 0)
        st.metric("Space Coverage", f"{space_cov:.1%}")
    
    with col4:
        d_eff = efficiency.get("d_efficiency", 0)
        st.metric("D-Efficiency", f"{d_eff:.1%}")
    
    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        g_eff = efficiency.get("g_efficiency", 0)
        st.metric("G-Efficiency", f"{g_eff:.1%}")
    
    with col6:
        max_corr = efficiency.get("max_correlation", 0)
        st.metric("Max Correlation", f"{max_corr:.3f}")
    
    with col7:
        cond_num = efficiency.get("condition_number", float('inf'))
        if cond_num == float('inf'):
            cond_display = "∞"
        else:
            cond_display = f"{cond_num:.1f}"
        st.metric("Condition Number", cond_display)
    
    with col8:
        orthogonal = efficiency.get("orthogonal", False)
        well_cond = efficiency.get("well_conditioned", False)
        if orthogonal and well_cond:
            quality_score = "Excellent"
        elif orthogonal:
            quality_score = "Good"
        else:
            quality_score = "Fair"
        st.metric("Design Quality", quality_score)
    
    # Design characteristics above the matrix
    st.markdown("###Design Characteristics")
    char_col1, char_col2, char_col3, char_col4 = st.columns(4)
    
    with char_col1:
        st.write(f"**Type:** {properties.get('type', 'Unknown')}")
    
    with char_col2:
        st.write(f"**Resolution:** {properties.get('resolution', 'N/A')}")
    
    with char_col3:
        orthogonal_status = '✅ Yes' if properties.get('orthogonal', False) else '❌ No'
        st.write(f"**Orthogonal:** {orthogonal_status}")
    
    with char_col4:
        balanced_status = '✅ Yes' if properties.get('balanced', False) else '❌ No'
        st.write(f"**Balanced:** {balanced_status}")
    
    # Design matrix table
    st.markdown("### 📋 Design Matrix")
    st.dataframe(
        design_matrix,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Run": st.column_config.NumberColumn("Run #", width="small"),
            "Fork": st.column_config.NumberColumn("Fork #", width="small")
        }
    )

def export_design_matrix(design_matrix: pd.DataFrame, properties: Dict[str, Any]) -> None:
    """Provide export options for the design matrix"""
    
    if design_matrix.empty:
        return
    
    # CSV Export
    csv_buffer = io.StringIO()
    design_matrix.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📊 Download CSV",
        data=csv_data,
        file_name=f"doe_design_{properties.get('type', 'unknown').replace(' ', '_').lower()}.csv",
        mime="text/csv",
        help="Download design matrix as CSV file"
    )

def merge_parameters_for_experiment(base_params: Dict[str, Any], experiment_row: pd.Series) -> Dict[str, Any]:
    """
    Merge base simulation parameters with DoE experiment factors.
    
    Args:
        base_params: Base parameters from simulation tab session state
        experiment_row: Single row from design matrix with DoE factor values
    
    Returns:
        dict: Combined parameters for this specific experiment
    """
    # Start with all base parameters from simulation tab
    experiment_params = base_params.copy()
    
    # Override only the DoE factors for this experiment
    for col in experiment_row.index:
        if col not in ['Run', 'Fork']:  # Skip the Run and Fork columns
            experiment_params[col] = experiment_row[col]
    
    return experiment_params

def extract_simulation_responses(sim_results: list) -> Dict[str, float]:
    """
    Extract final simulation state with same variable names as Simulation tab.
    
    Args:
        sim_results: List of simulation state dictionaries over time
    
    Returns:
        dict: Dictionary of final state variables using original parameter names
    """
    if not sim_results:
        return {
            'time': 0.0,
            'viable_cell_density': 0.0,
            'dead_cell_density': 0.0,
            'glucose_concentration': 0.0,
            'glutamine_concentration': 0.0,
            'lactate_concentration': 0.0,
            'ammonia_concentration': 0.0,
            'product_concentration': 0.0,
            'aggregated_product': 0.0,
            'pump_active': 0
        }
    
    # Return the final state directly with same names as simulation
    final_state = sim_results[-1]
    
    return {
        'time': final_state['time'],
        'viable_cell_density': final_state['viable_cell_density'],
        'dead_cell_density': final_state['dead_cell_density'],
        'glucose_concentration': final_state['glucose_concentration'],
        'glutamine_concentration': final_state['glutamine_concentration'],
        'lactate_concentration': final_state['lactate_concentration'],
        'ammonia_concentration': final_state['ammonia_concentration'],
        'product_concentration': final_state['product_concentration'],
        'aggregated_product': final_state['aggregated_product'],
        'pump_active': final_state['pump_active']
    }

def run_doe_batch_simulation(design_matrix: pd.DataFrame, base_params: Dict[str, Any]) -> pd.DataFrame:
    """
    Run all DoE experiments automatically.
    
    Args:
        design_matrix: DataFrame with experimental design
        base_params: Base simulation parameters from session state
    
    Returns:
        DataFrame: Results with factors and responses for all experiments
    """
    # Import simulation function
    from .simulation_engine import execute_simulation
    
    results = []
    total_runs = len(design_matrix)
    
    # Create progress tracking
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🚀 Batch Simulation Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        eta_col1, eta_col2 = st.columns(2)
        
        with eta_col1:
            completed_metric = st.empty()
        with eta_col2:
            eta_metric = st.empty()
    
    start_time = time.time()
    
    # Calculate starting fork number based on existing simulation segments
    if st.session_state.simulation_segments:
        starting_fork_id = max([seg['segment_id'] for seg in st.session_state.simulation_segments]) + 1
    else:
        starting_fork_id = 1  # Start with 1 if no segments exist yet
    
    for i, (index, experiment) in enumerate(design_matrix.iterrows()):
        # Calculate actual fork number that would be created
        actual_fork_number = starting_fork_id + i
        
        # Update status
        status_text.text(f"Running experiment {i+1}/{total_runs}: Run {experiment['Run']} (Fork {actual_fork_number})")
        
        # Merge base parameters with experiment parameters
        exp_parameters = merge_parameters_for_experiment(base_params, experiment)
        
        try:
            # Run single simulation
            sim_results = execute_simulation(exp_parameters)
            
            # Extract key responses
            responses = extract_simulation_responses(sim_results)
            
            # Store results with actual fork number
            experiment_result = {
                'Run': experiment['Run'],
                'Fork': actual_fork_number,
                **{col: experiment[col] for col in design_matrix.columns if col not in ['Run', 'Fork']},
                **responses
            }
            results.append(experiment_result)
            
            # Update progress
            progress = (i + 1) / total_runs
            progress_bar.progress(progress)
            
            # Update metrics
            completed_metric.metric("Completed", f"{i+1}/{total_runs}")
            
            # Calculate ETA
            elapsed_time = time.time() - start_time
            if i > 0:
                avg_time_per_run = elapsed_time / (i + 1)
                remaining_runs = total_runs - (i + 1)
                eta_seconds = remaining_runs * avg_time_per_run
                eta_minutes = eta_seconds / 60
                eta_metric.metric("ETA", f"{eta_minutes:.1f} min")
            
        except Exception as e:
            st.error(f"Error in Run {experiment['Run']}: {str(e)}")
            # Add failed result with NaN values using same names as simulation
            failed_result = {
                'Run': experiment['Run'],
                'Fork': actual_fork_number,
                **{col: experiment[col] for col in design_matrix.columns if col not in ['Run', 'Fork']},
                'time': np.nan,
                'viable_cell_density': np.nan,
                'dead_cell_density': np.nan,
                'glucose_concentration': np.nan,
                'glutamine_concentration': np.nan,
                'lactate_concentration': np.nan,
                'ammonia_concentration': np.nan,
                'product_concentration': np.nan,
                'aggregated_product': np.nan,
                'pump_active': np.nan
            }
            results.append(failed_result)
    
    # Final status
    total_time = time.time() - start_time
    status_text.text(f"✅ Batch completed! Total time: {total_time/60:.1f} minutes")
    progress_bar.progress(1.0)
    
    return pd.DataFrame(results)

def render_batch_results(results_df: pd.DataFrame) -> None:
    """
    Display batch simulation results with analysis.
    
    Args:
        results_df: DataFrame with experiment results
    """
    if results_df.empty:
        st.warning("No batch results to display")
        return
    
    st.markdown("### 📊 Batch Simulation Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        successful_runs = results_df['viable_cell_density'].notna().sum()
        st.metric("Successful Runs", f"{successful_runs}/{len(results_df)}")
    
    with col2:
        if successful_runs > 0:
            avg_product = results_df['aggregated_product'].mean()
            st.metric("Avg Product", f"{avg_product:.3f}")
        else:
            st.metric("Avg Product", "N/A")
    
    with col3:
        if successful_runs > 0:
            best_product = results_df['aggregated_product'].max()
            st.metric("Best Product", f"{best_product:.3f}")
        else:
            st.metric("Best Product", "N/A")
    
    with col4:
        if successful_runs > 0:
            avg_duration = results_df['time'].mean()
            st.metric("Avg Duration", f"{avg_duration:.1f} h")
        else:
            st.metric("Avg Duration", "N/A")
    
    # Results table
    st.markdown("#### 📋 Detailed Results")
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Run": st.column_config.NumberColumn("Run #", width="small"),
            "Fork": st.column_config.NumberColumn("Fork #", width="small"),
            "time": st.column_config.NumberColumn("Time (h)", format="%.1f"),
            "viable_cell_density": st.column_config.NumberColumn("Viable Density", format="%.2f"),
            "dead_cell_density": st.column_config.NumberColumn("Dead Density", format="%.2f"),
            "glucose_concentration": st.column_config.NumberColumn("Glucose", format="%.2f"),
            "glutamine_concentration": st.column_config.NumberColumn("Glutamine", format="%.2f"),
            "lactate_concentration": st.column_config.NumberColumn("Lactate", format="%.2f"),
            "ammonia_concentration": st.column_config.NumberColumn("Ammonia", format="%.2f"),
            "product_concentration": st.column_config.NumberColumn("Product Conc", format="%.3f"),
            "aggregated_product": st.column_config.NumberColumn("Total Product", format="%.3f"),
            "pump_active": st.column_config.NumberColumn("Pump Active", format="%.0f")
        }
    )
    
    # Export options
    st.markdown("#### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Export of results
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"doe_batch_results_{timestamp}.csv"
        
        st.download_button(
            label="📊 Download Results CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            help="Download complete batch results"
        )
    
    with col2:
        # Summary statistics export
        if successful_runs > 0:
            summary_stats = results_df.describe()
            summary_csv = summary_stats.to_csv()
            
            st.download_button(
                label="📈 Download Summary Stats",
                data=summary_csv,
                file_name=f"doe_summary_stats_{timestamp}.csv",
                mime="text/csv",
                help="Download statistical summary"
            )

def render_batch_controls(design_matrix: pd.DataFrame, design_properties: Dict[str, Any]) -> None:
    """
    Render batch simulation controls and execution interface.
    
    Args:
        design_matrix: The DoE design matrix to execute
        design_properties: Properties of the design
    """
    if design_matrix.empty:
        st.warning("No design matrix available for batch simulation")
        return
    
    st.markdown("### 🎯 Batch Simulation")
    
    # Check if simulation has been run before (which means parameters are configured)
    has_simulation_data = ('simulation_segments' in st.session_state and 
                          len(st.session_state.simulation_segments) > 0)
    
    has_base_parameters = ('base_parameters' in st.session_state and 
                          st.session_state.base_parameters is not None)
    
    if not has_simulation_data and not has_base_parameters:
        st.warning("⚠️ **Configuration Required**")
        st.info("""
        **Before running batch experiments, please:**
        1. Go to the **Simulation** tab
        2. Configure your simulation parameters  
        3. Run at least one simulation to establish baseline parameters
        4. Return to this tab to run batch experiments
        
        This ensures your DoE experiments use validated parameter settings.
        """)
        
        st.button(
            "🚀 Run All Experiments",
            disabled=True,
            help="Complete the simulation setup first"
        )
        return
    
    # Batch execution controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"**Ready to run {len(design_matrix)} experiments**")
        st.write(f"Design Type: {design_properties.get('type', 'Unknown')}")
        
        # Estimated time calculation
        est_time_per_run = 2.0  # Estimate 2 seconds per simulation
        total_est_time = len(design_matrix) * est_time_per_run / 60  # Convert to minutes
        st.write(f"Estimated Time: {total_est_time:.1f} minutes")
    
    with col2:
        run_batch = st.button(
            "🚀 Run All Experiments",
            type="primary",
            use_container_width=True,
            help="Execute all experiments in the design matrix"
        )
    
    # Execute batch if button clicked
    if run_batch:
        try:
            # Get base parameters from session state
            base_params = st.session_state.base_parameters.copy()
            
            # Run batch simulation
            with st.spinner("Running batch simulation..."):
                results_df = run_doe_batch_simulation(design_matrix, base_params)
            
            # Store results in session state
            st.session_state['doe_batch_results'] = {
                'results_df': results_df,
                'design_matrix': design_matrix,
                'design_properties': design_properties,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.success("✅ Batch simulation completed!")
            
        except Exception as e:
            st.error(f"❌ Batch simulation failed: {str(e)}")
    
    # Display results if available
    if ('doe_batch_results' in st.session_state and 
        st.session_state['doe_batch_results'] is not None):
        results_data = st.session_state['doe_batch_results']
        st.markdown("---")
        render_batch_results(results_data['results_df'])
