"""
DoE Optimization Module

This module provides optimization functionality for Design of Experiments,
including response surface modeling and parameter optimization using scikit-optimize.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn not available. Install with: pip install scikit-learn")

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ SciPy not available. Install with: pip install scipy")


def fit_response_models(results_df: pd.DataFrame, selected_params: Dict[str, Any], selected_responses: Dict[str, Any], model_type: str = "polynomial") -> Dict[str, Any]:
    """
    Fit predictive models for each response using the selected modeling approach.
    
    Args:
        results_df: DataFrame with experimental results
        selected_params: Dictionary of selected DoE parameters
        selected_responses: Dictionary of selected responses to optimize
        model_type: Type of model ("polynomial", "gaussian_process", "random_forest")
        
    Returns:
        dict: Dictionary of fitted models for each response
    """
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is required for optimization. Please install it.")
        return {}
    
    models = {}
    
    # Get factor columns (input parameters)
    factor_names = list(selected_params.keys())
    
    # Check if all factor columns exist in results
    missing_factors = [f for f in factor_names if f not in results_df.columns]
    if missing_factors:
        st.error(f"Missing factor columns in results: {missing_factors}")
        return {}
    
    X = results_df[factor_names].values
    
    for response_name in selected_responses.keys():
        if response_name not in results_df.columns:
            st.warning(f"Response '{response_name}' not found in results data")
            continue
            
        y = results_df[response_name].values
        
        # Remove any NaN values
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        if len(X_clean) < 3:
            st.warning(f"Not enough valid data points for {response_name}")
            continue
        
        try:
            if model_type == "polynomial":
                # Polynomial Regression (Response Surface Methodology)
                poly_features = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
                X_poly = poly_features.fit_transform(X_clean)
                model = LinearRegression()
                model.fit(X_poly, y_clean)
                
                # Calculate performance
                y_pred = model.predict(X_poly)
                r2 = r2_score(y_clean, y_pred)
                
                models[response_name] = {
                    'model': model,
                    'poly_features': poly_features,
                    'factor_names': factor_names,
                    'r2_score': r2,
                    'cv_score': r2,  # Simplified for polynomial
                    'n_samples': len(X_clean),
                    'model_type': 'polynomial'
                }
                
            elif model_type == "gaussian_process":
                # Gaussian Process Regression
                kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3)
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    random_state=42
                )
                model.fit(X_clean, y_clean)
                
                # Calculate performance
                y_pred = model.predict(X_clean)
                r2 = r2_score(y_clean, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_clean, y_clean, cv=min(3, len(X_clean)), scoring='r2')
                cv_mean = cv_scores.mean() if len(cv_scores) > 0 else r2
                
                models[response_name] = {
                    'model': model,
                    'factor_names': factor_names,
                    'r2_score': r2,
                    'cv_score': cv_mean,
                    'n_samples': len(X_clean),
                    'model_type': 'gaussian_process',
                    'log_likelihood': model.log_marginal_likelihood()
                }
                
            elif model_type == "random_forest":
                # Random Forest Regression
                model = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt'
                )
                model.fit(X_clean, y_clean)
                
                # Calculate performance
                y_pred = model.predict(X_clean)
                r2 = r2_score(y_clean, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_clean, y_clean, cv=min(5, len(X_clean)), scoring='r2')
                cv_mean = cv_scores.mean() if len(cv_scores) > 0 else r2
                
                models[response_name] = {
                    'model': model,
                    'factor_names': factor_names,
                    'r2_score': r2,
                    'cv_score': cv_mean,
                    'n_samples': len(X_clean),
                    'model_type': 'random_forest'
                }
            
        except Exception as e:
            st.warning(f"Failed to fit {model_type} model for {response_name}: {str(e)}")
    
    return models


def optimize_parameters(models: Dict[str, Any], selected_params: Dict[str, Any], selected_responses: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize parameters using differential evolution to maximize/minimize selected responses.
    
    Args:
        models: Dictionary of fitted models for each response
        selected_params: Dictionary of selected DoE parameters with ranges
        selected_responses: Dictionary of selected responses with objectives
        
    Returns:
        dict: Optimization results
    """
    if not SCIPY_AVAILABLE:
        st.error("scipy is required for optimization. Please install it.")
        return {}
    
    if not models:
        st.warning("No fitted models available for optimization")
        return {}
    
    # Get parameter bounds
    factor_names = list(selected_params.keys())
    bounds = []
    
    for param_name in factor_names:
        param_info = selected_params[param_name]
        if isinstance(param_info, dict):
            # Check for different possible parameter formats
            if 'range' in param_info:
                bounds.append(param_info['range'])
            elif 'min' in param_info and 'max' in param_info:
                bounds.append((param_info['min'], param_info['max']))
            else:
                st.error(f"Invalid parameter configuration for {param_name}. Missing 'range' or 'min'/'max' fields.")
                return {}
        else:
            st.error(f"Invalid parameter configuration for {param_name}. Expected dictionary format.")
            return {}
    
    def objective_function(x):
        """Objective function for optimization"""
        total_score = 0.0
        
        for response_name, response_config in selected_responses.items():
            if response_name not in models:
                continue
                
            model_info = models[response_name]
            model = model_info['model']
            
            try:
                # Predict response value
                if model_info['model_type'] == 'polynomial':
                    poly_features = model_info['poly_features']
                    x_poly = poly_features.transform(x.reshape(1, -1))
                    pred = model.predict(x_poly)[0]
                else:
                    pred = model.predict(x.reshape(1, -1))[0]
                
                # Apply objective (maximize = minimize negative)
                if response_config.get('objective') == 'maximize':
                    total_score -= pred
                else:  # minimize
                    total_score += pred
                    
            except Exception:
                # Penalty for invalid predictions
                total_score += 1e6
        
        return total_score
    
    try:
        # Run optimization
        result = differential_evolution(
            objective_function,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            atol=1e-6,
            tol=1e-6
        )
        
        if result.success:
            optimal_params = dict(zip(factor_names, result.x))
            
            # Get predictions for optimal parameters
            predictions = {}
            for response_name, model_info in models.items():
                model = model_info['model']
                try:
                    if model_info['model_type'] == 'polynomial':
                        poly_features = model_info['poly_features']
                        x_poly = poly_features.transform(result.x.reshape(1, -1))
                        pred = model.predict(x_poly)[0]
                    else:
                        pred = model.predict(result.x.reshape(1, -1))[0]
                    predictions[response_name] = pred
                except Exception:
                    predictions[response_name] = None
            
            return {
                'success': True,
                'optimal_params': optimal_params,
                'predictions': predictions,
                'objective_value': result.fun,
                'n_iterations': result.nit,
                'n_evaluations': result.nfev
            }
        else:
            return {
                'success': False,
                'message': f"Optimization failed: {result.message}"
            }
    
    except Exception as e:
        return {
            'success': False,
            'message': f"Optimization error: {str(e)}"
        }


def predict_optimal_performance(optimal_params: Dict[str, float], models: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Predict performance at optimal parameter settings.
    
    Args:
        optimal_params: Dictionary of optimal parameter values
        models: Dictionary of fitted models
        selected_responses: Dictionary of response configurations
        
    Returns:
        dict: Dictionary of predictions for each response
    """
    predictions = {}
    
    # Convert optimal params to array in correct order
    factor_names = list(optimal_params.keys())
    x_optimal = [optimal_params[name] for name in factor_names]
    
    for response_name, model_data in models.items():
        try:
            # Predict response based on model type
            if model_data['model_type'] == 'polynomial':
                # For polynomial models, use the polynomial features transform
                poly_features = model_data['poly_features']
                x_poly = poly_features.transform(np.array(x_optimal).reshape(1, -1))
                predicted_value = model_data['model'].predict(x_poly)[0]
            else:
                # For other models, use direct prediction
                predicted_value = model_data['model'].predict(np.array(x_optimal).reshape(1, -1))[0]
            
            predictions[response_name] = {
                'predicted_value': predicted_value,
                'model_r2': model_data['r2_score'],
                'cv_score': model_data['cv_score'],
                'n_samples': model_data['n_samples']
            }
            
        except Exception as e:
            st.warning(f"Failed to predict {response_name}: {str(e)}")
            predictions[response_name] = {
                'predicted_value': None,
                'model_r2': model_data.get('r2_score', 0),
                'cv_score': model_data.get('cv_score', 0),
                'n_samples': model_data.get('n_samples', 0)
            }
    
    return predictions


def render_optimization_interface(results_df: pd.DataFrame, selected_params: Dict[str, Any], selected_responses: Dict[str, Any]):
    """
    Render the complete optimization interface.
    
    Args:
        results_df: DataFrame with batch results
        selected_params: Dictionary of selected parameters
        selected_responses: Dictionary of selected responses
    """
    st.markdown("### 🎯 Parameter Optimization")
    
    if not selected_responses:
        st.info("👆 Select responses to optimize in the sidebar")
        return
    
    if len(results_df) < 3:
        st.warning("⚠️ Need at least 3 experimental runs for optimization")
        return
    
    # Model selection interface
    st.markdown("#### 🤖 Select Optimization Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_type = st.selectbox(
            "Choose modeling approach:",
            options=["polynomial", "gaussian_process", "random_forest"],
            format_func=lambda x: {
                "polynomial": "📈 Polynomial Regression (RSM)",
                "gaussian_process": "🔮 Gaussian Process Regression", 
                "random_forest": "🌳 Random Forest Regression"
            }[x],
            key="optimization_model_type",
            help="Select the machine learning model for response prediction"
        )
    
    with col2:
        # Model descriptions
        model_info = {
            "polynomial": "Good for smooth response surfaces with clear trends. Fast and interpretable.",
            "gaussian_process": "Excellent for capturing uncertainty and complex non-linear relationships.", 
            "random_forest": "Robust for noisy data and complex interactions. Less prone to overfitting."
        }
        st.info(f"💡 {model_info[model_type]}")
    
    # Optimization button
    if st.button("🚀 Find Optimal Parameters", type="primary"):
        
        # Debug: Show parameter structure
        with st.expander("🔍 Debug: Parameter Structure", expanded=False):
            st.write("Selected Parameters:")
            st.json(selected_params)
            st.write("Selected Responses:")
            st.json(selected_responses)
        
        with st.spinner("🧠 Training predictive models..."):
            models = fit_response_models(results_df, selected_params, selected_responses, model_type)
        
        if not models:
            st.error("❌ Failed to train models. Check your data and selections.")
            return
        
        # Optimize
        with st.spinner("🔍 Finding optimal parameters..."):
            optimization_result = optimize_parameters(models, selected_params, selected_responses)
        
        if not optimization_result.get('success', False):
            st.error(f"❌ Optimization failed: {optimization_result.get('message', 'Unknown error')}")
            return
        
        optimal_params = optimization_result['optimal_params']
        
        # Predict performance
        predictions = predict_optimal_performance(optimal_params, models)
        
        # Store optimization results in session state to persist across reruns
        st.session_state['optimization_results'] = {
            'models': models,
            'optimal_params': optimal_params,
            'predictions': predictions,
            'optimization_result': optimization_result,
            'model_type': model_type,
            'selected_params': selected_params,
            'selected_responses': selected_responses
        }
        
        st.success("✅ Optimization Complete!")
        st.rerun()  # Rerun to display the stored results
    
    # Display optimization results if they exist in session state
    if 'optimization_results' in st.session_state:
        stored_results = st.session_state['optimization_results']
        models = stored_results['models']
        optimal_params = stored_results['optimal_params']
        predictions = stored_results['predictions']
        model_type = stored_results['model_type']
        selected_params = stored_results['selected_params']
        selected_responses = stored_results['selected_responses']
        
        # Show model quality
        st.markdown("#### 📊 Model Quality")
        
        # Display selected model info
        model_names = {
            "polynomial": "Polynomial Regression (RSM)",
            "gaussian_process": "Gaussian Process Regression",
            "random_forest": "Random Forest Regression"
        }
        st.info(f"🤖 Using **{model_names[model_type]}** for response modeling")
        
        model_cols = st.columns(len(models))
        
        for i, (response_name, model_data) in enumerate(models.items()):
            with model_cols[i]:
                # Base metrics
                metrics_help = f"Cross-validation R² = {model_data['cv_score']:.3f}\nSamples = {model_data['n_samples']}"
                
                # Add model-specific metrics
                if model_data['model_type'] == 'gaussian_process' and 'log_likelihood' in model_data:
                    metrics_help += f"\nLog Likelihood = {model_data['log_likelihood']:.2f}"
                
                st.metric(
                    label=f"{response_name}",
                    value=f"R² = {model_data['r2_score']:.3f}",
                    help=metrics_help
                )
        
        # Check model quality
        min_r2 = min(model['r2_score'] for model in models.values())
        if min_r2 < 0.5:
            st.warning(f"⚠️ Model fit quality is poor (lowest R² = {min_r2:.3f}). Consider running more experiments.")
        elif min_r2 < 0.7:
            st.warning(f"⚠️ Model fit quality is moderate (lowest R² = {min_r2:.3f}). Predictions may be uncertain.")
        
        st.success("✅ Optimization Complete!")
        
        # Show optimal parameters
        st.markdown("#### 🎯 Optimal Parameter Settings")
        param_cols = st.columns(len(optimal_params))
        
        for i, (param_name, optimal_value) in enumerate(optimal_params.items()):
            with param_cols[i]:
                param_info = selected_params[param_name]
                st.metric(
                    label=f"{param_name}",
                    value=f"{optimal_value:.3f}",
                    help=f"Units: {param_info['units']}\nRange: {param_info['min']:.3f} - {param_info['max']:.3f}"
                )
        
        # Show predicted performance
        st.markdown("#### 📈 Predicted Performance")
        pred_cols = st.columns(len(predictions))
        
        for i, (response_name, pred_info) in enumerate(predictions.items()):
            with pred_cols[i]:
                response_config = selected_responses[response_name]
                objective_icon = "📈" if response_config['objective'] == 'maximize' else "📉"
                
                if pred_info['predicted_value'] is not None:
                    st.metric(
                        label=f"{objective_icon} {response_name}",
                        value=f"{pred_info['predicted_value']:.3f}",
                        help=f"Units: {response_config['units']}\nModel R²: {pred_info['model_r2']:.3f}\nCV Score: {pred_info['cv_score']:.3f}"
                    )
                else:
                    st.error(f"❌ Failed to predict {response_name}")
        
        # Run Optimal Parameters Button
        st.markdown("#### 🚀 Execute Optimal Settings")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🎯 Run Optimal Parameters", type="primary", help="Execute simulation with optimal parameter settings"):
                # Execute simulation directly with optimal parameters
                with st.spinner("🚀 Running simulation with optimal parameters..."):
                    # Import simulation engine
                    try:
                        from .doe_design import run_doe_batch_simulation
                        
                        # Convert optimal params to simulation format (single experiment)
                        # Create a design matrix with single row for optimal parameters
                        import pandas as pd
                        design_matrix = pd.DataFrame([optimal_params])
                        # Add required 'Run' column (matching DoE design matrix format)
                        design_matrix.insert(0, 'Run', [1])
                        
                        # Get base parameters from session state (same as used in DoE experiments)
                        if 'base_parameters' not in st.session_state:
                            st.error("❌ Base parameters not found. Please run a simulation first to set base parameters.")
                            return
                        
                        base_params = st.session_state.base_parameters.copy()
                        
                        # Run batch simulation with single optimal parameter set
                        results_df = run_doe_batch_simulation(design_matrix, base_params)
                        
                        if results_df is not None and len(results_df) > 0:
                            result = results_df.iloc[0].to_dict()  # Convert first row to dict
                            
                            # The fork number is already calculated correctly by the batch system
                            fork_number = result.get('Fork', 'Unknown')
                            
                            # Store optimal execution results
                            st.session_state.optimal_execution_results = {
                                'fork_number': fork_number,
                                'simulation_results': result,
                                'executed_params': optimal_params.copy(),
                                'predicted_vs_actual': {}
                            }
                            
                            # Compare predicted vs actual for each response
                            for response_name in selected_responses.keys():
                                if response_name in result and response_name in predictions:
                                    predicted_val = predictions[response_name]['predicted_value']
                                    actual_val = result[response_name]
                                    if predicted_val is not None and actual_val is not None:
                                        error_pct = abs((actual_val - predicted_val) / predicted_val * 100) if predicted_val != 0 else 0
                                        st.session_state.optimal_execution_results['predicted_vs_actual'][response_name] = {
                                            'predicted': predicted_val,
                                            'actual': actual_val,
                                            'error_percent': error_pct
                                        }
                            
                            st.success(f"✅ Optimal parameters executed successfully! Fork #{fork_number}")
                            st.rerun()
                            
                        else:
                            st.error("❌ Simulation returned no results")
                    
                    except Exception as e:
                        st.error(f"❌ Simulation failed: {str(e)}")
                        st.exception(e)
        
        with col2:
            if st.button("🔄 Clear Results", help="Clear optimization results and start over"):
                if 'optimization_results' in st.session_state:
                    del st.session_state['optimization_results']
                st.rerun()
    
    # Display optimal execution results if available
    if 'optimal_execution_results' in st.session_state:
        execution_info = st.session_state['optimal_execution_results']
        
        st.markdown("---")
        st.markdown("### ✅ Optimal Parameters Execution Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="🔗 Fork Number",
                value=f"#{execution_info['fork_number']}",
                help="Simulation fork number for tracking results"
            )
        
        with col2:
            if st.button("🗑️ Clear Execution Results", help="Clear the optimal execution results"):
                del st.session_state['optimal_execution_results']
                st.rerun()
        
        # Show simulation results
        st.markdown("#### 📊 Simulation Results")
        sim_results = execution_info['simulation_results']
        
        # Display key outputs using correct variable names from extract_simulation_responses
        key_outputs = ['viable_cell_density', 'dead_cell_density', 'glucose_concentration', 'lactate_concentration', 'product_concentration', 'aggregated_product']
        available_outputs = [key for key in key_outputs if key in sim_results]
        
        if available_outputs:
            result_cols = st.columns(len(available_outputs))
            for i, output_name in enumerate(available_outputs):
                with result_cols[i]:
                    value = sim_results[output_name]
                    # Format display name
                    display_name = output_name.replace('_', ' ').title()
                    if 'concentration' in output_name:
                        display_name = display_name.replace(' Concentration', '')
                    
                    st.metric(
                        label=display_name,
                        value=f"{value:.3f}",
                        help=f"Final simulation value for {output_name}"
                    )
        
        # Show predicted vs actual comparison if available
        if 'predicted_vs_actual' in execution_info and execution_info['predicted_vs_actual']:
            st.markdown("#### 🎯 Predicted vs Actual Comparison")
            
            comparison_data = execution_info['predicted_vs_actual']
            comp_cols = st.columns(len(comparison_data))
            
            for i, (response_name, comp_info) in enumerate(comparison_data.items()):
                with comp_cols[i]:
                    predicted = comp_info['predicted']
                    actual = comp_info['actual']
                    error_pct = comp_info['error_percent']
                    
                    # Color code based on error percentage
                    if error_pct < 5:
                        delta_color = "normal"
                        status = "🎯 Excellent"
                    elif error_pct < 15:
                        delta_color = "normal" 
                        status = "✅ Good"
                    else:
                        delta_color = "inverse"
                        status = "⚠️ Check"
                    
                    st.metric(
                        label=f"{response_name}",
                        value=f"Actual: {actual:.3f}",
                        delta=f"Predicted: {predicted:.3f}",
                        delta_color=delta_color,
                        help=f"{status} - Error: {error_pct:.1f}%"
                    )
        
        st.info(f"🎯 Optimal parameters successfully executed and results are available in Fork #{execution_info['fork_number']}")


def run_optimal_parameters_in_simulation(optimal_params: Dict[str, float]) -> Dict[str, Any]:
    """
    Execute simulation with optimal parameters and return results with fork number.
    
    Args:
        optimal_params: Dictionary of optimal parameter values
        
    Returns:
        dict: Simulation results with fork number
    """
    try:
        # Import batch simulation function
        from .doe_design import run_doe_batch_simulation
        import pandas as pd
        
        # Create design matrix with single row for optimal parameters
        design_matrix = pd.DataFrame([optimal_params])
        # Add required 'Run' column (matching DoE design matrix format)
        design_matrix.insert(0, 'Run', [1])
        
        # Get base parameters from session state
        if 'base_parameters' not in st.session_state:
            return {
                'success': False,
                'message': 'Base parameters not found. Please run a simulation first.'
            }
        
        base_params = st.session_state.base_parameters.copy()
        
        # Run simulation
        results_df = run_doe_batch_simulation(design_matrix, base_params)
        
        if results_df is not None and len(results_df) > 0:
            result = results_df.iloc[0].to_dict()  # Convert first row to dict
            
            # The fork number is already calculated correctly by the batch system
            fork_number = result.get('Fork', 'Unknown')
            
            return {
                'success': True,
                'results': result,
                'fork_number': fork_number
            }
        else:
            return {
                'success': False,
                'message': 'Simulation returned no results'
            }
            
    except Exception as e:
        return {
            'success': False,
            'message': f'Simulation failed: {str(e)}'
        }

def create_optimization_plots(results_df: pd.DataFrame, selected_params: Dict[str, Any], selected_responses: Dict[str, Any]):
    """
    Create visualization plots for optimization results.
    
    Args:
        results_df: DataFrame with experimental results
        optimal_params: Dictionary of optimal parameter values
        selected_params: Dictionary of parameter metadata
        selected_responses: Dictionary of response configurations
    """
    if len(selected_params) < 2 or len(selected_responses) < 1:
        return
    
    st.markdown("#### 📊 Optimization Visualization")
    
    # Parameter correlation plot
    factor_names = list(selected_params.keys())
    response_names = list(selected_responses.keys())
    
    if len(factor_names) >= 2 and len(response_names) >= 1:
        # Create correlation matrix
        all_columns = factor_names + response_names
        subset_df = results_df[all_columns].dropna()
        
        if len(subset_df) > 1:
            corr_matrix = subset_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(3),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Parameter-Response Correlation Matrix",
                height=500,
                width=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
