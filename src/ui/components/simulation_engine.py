"""
Simulation Engine Module

This module handles all simulation execution logic including initial runs
and fork creation. It manages BioreactorSimulation instances and state restoration.

Key Features:
- Initial simulation execution
- Fork simulation from specific time points
- State restoration and continuation
- Simulation parameter management
"""

import streamlit as st
import copy
import sys
from pathlib import Path

# Add parent directory to Python path to import bioreactor_sim
sys.path.append(str(Path(__file__).parent.parent.parent))
from bioreactor_sim import BioreactorSimulation


def execute_simulation(base_parameters):
    """
    Execute simulation based on current state (initial or fork).
    
    Args:
        base_parameters (dict): Current simulation parameters
    
    Returns:
        bool: True if simulation completed successfully, False otherwise
    """
    if not st.session_state.simulation_segments:
        return run_initial_simulation(base_parameters)
    else:
        return run_fork_simulation(base_parameters)


def run_initial_simulation(base_parameters):
    """
    Run the initial simulation with base parameters.
    
    Args:
        base_parameters (dict): Base simulation parameters
        
    Returns:
        bool: True if simulation completed successfully
    """
    with st.spinner("Running initial simulation..."):
        # Create new simulator instance with base parameters
        simulator = create_simulator(base_parameters)
        
        # Run complete simulation
        states = simulator.simulate_all(
            base_parameters['total_hours'],
            culture_temp=base_parameters['culture_temp'],
            optimal_temp=base_parameters['optimal_temp'],
            dissolved_oxygen_percent=base_parameters['dissolved_oxygen_percent'],
            perfusion_rate_high=base_parameters['perfusion_rate_high'],
            glucose_threshold=base_parameters['glucose_threshold']
        )
        
        # Store as first segment (Original)
        st.session_state.simulation_segments.append({
            'start_time': 0.0,
            'end_time': base_parameters['total_hours'],
            'states': states,
            'parameters': copy.deepcopy(base_parameters),
            'is_fork': False,
            'segment_id': 0
        })
        
    st.success("Initial simulation completed!")
    return True


def run_fork_simulation(base_parameters):
    """
    Run a fork simulation from the current time point with modified parameters.
    
    Args:
        base_parameters (dict): Modified simulation parameters for the fork
        
    Returns:
        bool: True if simulation completed successfully
    """
    with st.spinner(f"Running simulation from {st.session_state.current_time_point:.1f} hours..."):
        # Find the exact state at current time point from existing segments
        current_state = find_state_at_time(st.session_state.current_time_point)
        
        if current_state:
            # Create new simulator with modified parameters
            simulator = create_simulator(base_parameters)
            
            # Restore simulator state to current time point
            restore_simulator_state(simulator, current_state)
            
            # Run simulation from current point to completion
            remaining_time = base_parameters['total_hours'] - st.session_state.current_time_point
            if remaining_time > 0:
                states = simulator.simulate_all(
                    remaining_time,
                    culture_temp=base_parameters['culture_temp'],
                    optimal_temp=base_parameters['optimal_temp'],
                    dissolved_oxygen_percent=base_parameters['dissolved_oxygen_percent'],
                    perfusion_rate_high=base_parameters['perfusion_rate_high'],
                    glucose_threshold=base_parameters['glucose_threshold']
                )
                
                # Add as new fork segment
                next_id = max([seg['segment_id'] for seg in st.session_state.simulation_segments]) + 1
                st.session_state.simulation_segments.append({
                    'start_time': st.session_state.current_time_point,
                    'end_time': base_parameters['total_hours'],
                    'states': states,
                    'parameters': copy.deepcopy(base_parameters),
                    'is_fork': True,
                    'segment_id': next_id
                })
                
                st.success(f"Fork simulation completed from {st.session_state.current_time_point:.1f} hours!")
                return True
        
        st.error("Could not find state at current time point")
        return False


def create_simulator(parameters):
    """
    Create a BioreactorSimulation instance with given parameters.
    
    Args:
        parameters (dict): Simulation parameters
        
    Returns:
        BioreactorSimulation: Configured simulator instance
    """
    return BioreactorSimulation(
        time_step=parameters['time_step'],
        perfusion_rate_base=parameters['perfusion_rate_base'],
        max_growth_rate=parameters['max_growth_rate'],
        death_rate=parameters['death_rate'],
        glutamine_uptake_rate=parameters['glutamine_uptake_rate'],
        specific_productivity=parameters['specific_productivity'],
        lactate_yield_per_glucose=parameters['lactate_yield_per_glucose'],
        ammonia_yield_per_glutamine=parameters['ammonia_yield_per_glutamine'],
        glucose_monod_const=parameters['glucose_monod_const'],
        glutamine_monod_const=parameters['glutamine_monod_const'],
        oxygen_monod_const=parameters['oxygen_monod_const'],
        max_viable_density=parameters['max_viable_density'],
        lactate_inhibition_coeff=parameters['lactate_inhibition_coeff'],
        ammonia_inhibition_coeff=parameters['ammonia_inhibition_coeff'],
        temp_heat_sensitivity=parameters['temp_heat_sensitivity'],
        temp_cold_sensitivity=parameters['temp_cold_sensitivity'],
        temp_death_threshold=parameters['temp_death_threshold'],
        ph_alkaline_sensitivity=parameters['ph_alkaline_sensitivity'],
        ph_acidic_sensitivity=parameters['ph_acidic_sensitivity'],
        ph_death_min=parameters['ph_death_min'],
        ph_death_max=parameters['ph_death_max'],
        ph_optimal_min=parameters['ph_optimal_min'],
        ph_optimal_max=parameters['ph_optimal_max'],
        culture_ph=parameters['culture_ph'],
        glucose_feed_conc=parameters['glucose_feed_conc'],
        glutamine_feed_conc=parameters['glutamine_feed_conc'],
        measurement_noise=parameters['measurement_noise'],
        glucose_growth_coeff=parameters['glucose_growth_coeff'],
        glucose_maintenance_rate=parameters['glucose_maintenance_rate'],
        lactate_shift_glucose=parameters['lactate_shift_glucose'],
        lactate_switch_steepness=parameters['lactate_switch_steepness'],
        lactate_consumption_max=parameters['lactate_consumption_max'],
        lactate_half_saturation=parameters['lactate_half_saturation']
    )


def find_state_at_time(target_time):
    """
    Find the simulation state at a specific time point.
    
    Args:
        target_time (float): Target time in hours
        
    Returns:
        dict: State dictionary at target time, or None if not found
    """
    for segment in st.session_state.simulation_segments:
        for state in segment['states']:
            if abs(state['time'] - target_time) < 0.01:
                return state
    return None


def restore_simulator_state(simulator, state):
    """
    Restore a simulator to a specific state.
    
    Args:
        simulator (BioreactorSimulation): Simulator instance to restore
        state (dict): State dictionary containing all variable values
    """
    simulator.time = state['time']
    simulator.viable_cell_density = state['viable_cell_density']
    simulator.dead_cell_density = state['dead_cell_density']
    simulator.glucose_concentration = state['glucose_concentration']
    simulator.glutamine_concentration = state['glutamine_concentration']
    simulator.lactate_concentration = state['lactate_concentration']
    simulator.ammonia_concentration = state['ammonia_concentration']
    simulator.product_concentration = state['product_concentration']
    simulator.aggregated_product = state['aggregated_product']
