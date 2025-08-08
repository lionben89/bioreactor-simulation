"""
Parameter Input Components for Bioreactor Simulation

This module provides a comprehensive parameter input interface organized into
logical sections within the Streamlit sidebar. It handles all simulation
parameters including initial conditions, operation settings, biological
parameters, and environmental conditions.

Organization:
- Initial Parameters: Feed concentrations and initial conditions
- Operation Parameters: Time settings and perfusion control
- Cell Growth Parameters: Growth kinetics and metabolic rates  
- Media Parameters: Substrate constants and metabolic yields
- Environment Parameters: Culture conditions and inhibition factors

Features:
- Organized in expandable sections for better UX
- Input validation with minimum value constraints
- Consistent parameter naming and units
- Real-time parameter updates for interactive analysis
"""

import streamlit as st

# ==================== INITIAL CONDITIONS ====================

def render_initial_parameters():
    """
    Render initial parameter inputs for feed concentrations and starting conditions.
    
    This section contains the fundamental parameters that define the initial
    state of the bioreactor, including nutrient feed concentrations that
    are supplied during perfusion operations.
    
    Returns:
        dict: Dictionary containing initial parameter values with keys:
              - glucose_feed_conc: Glucose concentration in feed medium (g/L)
              - glutamine_feed_conc: Glutamine concentration in feed medium (g/L)
    """
    st.sidebar.subheader("Initial Parameters")
    
    with st.sidebar.expander("Feed Concentrations", expanded=True):
        # Feed medium concentrations (supplied during perfusion)
        glucose_feed_conc = st.number_input(
            "Glucose Feed Concentration (Gin)", 
            value=8.0, 
            min_value=0.0, 
            help="Glucose concentration in the feed medium (g/L)",
            key="glucose_feed_conc"
        )
        glutamine_feed_conc = st.number_input(
            "Glutamine Feed Concentration (Qin)", 
            value=5.0, 
            min_value=0.0,
            help="Glutamine concentration in the feed medium (g/L)",
            key="glutamine_feed_conc"
        )
    
    return {
        "glucose_feed_conc": glucose_feed_conc,
        "glutamine_feed_conc": glutamine_feed_conc
    }

# ==================== OPERATION PARAMETERS ====================

def render_operation_parameters():
    """
    Render operation parameter inputs for time settings and perfusion control.
    
    This section defines the operational parameters that control how the
    bioreactor runs, including simulation duration, time resolution, and
    the perfusion system that maintains nutrient levels.
    
    Returns:
        dict: Dictionary containing operation parameters:
              - total_hours: Total simulation duration (hours)
              - time_step: Simulation time resolution (hours)
              - perfusion_rate_high: High perfusion rate when pump is active
              - perfusion_rate_base: Base perfusion rate (continuous)
              - glucose_threshold: Glucose level that triggers pump activation
    """
    st.sidebar.subheader("Operation Parameters")
    
    with st.sidebar.expander("Time Settings", expanded=True):
        # Simulation time configuration
        total_hours = st.number_input(
            "Total Hours", 
            value=500, 
            min_value=1,
            help="Total duration of the simulation (hours)",
            key="total_hours"
        )
        time_step = st.number_input(
            "Time Step", 
            value=0.1, 
            min_value=0.01, 
            step=0.01,
            help="Time resolution for simulation calculations (hours)",
            key="time_step"
        )
    
    with st.sidebar.expander("Perfusion Settings", expanded=True):
        # Perfusion control system parameters
        perfusion_rate_high = st.number_input(
            "Perfusion Rate High (Dpump)", 
            value=0.12, 
            min_value=0.0,
            help="High perfusion rate when pump is actively feeding (1/h)",
            key="perfusion_rate_high"
        )
        perfusion_rate_base = st.number_input(
            "Perfusion Rate Base (Dbase)", 
            value=0.001, 
            min_value=0.0,
            help="Continuous base perfusion rate (1/h)",
            key="perfusion_rate_base"
        )
        glucose_threshold = st.number_input(
            "Glucose Threshold (Gmin)", 
            value=4.0, 
            min_value=0.0,
            help="Glucose concentration that triggers pump activation (g/L)",
            key="glucose_threshold"
        )
    
    return {
        "total_hours": total_hours,
        "time_step": time_step,
        "perfusion_rate_high": perfusion_rate_high,
        "perfusion_rate_base": perfusion_rate_base,
        "glucose_threshold": glucose_threshold
    }

# ==================== CELL GROWTH PARAMETERS ====================

def render_growth_parameters():
    """
    Render cell growth parameter inputs for biological kinetics and metabolism.
    
    This section contains the fundamental biological parameters that govern
    cell growth, death, and metabolic activity. These parameters define the
    intrinsic behavior of the CHO cell culture.
    
    Returns:
        dict: Dictionary containing growth parameters:
              - max_growth_rate: Maximum specific growth rate (1/h)
              - death_rate: Cell death rate constant (1/h)
              - max_viable_density: Maximum sustainable cell density (million cells/mL)
              - specific_productivity: Product formation rate per cell (g/cell/h)
              - glucose_uptake_rate: Glucose consumption rate per cell (g/cell/h)
              - glutamine_uptake_rate: Glutamine consumption rate per cell (g/cell/h)
    """
    st.sidebar.subheader("Cell Growth Parameters")
    
    with st.sidebar.expander("Growth Kinetics", expanded=False):
        # Cell growth and death kinetics
        max_growth_rate = st.number_input(
            "Max Growth Rate (mu_max)", 
            value=0.04, 
            min_value=0.0,
            help="Maximum specific growth rate under optimal conditions (1/h)",
            key="max_growth_rate"
        )
        death_rate = st.number_input(
            "Death Rate (kd)", 
            value=0.005, 
            min_value=0.0,
            help="Cell death rate constant (1/h)",
            key="death_rate"
        )
        max_viable_density = st.number_input(
            "Max Viable Density (million cells/mL)", 
            value=20.0, 
            min_value=0.0,
            help="Maximum sustainable viable cell density (million cells/mL)",
            key="max_viable_density"
        )
    
    with st.sidebar.expander("Metabolic Rates", expanded=False):
        # Cellular metabolic activity rates
        specific_productivity = st.number_input(
            "Specific Productivity (qP)", 
            value=0.001, 
            min_value=0.0,
            help="Product formation rate per viable cell (g/cell/h)",
            key="specific_productivity"
        )
        glucose_uptake_rate = st.number_input(
            "Glucose Uptake Rate (qG)", 
            value=0.03, 
            min_value=0.0,
            help="Glucose consumption rate per viable cell (g/cell/h)",
            key="glucose_uptake_rate"
        )
        glutamine_uptake_rate = st.number_input(
            "Glutamine Uptake Rate (qQ)", 
            value=0.01, 
            min_value=0.0,
            help="Glutamine consumption rate per viable cell (g/cell/h)",
            key="glutamine_uptake_rate"
        )
    
    return {
        "max_growth_rate": max_growth_rate,
        "death_rate": death_rate,
        "max_viable_density": max_viable_density,
        "specific_productivity": specific_productivity,
        "glucose_uptake_rate": glucose_uptake_rate,
        "glutamine_uptake_rate": glutamine_uptake_rate
    }

# ==================== MEDIA PARAMETERS ====================

def render_media_parameters():
    """
    Render media parameter inputs for substrate kinetics and metabolic yields.
    
    This section defines the medium composition effects and metabolic
    stoichiometry that govern substrate utilization and metabolite production.
    These parameters control the efficiency of nutrient conversion.
    
    Returns:
        dict: Dictionary containing media parameters:
              - glucose_monod_const: Half-saturation constant for glucose (g/L)
              - glutamine_monod_const: Half-saturation constant for glutamine (g/L)
              - lactate_yield_per_glucose: Lactate produced per glucose consumed (g/g)
              - ammonia_yield_per_glutamine: Ammonia produced per glutamine consumed (g/g)
    """
    st.sidebar.subheader("Media Parameters")
    
    with st.sidebar.expander("Monod Constants", expanded=False):
        # Substrate saturation kinetics (Monod model parameters)
        glucose_monod_const = st.number_input(
            "Glucose Monod Constant (KG)", 
            value=0.5, 
            min_value=0.0,
            help="Half-saturation constant for glucose uptake (g/L)",
            key="glucose_monod_const"
        )
        glutamine_monod_const = st.number_input(
            "Glutamine Monod Constant (KQ)", 
            value=0.3, 
            min_value=0.0,
            help="Half-saturation constant for glutamine uptake (g/L)",
            key="glutamine_monod_const"
        )
    
    with st.sidebar.expander("Metabolic Yields", expanded=False):
        # Stoichiometric coefficients for metabolite production
        lactate_yield_per_glucose = st.number_input(
            "Lactate Yield per Glucose (YL_G)", 
            value=0.9, 
            min_value=0.0,
            help="Mass of lactate produced per mass of glucose consumed (g/g)",
            key="lactate_yield_per_glucose"
        )
        ammonia_yield_per_glutamine = st.number_input(
            "Ammonia Yield per Glutamine (YA_Q)", 
            value=1.2, 
            min_value=0.0,
            help="Mass of ammonia produced per mass of glutamine consumed (g/g)",
            key="ammonia_yield_per_glutamine"
        )
    
    return {
        "glucose_monod_const": glucose_monod_const,
        "glutamine_monod_const": glutamine_monod_const,
        "lactate_yield_per_glucose": lactate_yield_per_glucose,
        "ammonia_yield_per_glutamine": ammonia_yield_per_glutamine
    }

# ==================== ENVIRONMENTAL PARAMETERS ====================

def render_environment_parameters():
    """
    Render environmental parameter inputs for culture conditions and inhibition effects.
    
    This section contains parameters that define the culture environment and
    how adverse conditions (metabolite accumulation, suboptimal pH/temperature)
    affect cell growth and viability.
    
    Returns:
        dict: Dictionary containing environmental parameters:
              - dissolved_oxygen_percent: Oxygen saturation level (%)
              - oxygen_monod_const: Half-saturation constant for oxygen
              - lactate_inhibition_coeff: Growth inhibition coefficient for lactate
              - ammonia_inhibition_coeff: Growth inhibition coefficient for ammonia
              - ph_inhibition_coeff: Growth inhibition coefficient for pH deviation
              - temp_inhibition_coeff: Growth inhibition coefficient for temperature deviation
              - culture_ph: Current culture pH
              - optimal_ph: Optimal pH for growth
              - culture_temp: Current culture temperature (°C)
              - optimal_temp: Optimal temperature for growth (°C)
    """
    st.sidebar.subheader("Environmental Parameters")
    
    with st.sidebar.expander("Oxygen & Inhibition", expanded=False):
        # Oxygen availability and metabolite inhibition
        dissolved_oxygen_percent = st.number_input(
            "Dissolved Oxygen (%)", 
            value=100, 
            min_value=0,
            help="Dissolved oxygen saturation level (%)",
            key="dissolved_oxygen_percent"
        )
        oxygen_monod_const = st.number_input(
            "Oxygen Monod Constant (KDO)", 
            value=5.0, 
            min_value=0.0,
            help="Half-saturation constant for oxygen utilization",
            key="oxygen_monod_const"
        )
        lactate_inhibition_coeff = st.number_input(
            "Lactate Inhibition Coeff", 
            value=0.02, 
            min_value=0.0,
            help="Growth inhibition coefficient for lactate accumulation",
            key="lactate_inhibition_coeff"
        )
        ammonia_inhibition_coeff = st.number_input(
            "Ammonia Inhibition Coeff", 
            value=0.03, 
            min_value=0.0,
            help="Growth inhibition coefficient for ammonia accumulation",
            key="ammonia_inhibition_coeff"
        )
    
    with st.sidebar.expander("Culture Conditions", expanded=False):
        # pH and temperature conditions
        ph_inhibition_coeff = st.number_input(
            "pH Inhibition Coeff", 
            value=0.05, 
            min_value=0.0,
            help="Growth inhibition coefficient for pH deviation from optimum",
            key="ph_inhibition_coeff"
        )
        temp_inhibition_coeff = st.number_input(
            "Temp Inhibition Coeff", 
            value=0.04, 
            min_value=0.0,
            help="Growth inhibition coefficient for temperature deviation from optimum",
            key="temp_inhibition_coeff"
        )
        culture_ph = st.number_input(
            "Culture pH", 
            value=6.5,
            help="Current pH of the culture medium",
            key="culture_ph"
        )
        optimal_ph = st.number_input(
            "Optimal pH", 
            value=7.2,
            help="Optimal pH for maximum cell growth",
            key="optimal_ph"
        )
        culture_temp = st.number_input(
            "Culture Temp (°C)", 
            value=22,
            help="Current temperature of the culture (°C)",
            key="culture_temp"
        )
        optimal_temp = st.number_input(
            "Optimal Temp (°C)", 
            value=37,
            help="Optimal temperature for maximum cell growth (°C)",
            key="optimal_temp"
        )
    
    return {
        "dissolved_oxygen_percent": dissolved_oxygen_percent,
        "oxygen_monod_const": oxygen_monod_const,
        "lactate_inhibition_coeff": lactate_inhibition_coeff,
        "ammonia_inhibition_coeff": ammonia_inhibition_coeff,
        "ph_inhibition_coeff": ph_inhibition_coeff,
        "temp_inhibition_coeff": temp_inhibition_coeff,
        "culture_ph": culture_ph,
        "optimal_ph": optimal_ph,
        "culture_temp": culture_temp,
        "optimal_temp": optimal_temp
    }

# ==================== MAIN PARAMETER INTERFACE ====================

def render_parameter_tabs():
    """
    Render the complete parameter input interface in the sidebar.
    
    This function orchestrates all parameter sections and combines their
    outputs into a single parameter dictionary for the simulation engine.
    It provides the main interface for parameter input and modification.
    
    Returns:
        dict: Complete parameter dictionary containing all simulation parameters
              organized by functional groups (initial, operation, growth, media, environment)
              
    Usage:
        This function is called by the main application to generate the sidebar
        parameter interface and collect all user inputs for simulation configuration.
    """
    # Initialize parameter dictionary
    parameters = {}
    
    # Collect parameters from all sections
    parameters.update(render_initial_parameters())
    parameters.update(render_operation_parameters())
    parameters.update(render_growth_parameters())
    parameters.update(render_media_parameters())
    parameters.update(render_environment_parameters())
    
    return parameters
