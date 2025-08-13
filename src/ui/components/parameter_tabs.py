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
        dict: Dictionary containing growth parameters including mechanistic kinetics
    """
    st.sidebar.subheader("Cell Growth Parameters")
    
    with st.sidebar.expander("Growth & Death Kinetics", expanded=False):
        # Cell growth and death kinetics
        max_growth_rate = st.number_input(
            "Max Growth Rate (μ_max)", 
            value=0.04, 
            min_value=0.0,
            help="Maximum specific growth rate under optimal conditions (1/h)",
            key="max_growth_rate"
        )
        death_rate = st.number_input(
            "Death Rate (k_d)", 
            value=0.005, 
            min_value=0.0,
            help="Cell death rate constant (1/h)",
            key="death_rate"
        )
        max_viable_density = st.number_input(
            "Max Viable Density (X_max)", 
            value=20.0, 
            min_value=0.0,
            help="Maximum sustainable viable cell density (million cells/mL)",
            key="max_viable_density"
        )
    
    with st.sidebar.expander("Product Formation", expanded=False):
        # Product formation
        specific_productivity = st.number_input(
            "Specific Productivity (q_P)", 
            value=0.001, 
            min_value=0.0,
            help="Product formation rate per viable cell (g/cell/h)",
            key="specific_productivity"
        )
    
    # NEW: Mechanistic Glucose Kinetics - Primary section
    with st.sidebar.expander("🔬 Mechanistic Glucose Kinetics", expanded=True):
        st.markdown("**Condition-Dependent Glucose Consumption (Pirt Model)**")
        st.info("q_G = glucose_growth_coeff × μ + glucose_maintenance_rate")
        
        glucose_growth_coeff = st.number_input(
            "Growth-Associated Glucose (Y_G/X⁻¹)", 
            value=0.5, 
            min_value=0.0,
            help="Glucose needed per unit growth - scales with growth rate (g glucose/g cells)",
            key="glucose_growth_coeff"
        )
        glucose_maintenance_rate = st.number_input(
            "Maintenance Glucose (m_G)", 
            value=0.01, 
            min_value=0.0,
            help="Baseline glucose consumption for cell maintenance (g glucose per cell per hour)",
            key="glucose_maintenance_rate"
        )
        
        st.markdown("💡 **Benefit**: Glucose consumption automatically responds to culture conditions!")
    
    # NEW: Mechanistic Lactate Switching - Primary section  
    with st.sidebar.expander("🔄 Mechanistic Lactate Switching", expanded=True):
        st.markdown("**Metabolic Switching: Production ↔ Consumption**")
        st.info("High glucose → Lactate production | Low glucose → Lactate consumption")
        
        lactate_shift_glucose = st.number_input(
            "Glucose Shift Threshold (G_shift)", 
            value=2.5, 
            min_value=0.0,
            help="Glucose level where metabolism shifts between production and consumption (g/L)",
            key="lactate_shift_glucose"
        )
        lactate_switch_steepness = st.number_input(
            "Switch Steepness (k)", 
            value=2.0, 
            min_value=0.1,
            help="Sharpness of metabolic transition (higher = more abrupt switch)",
            key="lactate_switch_steepness"
        )
        lactate_consumption_max = st.number_input(
            "Max Lactate Consumption (q_L,max)", 
            value=0.02, 
            min_value=0.0,
            help="Maximum lactate consumption rate per cell (g lactate per cell per hour)",
            key="lactate_consumption_max"
        )
        lactate_half_saturation = st.number_input(
            "Lactate Half-Saturation (K_L)", 
            value=0.5, 
            min_value=0.0,
            help="Half-saturation constant for lactate consumption kinetics (g/L)",
            key="lactate_half_saturation"
        )
        
        st.markdown("💡 **Benefit**: Realistic CHO cell lactate metabolism with glycolytic shift!")
    
    # Simple glutamine kinetics (constant rate model)
    with st.sidebar.expander("Simple Glutamine Kinetics", expanded=False):
        glutamine_uptake_rate = st.number_input(
            "Glutamine Uptake Rate (q_Q)", 
            value=0.01, 
            min_value=0.0,
            help="Glutamine consumption rate per viable cell (g/cell/h) - simple constant rate",
            key="glutamine_uptake_rate"
        )
        st.markdown("ℹ️ *Uses simple constant rate - could be upgraded to mechanistic model*")
    
    return {
        "max_growth_rate": max_growth_rate,
        "death_rate": death_rate,
        "max_viable_density": max_viable_density,
        "specific_productivity": specific_productivity,
        "glutamine_uptake_rate": glutamine_uptake_rate,
        "glucose_growth_coeff": glucose_growth_coeff,
        "glucose_maintenance_rate": glucose_maintenance_rate,
        "lactate_shift_glucose": lactate_shift_glucose,
        "lactate_switch_steepness": lactate_switch_steepness,
        "lactate_consumption_max": lactate_consumption_max,
        "lactate_half_saturation": lactate_half_saturation
    }

# ==================== MEDIA PARAMETERS ====================

def render_media_parameters():
    """
    Render media parameter inputs for substrate kinetics and metabolic yields.
    
    This section defines the medium composition effects and metabolic
    stoichiometry that govern substrate utilization and metabolite production.
    These parameters control the efficiency of nutrient conversion.
    
    Returns:
        dict: Dictionary containing media parameters for Monod kinetics and yields
    """
    st.sidebar.subheader("Media & Substrate Parameters")
    
    with st.sidebar.expander("Substrate Saturation (Monod Kinetics)", expanded=False):
        # Substrate saturation kinetics (Monod model parameters)
        st.markdown("**Half-saturation constants for substrate limitation**")
        
        glucose_monod_const = st.number_input(
            "Glucose Monod Constant (K_G)", 
            value=0.5, 
            min_value=0.0,
            help="Half-saturation constant for glucose utilization in growth (g/L)",
            key="glucose_monod_const"
        )
        glutamine_monod_const = st.number_input(
            "Glutamine Monod Constant (K_Q)", 
            value=0.3, 
            min_value=0.0,
            help="Half-saturation constant for glutamine utilization in growth (g/L)",
            key="glutamine_monod_const"
        )
    
    with st.sidebar.expander("Stoichiometric Yields", expanded=False):
        # Stoichiometric coefficients for metabolite production
        st.markdown("**Simple stoichiometric yields**")
        
        lactate_yield_per_glucose = st.number_input(
            "Lactate Yield per Glucose (Y_L/G)", 
            value=0.9, 
            min_value=0.0,
            help="Mass of lactate produced per glucose in production phase - used in mechanistic model (g/g)",
            key="lactate_yield_per_glucose"
        )
        ammonia_yield_per_glutamine = st.number_input(
            "Ammonia Yield per Glutamine (Y_A/Q)", 
            value=1.2, 
            min_value=0.0,
            help="Mass of ammonia produced per mass of glutamine consumed (g/g)",
            key="ammonia_yield_per_glutamine"
        )
        
        st.markdown("ℹ️ *Lactate now uses mechanistic switching model, ammonia still uses simple yield*")
    
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
    how adverse conditions affect cell growth and viability using realistic
    asymmetric response models.
    
    Returns:
        dict: Dictionary containing environmental parameters with asymmetric models
    """
    st.sidebar.subheader("Environmental Parameters")
    
    with st.sidebar.expander("Oxygen & Metabolite Inhibition", expanded=False):
        # Oxygen availability and metabolite inhibition
        dissolved_oxygen_percent = st.number_input(
            "Dissolved Oxygen (%)", 
            value=100, 
            min_value=0,
            help="Dissolved oxygen saturation level (%) - affects growth rate",
            key="dissolved_oxygen_percent"
        )
        oxygen_monod_const = st.number_input(
            "Oxygen Monod Constant (K_DO)", 
            value=5.0, 
            min_value=0.0,
            help="Half-saturation constant for oxygen utilization in growth",
            key="oxygen_monod_const"
        )
        
        st.markdown("**Waste Product Inhibition**")
        lactate_inhibition_coeff = st.number_input(
            "Lactate Inhibition Coefficient (α_L)", 
            value=0.02, 
            min_value=0.0,
            help="Growth inhibition coefficient for lactate accumulation",
            key="lactate_inhibition_coeff"
        )
        ammonia_inhibition_coeff = st.number_input(
            "Ammonia Inhibition Coefficient (α_A)", 
            value=0.03, 
            min_value=0.0,
            help="Growth inhibition coefficient for ammonia accumulation",
            key="ammonia_inhibition_coeff"
        )
    
    with st.sidebar.expander("🌡️ Asymmetric Temperature Effects", expanded=False):
        # Temperature parameters with asymmetric sensitivity
        st.markdown("**Biologically Realistic Temperature Response**")
        st.info("Heat is more damaging than cold - matches real CHO cell behavior")
        
        optimal_temp = st.number_input(
            "Optimal Temperature (°C)", 
            value=37,
            help="Optimal temperature for maximum cell growth (°C)",
            key="optimal_temp"
        )
        temp_death_threshold = st.number_input(
            "Heat Death Threshold (°C)", 
            value=42.0, 
            min_value=0.0,
            help="Temperature above which complete growth arrest occurs (°C)",
            key="temp_death_threshold"
        )
        temp_heat_sensitivity = st.number_input(
            "Heat Stress Sensitivity (α_T,hot)", 
            value=0.8, 
            min_value=0.0,
            help="Sensitivity coefficient for heat stress (higher = more sensitive to heat)",
            key="temp_heat_sensitivity"
        )
        temp_cold_sensitivity = st.number_input(
            "Cold Stress Sensitivity (α_T,cold)", 
            value=0.3, 
            min_value=0.0,
            help="Sensitivity coefficient for cold stress (lower = less sensitive to cold)",
            key="temp_cold_sensitivity"
        )
    
    with st.sidebar.expander("🧪 Asymmetric pH Effects", expanded=False):
        # pH parameters with asymmetric sensitivity
        st.markdown("**Biologically Realistic pH Response**")
        st.info("Alkaline conditions are more damaging than acidic - matches real cell physiology")
        
        col1, col2 = st.columns(2)
        with col1:
            ph_death_min = st.number_input(
                "pH Death Minimum", 
                value=6.5,
                help="Minimum pH for cell survival",
                key="ph_death_min"
            )
            ph_optimal_min = st.number_input(
                "pH Optimal Minimum", 
                value=7.0,
                help="Lower bound of optimal pH range",
                key="ph_optimal_min"
            )
        with col2:
            ph_death_max = st.number_input(
                "pH Death Maximum", 
                value=8.0,
                help="Maximum pH for cell survival",
                key="ph_death_max"
            )
            ph_optimal_max = st.number_input(
                "pH Optimal Maximum", 
                value=7.4,
                help="Upper bound of optimal pH range",
                key="ph_optimal_max"
            )
            
        ph_acidic_sensitivity = st.number_input(
            "Acidic Stress Sensitivity (α_pH,acid)", 
            value=0.3, 
            min_value=0.0,
            help="Sensitivity coefficient for acidic stress (lower = less sensitive to low pH)",
            key="ph_acidic_sensitivity"
        )
        ph_alkaline_sensitivity = st.number_input(
            "Alkaline Stress Sensitivity (α_pH,alk)", 
            value=0.8, 
            min_value=0.0,
            help="Sensitivity coefficient for alkaline stress (higher = more sensitive to high pH)",
            key="ph_alkaline_sensitivity"
        )
    
    with st.sidebar.expander("Current Culture Conditions", expanded=True):
        # Current culture conditions
        st.markdown("**Set Current Operating Conditions**")
        
        col1, col2 = st.columns(2)
        with col1:
            culture_ph = st.number_input(
                "Culture pH", 
                value=6.5,
                help="Current pH of the culture medium",
                key="culture_ph"
            )
        with col2:
            culture_temp = st.number_input(
                "Culture Temp (°C)", 
                value=22,
                help="Current temperature of the culture (°C)",
                key="culture_temp"
            )
    
    # ===== MEASUREMENT SYSTEM PARAMETERS =====
    with st.expander("📊 Measurement System", expanded=False):
        st.markdown("**Sensor and measurement system parameters**")
        
        measurement_noise = st.slider(
            "Measurement Noise (%)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1,
            help="Percentage noise added to measurements in visualization and export to simulate real sensor variations. "
                 "0% = perfect measurements, 2% = good lab conditions, 5% = typical conditions, 10%+ = challenging conditions. "
                 "Note: This only affects display and export - simulation dynamics remain clean.",
            key="measurement_noise"
        ) / 100.0  # Convert percentage to decimal
    
    return {
        "dissolved_oxygen_percent": dissolved_oxygen_percent,
        "oxygen_monod_const": oxygen_monod_const,
        "lactate_inhibition_coeff": lactate_inhibition_coeff,
        "ammonia_inhibition_coeff": ammonia_inhibition_coeff,
        "temp_heat_sensitivity": temp_heat_sensitivity,
        "temp_cold_sensitivity": temp_cold_sensitivity,
        "temp_death_threshold": temp_death_threshold,
        "ph_alkaline_sensitivity": ph_alkaline_sensitivity,
        "ph_acidic_sensitivity": ph_acidic_sensitivity,
        "ph_death_min": ph_death_min,
        "ph_death_max": ph_death_max,
        "ph_optimal_min": ph_optimal_min,
        "ph_optimal_max": ph_optimal_max,
        "culture_ph": culture_ph,
        "culture_temp": culture_temp,
        "optimal_temp": optimal_temp,
        "measurement_noise": measurement_noise
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
