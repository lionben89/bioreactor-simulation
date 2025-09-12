# CHO Cell Perfusion Bioreactor Simulation with Simple Kinetics



import numpy as np

class BioreactorSimulation:
    """
    A class to simulate a CHO Cell Perfusion Bioreactor with simple kinetics.
    
    This simulation models the growth of Chinese Hamster Ovary (CHO) cells in a perfusion bioreactor,
    including cell growth, death, substrate consumption, metabolite production, and environmental effects.
    
    The model includes:
    - Cell growth and death kinetics with Monod substrate limitation
    - Simple glucose and glutamine consumption with constant uptake rates
    - Simple lactate production with constant production rate
    - Ammonia production from glutamine consumption
    - Product formation
    - Environmental inhibition effects (pH, temperature) with asymmetric responses
    - Density-dependent growth limitation
    - Perfusion control based on glucose concentration
    
    Key Features:
    - Simple substrate uptake: constant rates per cell (glucose_uptake_rate, glutamine_uptake_rate)
    - Simple lactate production: calculated from glucose consumption using yield coefficient
    - Asymmetric environmental responses: heat/alkaline conditions more damaging than cold/acidic
    - Perfusion-based feeding with glucose threshold control
    
    Attributes:
        time_step (float): Simulation time step (hours)
        time (float): Current simulation time
        viable_cell_density (float): Concentration of viable cells (cells/mL)
        dead_cell_density (float): Concentration of dead cells (cells/mL)
        glucose_concentration (float): Concentration of glucose (g/L)
        glutamine_concentration (float): Concentration of glutamine (g/L)
        lactate_concentration (float): Concentration of lactate (g/L)
        ammonia_concentration (float): Concentration of ammonia (g/L)
        product_concentration (float): Concentration of product (g/L)
        aggregated_product (float): Total product produced including removed product (g/L)
        pump_active (int): Perfusion pump state (0 or 1)
    """
    
    def __init__(self,
        time_step=0.1,
        perfusion_rate_base=0.001,
        max_growth_rate=0.04,
        death_rate=0.005,
        glucose_uptake_rate=0.03,
        glutamine_uptake_rate=0.01,
        specific_productivity=0.001,
        lactate_yield_per_glucose=0.9,
        ammonia_yield_per_glutamine=1.2,
        glucose_monod_const=0.5,
        glutamine_monod_const=0.3,
        oxygen_monod_const=5,
        max_viable_density=20,
        lactate_inhibition_coeff=0.02,
        ammonia_inhibition_coeff=0.03,
        temp_heat_sensitivity=0.8,
        temp_cold_sensitivity=0.3,
        temp_death_threshold=42.0,
        ph_alkaline_sensitivity=0.8,
        ph_acidic_sensitivity=0.3,
        ph_death_min=6.5,
        ph_death_max=8.0,
        ph_optimal_min=7.0,
        ph_optimal_max=7.4,
        culture_ph=6.5,
        glucose_feed_conc=8.0,
        glutamine_feed_conc=5.0,
        measurement_noise=0.0
    ):
        """
        Initialize a new BioreactorSimulation instance.
        
        Args:
            time_step (float): Simulation time step in hours (default: 0.1)
            perfusion_rate_base (float): Base perfusion rate (1/h) when glucose is above threshold (default: 0.001)
            max_growth_rate (float): Maximum specific growth rate (1/h) under optimal conditions (default: 0.04)
            death_rate (float): Specific death rate (1/h) (default: 0.005)
            glucose_uptake_rate (float): Glucose consumption rate per viable cell (g/cell/h) (default: 0.03)
            glutamine_uptake_rate (float): Glutamine consumption rate per viable cell (g/cell/h) (default: 0.01)
            specific_productivity (float): Product formation rate per viable cell (g/cell/h) (default: 0.001)
            lactate_yield_per_glucose (float): Lactate yield coefficient (g lactate/g glucose) - legacy parameter (default: 0.9)
            ammonia_yield_per_glutamine (float): Ammonia yield coefficient (g ammonia/g glutamine) (default: 1.2)
            glucose_monod_const (float): Glucose half-saturation constant for growth (g/L) (default: 0.5)
            glutamine_monod_const (float): Glutamine half-saturation constant for growth (g/L) (default: 0.3)
            oxygen_monod_const (float): Oxygen half-saturation constant for growth (%) (default: 5)
            max_viable_density (float): Maximum viable cell density for density limitation (cells/mL × 10⁶) (default: 20)
            lactate_inhibition_coeff (float): Lactate inhibition coefficient (L/g) (default: 0.02)
            ammonia_inhibition_coeff (float): Ammonia inhibition coefficient (L/g) (default: 0.03)
            temp_heat_sensitivity (float): Temperature sensitivity coefficient for heat stress (1/°C) (default: 0.8)
            temp_cold_sensitivity (float): Temperature sensitivity coefficient for cold stress (1/°C) (default: 0.3)
            temp_death_threshold (float): Temperature threshold for cell death (°C) (default: 42.0)
            ph_alkaline_sensitivity (float): pH sensitivity coefficient for alkaline stress (1/pH unit) (default: 0.8)
            ph_acidic_sensitivity (float): pH sensitivity coefficient for acidic stress (1/pH unit) (default: 0.3)
            ph_death_min (float): Minimum pH for cell survival (default: 6.5)
            ph_death_max (float): Maximum pH for cell survival (default: 8.0)
            ph_optimal_min (float): Lower bound of optimal pH range (default: 7.0)
            ph_optimal_max (float): Upper bound of optimal pH range (default: 7.4)
            culture_ph (float): Current culture pH (default: 6.5)
            glucose_feed_conc (float): Glucose concentration in feed medium (g/L) (default: 8.0)
            glutamine_feed_conc (float): Glutamine concentration in feed medium (g/L) (default: 5.0)
            measurement_noise (float): Measurement noise level (0-1) for realistic sensor simulation (default: 0.0)
        """
        self.time_step = time_step
        self.perfusion_rate_base = perfusion_rate_base
        self.max_growth_rate = max_growth_rate
        self.death_rate = death_rate
        self.glucose_uptake_rate = glucose_uptake_rate
        self.glutamine_uptake_rate = glutamine_uptake_rate
        self.specific_productivity = specific_productivity
        self.lactate_yield_per_glucose = lactate_yield_per_glucose
        self.ammonia_yield_per_glutamine = ammonia_yield_per_glutamine
        self.glucose_monod_const = glucose_monod_const
        self.glutamine_monod_const = glutamine_monod_const
        self.oxygen_monod_const = oxygen_monod_const
        self.max_viable_density = max_viable_density
        self.lactate_inhibition_coeff = lactate_inhibition_coeff
        self.ammonia_inhibition_coeff = ammonia_inhibition_coeff
        self.temp_heat_sensitivity = temp_heat_sensitivity
        self.temp_cold_sensitivity = temp_cold_sensitivity
        self.temp_death_threshold = temp_death_threshold
        self.ph_alkaline_sensitivity = ph_alkaline_sensitivity
        self.ph_acidic_sensitivity = ph_acidic_sensitivity
        self.ph_death_min = ph_death_min
        self.ph_death_max = ph_death_max
        self.ph_optimal_min = ph_optimal_min
        self.ph_optimal_max = ph_optimal_max
        self.culture_ph = culture_ph
        self.glucose_feed_conc = glucose_feed_conc
        self.glutamine_feed_conc = glutamine_feed_conc
        self.measurement_noise = measurement_noise
        
        # Initialize state variables
        self.time = 0
        self.viable_cell_density = 0.5  # Initial value
        self.dead_cell_density = 0.0
        self.glucose_concentration = self.glucose_feed_conc
        self.glutamine_concentration = self.glutamine_feed_conc
        self.lactate_concentration = 0.0
        self.ammonia_concentration = 0.0
        self.product_concentration = 0.0
        self.aggregated_product = 0.0  # Total product produced (removed + remaining)
        self.pump_active = 0

    def get_state(self):
        """
        Get the current state of the bioreactor simulation.
        
        Returns:
            dict: A dictionary containing all current state variables including:
                - time: Current simulation time
                - viable_cell_density: Current viable cell concentration
                - dead_cell_density: Current dead cell concentration
                - glucose_concentration: Current glucose concentration
                - glutamine_concentration: Current glutamine concentration
                - lactate_concentration: Current lactate concentration
                - ammonia_concentration: Current ammonia concentration
                - product_concentration: Current product concentration
                - pump_active: Current pump state (0 or 1)
        """
        return {
            "time": self.time,
            "viable_cell_density": self.viable_cell_density,
            "dead_cell_density": self.dead_cell_density,
            "glucose_concentration": self.glucose_concentration,
            "glutamine_concentration": self.glutamine_concentration,
            "lactate_concentration": self.lactate_concentration,
            "ammonia_concentration": self.ammonia_concentration,
            "product_concentration": self.product_concentration,
            "aggregated_product": self.aggregated_product,
            "pump_active": self.pump_active
        }

    def get_state_with_noise(self):
        """
        Get the current state with measurement noise applied for visualization/export.
        
        This method applies Gaussian noise to simulate real sensor measurements
        while preserving the clean simulation data for dynamics calculations.
        
        Returns:
            dict: State dictionary with measurement noise applied to all variables except time and pump_active
        """
        state = self.get_state()
        
        if self.measurement_noise <= 0:
            return state
        
        # Create a copy to avoid modifying the original state
        noisy_state = state.copy()
        
        # Variables that should have measurement noise applied
        noisy_variables = [
            'viable_cell_density', 'dead_cell_density', 'glucose_concentration',
            'glutamine_concentration', 'lactate_concentration', 'ammonia_concentration',
            'product_concentration', 'aggregated_product'
        ]
        
        # Initialize random generator
        rng = np.random.default_rng(None)  # None allows random seeding for realistic noise variation
        
        # Apply Gaussian noise to each measurement variable
        for var in noisy_variables:
            if var in noisy_state:
                # Generate noise factor (1 + gaussian noise with std = measurement_noise)
                noise_factor = 1 + rng.normal(0, self.measurement_noise)
                # Apply noise and ensure non-negative values
                noisy_state[var] = max(0, noisy_state[var] * noise_factor)
        
        return noisy_state

    def simulate_step(self, culture_temp, optimal_temp, dissolved_oxygen_percent,
                     perfusion_rate_high, glucose_threshold):
        """
        Perform a single time step of the bioreactor simulation.

        Args:
            culture_temp (float): Current culture temperature
            optimal_temp (float): Optimal culture temperature
            dissolved_oxygen_percent (float): Dissolved oxygen concentration as percentage
            perfusion_rate_high (float): High perfusion rate when glucose is below threshold
            glucose_threshold (float): Glucose concentration threshold for perfusion control

        Returns:
            dict: Current state of the bioreactor after the time step
        """
        # Calculate dilution rate based on glucose level
        dilution_rate = perfusion_rate_high if self.glucose_concentration < glucose_threshold else self.perfusion_rate_base
        
        # Calculate density inhibition
        density_inhibition = max(0, 1 - self.viable_cell_density / self.max_viable_density)
        
        # Calculate growth rate with Monod kinetics and environmental inhibition
        specific_growth_rate = self.max_growth_rate \
            * (self.glucose_concentration / (self.glucose_concentration + self.glucose_monod_const)) \
            * (self.glutamine_concentration / (self.glutamine_concentration + self.glutamine_monod_const)) \
            * (dissolved_oxygen_percent / (dissolved_oxygen_percent + self.oxygen_monod_const)) \
            * density_inhibition
        
        # Apply environmental inhibition factors with asymmetric temperature response
        # Calculate temperature factor with asymmetric response (heat more damaging than cold)
        if culture_temp > self.temp_death_threshold:  # Heat death threshold for CHO cells
            temp_factor = 0.0  # Complete growth arrest
        elif culture_temp > optimal_temp:  # Too hot (more sensitive)
            temp_deviation = culture_temp - optimal_temp
            temp_factor = np.exp(-self.temp_heat_sensitivity * temp_deviation)  # Configurable heat stress
        else:  # Too cold (less sensitive)  
            temp_deviation = optimal_temp - culture_temp
            temp_factor = np.exp(-self.temp_cold_sensitivity * temp_deviation)  # Configurable cold stress
        
        # Calculate pH factor with asymmetric response (alkaline more damaging than acidic)
        if self.culture_ph < self.ph_death_min or self.culture_ph > self.ph_death_max:  # pH death boundaries
            ph_factor = 0.0  # Complete growth arrest
        elif self.ph_optimal_min <= self.culture_ph <= self.ph_optimal_max:  # Optimal pH range
            ph_factor = 1.0  # No inhibition in optimal range
        elif self.culture_ph < self.ph_optimal_min:  # Acidic side (less sensitive)
            ph_deviation = self.ph_optimal_min - self.culture_ph
            ph_factor = np.exp(-self.ph_acidic_sensitivity * ph_deviation)  # Configurable acidic stress
        else:  # Alkaline side (more sensitive)
            ph_deviation = self.culture_ph - self.ph_optimal_max
            ph_factor = np.exp(-self.ph_alkaline_sensitivity * ph_deviation)  # Configurable alkaline stress
        
        # Apply all inhibition factors
        specific_growth_rate *= np.exp(
            -self.lactate_inhibition_coeff * self.lactate_concentration
            - self.ammonia_inhibition_coeff * self.ammonia_concentration
        ) * temp_factor * ph_factor  # Apply asymmetric temperature and pH factors separately

        # Update state variables
        self.viable_cell_density += self.time_step * (
            specific_growth_rate * self.viable_cell_density - self.death_rate * self.viable_cell_density
        )
        
        self.dead_cell_density += self.time_step * (
            self.death_rate * self.viable_cell_density - dilution_rate * self.dead_cell_density
        )
        
        self.glucose_concentration += self.time_step * (
            -self.glucose_uptake_rate * self.viable_cell_density + dilution_rate * (self.glucose_feed_conc - self.glucose_concentration)
        )
        
        self.glutamine_concentration += self.time_step * (
            -self.glutamine_uptake_rate * self.viable_cell_density + dilution_rate * (self.glutamine_feed_conc - self.glutamine_concentration)
        )
        
        self.lactate_concentration += self.time_step * (
            self.lactate_yield_per_glucose * self.glucose_uptake_rate * self.viable_cell_density - dilution_rate * self.lactate_concentration
        )
        
        # Ensure non-negative concentrations
        self.glucose_concentration = max(0.0, self.glucose_concentration)
        self.lactate_concentration = max(0.0, self.lactate_concentration)
        
        self.ammonia_concentration += self.time_step * (
            self.ammonia_yield_per_glutamine * self.glutamine_uptake_rate * self.viable_cell_density - dilution_rate * self.ammonia_concentration
        )
        
        # Calculate product removed by dilution this step
        product_removed_this_step = dilution_rate * self.product_concentration * self.time_step
        
        # Update aggregated product (accumulate removed product)
        self.aggregated_product += product_removed_this_step
        
        self.product_concentration += self.time_step * (
            self.specific_productivity * self.viable_cell_density - dilution_rate * self.product_concentration
        )
        
        self.pump_active = 1 if self.glucose_concentration < glucose_threshold else 0
        self.time += self.time_step

        return self.get_state()

    def simulate_all(self, total_hours, culture_temp, optimal_temp, dissolved_oxygen_percent,
                    perfusion_rate_high, glucose_threshold):
        """
        Simulate the bioreactor for a specified number of hours.

        Args:
            total_hours (float): Total simulation time in hours
            culture_temp (float): Culture temperature
            optimal_temp (float): Optimal temperature
            dissolved_oxygen_percent (float): Dissolved oxygen concentration as percentage
            perfusion_rate_high (float): High perfusion rate when glucose is below threshold
            glucose_threshold (float): Glucose concentration threshold for perfusion control

        Returns:
            list[dict]: List of states at each time step, where each state is a dictionary containing
                       all the bioreactor variables at that time point
        """
        num_steps = int(total_hours / self.time_step)
        states = []
        
        for _ in range(num_steps):
            state = self.simulate_step(
                culture_temp, optimal_temp, dissolved_oxygen_percent,
                perfusion_rate_high, glucose_threshold
            )
            states.append(state)
            
        return states