# CHO Cell Perfusion Bioreactor Simulation with Descriptive Variable Names



import numpy as np

class BioreactorSimulation:
    """
    A class to simulate a CHO Cell Perfusion Bioreactor with mechanistic kinetics.
    
    This simulation models the growth of Chinese Hamster Ovary (CHO) cells in a perfusion bioreactor,
    including cell growth, death, substrate consumption, metabolite production, and environmental effects.
    
    The model includes:
    - Cell growth and death kinetics
    - Mechanistic glucose consumption (growth-associated + maintenance)
    - Lactate metabolic switching (production ↔ consumption based on glucose levels)
    - Substrate (glucose, glutamine) consumption with condition-dependent rates
    - Metabolite (lactate, ammonia) production
    - Product formation
    - Environmental inhibition effects (pH, temperature) with asymmetric responses
    - Density-dependent growth limitation
    - Perfusion control based on glucose concentration
    
    Key Features:
    - Glucose uptake follows Pirt model: q_G = glucose_growth_coeff × μ + glucose_maintenance_rate
    - Lactate metabolism switches between production (high glucose) and consumption (low glucose)
    - Environmental conditions affect growth rate, which propagates to glucose consumption
    
    Attributes:
        time_step (float): Simulation time step (hours)
        time (float): Current simulation time
        viable_cell_density (float): Concentration of viable cells
        dead_cell_density (float): Concentration of dead cells
        glucose_concentration (float): Concentration of glucose
        glutamine_concentration (float): Concentration of glutamine
        lactate_concentration (float): Concentration of lactate
        ammonia_concentration (float): Concentration of ammonia
        product_concentration (float): Concentration of product
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
        measurement_noise=0.0,
        # NEW: Mechanistic glucose and lactate kinetics parameters
        glucose_growth_coeff=0.5,        # Glucose needed per unit growth (g glucose/g cells)
        glucose_maintenance_rate=0.01,   # Maintenance glucose consumption (g/(cell·h))
        lactate_shift_glucose=2.5,       # Glucose level where lactate shifts from production to consumption
        lactate_switch_steepness=2.0,    # Sharpness of the metabolic switch
        lactate_consumption_max=0.02,    # Maximum lactate consumption rate per cell
        lactate_half_saturation=0.5      # Half-saturation constant for lactate consumption
    ):
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
        
        # NEW: Mechanistic kinetics parameters
        self.glucose_growth_coeff = glucose_growth_coeff
        self.glucose_maintenance_rate = glucose_maintenance_rate
        self.lactate_shift_glucose = lactate_shift_glucose
        self.lactate_switch_steepness = lactate_switch_steepness
        self.lactate_consumption_max = lactate_consumption_max
        self.lactate_half_saturation = lactate_half_saturation
        
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

    def calculate_condition_dependent_glucose_rate(self, specific_growth_rate):
        """
        Calculate glucose uptake rate based on growth conditions (Pirt model).
        
        Args:
            specific_growth_rate (float): Current specific growth rate (1/h)
            
        Returns:
            float: Specific glucose uptake rate (g glucose per cell per hour)
        """
        # Growth-associated glucose consumption + maintenance glucose consumption
        growth_glucose = self.glucose_growth_coeff * specific_growth_rate
        maintenance_glucose = self.glucose_maintenance_rate
        
        return max(0.0, growth_glucose + maintenance_glucose)

    def calculate_condition_dependent_lactate_rate(self, glucose_rate):
        """
        Calculate lactate production/consumption rate with metabolic switching.
        
        Args:
            glucose_rate (float): Current glucose uptake rate
            
        Returns:
            float: Specific lactate rate (g lactate per cell per hour)
                   Positive = production, Negative = consumption
        """
        # Logistic switching function based on glucose concentration
        # sigma = 1 (production) when glucose is high, sigma = 0 (consumption) when glucose is low
        sigma = 1.0 / (1.0 + np.exp(-self.lactate_switch_steepness * 
                                    (self.glucose_concentration - self.lactate_shift_glucose)))
        
        # Production phase: lactate produced from glucose consumption
        lactate_production = self.lactate_yield_per_glucose * glucose_rate
        
        # Consumption phase: lactate consumed with Monod kinetics
        lactate_consumption = (self.lactate_consumption_max * 
                              self.lactate_concentration / 
                              (self.lactate_half_saturation + self.lactate_concentration + 1e-12))
        
        # Smooth blend between production and consumption
        lactate_rate = sigma * lactate_production - (1.0 - sigma) * lactate_consumption
        
        return lactate_rate

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

        # NEW: Calculate condition-dependent rates
        condition_glucose_rate = self.calculate_condition_dependent_glucose_rate(specific_growth_rate)
        condition_lactate_rate = self.calculate_condition_dependent_lactate_rate(condition_glucose_rate)

        # Update state variables
        self.viable_cell_density += self.time_step * (
            specific_growth_rate * self.viable_cell_density - self.death_rate * self.viable_cell_density
        )
        
        self.dead_cell_density += self.time_step * (
            self.death_rate * self.viable_cell_density - dilution_rate * self.dead_cell_density
        )
        
        self.glucose_concentration += self.time_step * (
            -condition_glucose_rate * self.viable_cell_density + dilution_rate * (self.glucose_feed_conc - self.glucose_concentration)
        )
        
        self.glutamine_concentration += self.time_step * (
            -self.glutamine_uptake_rate * self.viable_cell_density + dilution_rate * (self.glutamine_feed_conc - self.glutamine_concentration)
        )
        
        self.lactate_concentration += self.time_step * (
            condition_lactate_rate * self.viable_cell_density - dilution_rate * self.lactate_concentration
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