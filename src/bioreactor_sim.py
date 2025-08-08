# CHO Cell Perfusion Bioreactor Simulation with Descriptive Variable Names



import numpy as np

class BioreactorSimulation:
    """
    A class to simulate a CHO Cell Perfusion Bioreactor.
    
    This simulation models the growth of Chinese Hamster Ovary (CHO) cells in a perfusion bioreactor,
    including cell growth, death, substrate consumption, metabolite production, and environmental effects.
    
    The model includes:
    - Cell growth and death kinetics
    - Substrate (glucose, glutamine) consumption
    - Metabolite (lactate, ammonia) production
    - Product formation
    - Environmental inhibition effects (pH, temperature)
    - Density-dependent growth limitation
    - Perfusion control based on glucose concentration
    
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
        ph_inhibition_coeff=0.05,
        temp_inhibition_coeff=0.04,
        culture_ph=6.5,
        optimal_ph=7.2,
        glucose_feed_conc=8.0,
        glutamine_feed_conc=5.0
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
        self.ph_inhibition_coeff = ph_inhibition_coeff
        self.temp_inhibition_coeff = temp_inhibition_coeff
        self.culture_ph = culture_ph
        self.optimal_ph = optimal_ph
        self.glucose_feed_conc = glucose_feed_conc
        self.glutamine_feed_conc = glutamine_feed_conc
        
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
        
        # Apply environmental inhibition factors
        specific_growth_rate *= np.exp(
            -self.lactate_inhibition_coeff * self.lactate_concentration
            - self.ammonia_inhibition_coeff * self.ammonia_concentration
            - self.ph_inhibition_coeff * abs(self.culture_ph - self.optimal_ph)
            - self.temp_inhibition_coeff * abs(culture_temp - optimal_temp)
        )

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