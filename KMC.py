""" 
This module creates a general framework to run kinetic monte carlo
simulations.

Author: Luke Causer
"""

import numpy as np

class KMC():
    
    # Initialize the simulator
    def __init__(self, model):
        """ A simulator to perform kinetic monte carlo simulations.
        
        Parameters:
           model (class): A model built on a template (basis) with the
                          transition rules defined.
        """
        
        # Store the model and set variables
        self.model = model
        self.num_sims = 0
        
        return None
    
    
    # Pick an index from a list of weightings
    def random_choice(self, weights):
        """ Picks an index from a list of weightings randomly.
        
        Parameters:
            weights: list of weightings
        """
        
        cs = np.cumsum(weights) # Cumulative sum
        r = np.random.rand() * cs[-1] # Random number in the whole range
        idx = np.argmin(cs < r) # Find first number greater than r
        return idx
    
    
    # Calculates the escape rate
    def escape_rate(self, transition_rates):
        """ Calcualtes the escape rate of a configuration.
        
        Parameters:
            transition_rates
        """
        
        return np.sum(transition_rates)
    
    
    # Calculates a random transition time
    def transition_time(self, escape_rate):
        """ Calculates a random time for the system to transition
        
        Parameters:
            escape_rate
        """
        
        r = np.random.rand() # Generate a random number
        time = -np.log(r) / escape_rate
        return time
    
    
    # Run a KMC simulation
    def simulation(self, initial, max_time):
        """ Runs a simulation.
        
        Parameters:
            initial: The initial state of the system
            max_time: Time to run the simulation for
            
        Returns:
            idxs: List of identifiers which tell how the system transitions
            times: List of times when transitions occur
        """
        
        # Update the model with the IS, and calculate initial TRs
        self.model.update_state(initial)
        self.model.update_transition_rates(0, True)
        
        # Keep track of how much time has occur and store trajectory
        t = 0
        idxs = []
        times = []
        
        # Loop until the max time is passed
        while t < max_time:
            # Pick a configuration to transition into
            idx = self.random_choice(self.model.transition_rates)
            
            # Randomly pick a jump time
            escape_rate = self.escape_rate(self.model.transition_rates)
            time = self.transition_time(escape_rate)
            
            # Update the system
            self.model.transition(idx)
            self.model.update_transition_rates(idx)
            t += time
            
            # Check to see we haven't surpassed max_time
            if t <= max_time:
                # Store new trajectory information
                idxs.append(idx)
                times.append(t)
        
        return [idxs, times]
    
    
    # Run a number of simulations
    def run(self, num, max_time):
        """ Runs a number of simulation.
        
        Parameters:
            num: Number of simulations
            max_time
        """
        
        
            
            
            