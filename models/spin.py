"""
A model for N two-level systems, which flip independantly from one-another.
"""

import numpy as np
import copy

class spin():
    
    # Initiate the class
    def __init__(self, N, c = 0.5):
        """ Creates a model for N two-level systems.
        
        Parameters:
            N (int): The number of spins
            c (double): The bias for up spins, must be between 0 and 1
                        (default = 0.5)
        """
        
        # Validate inputs
        if not isinstance(N, int):
            print("N must be an integer.")
            return None
        
        if not isinstance(c, float):
            print("c must be a number between 0 and 1.")
            return None
        else:
            if c < 0 or c > 1:
               print("c must be a number between 0 and 1.")
               return None 
        
        # Store the state of the system
        self.size = N
        self.c = c
        self.state = np.zeros(N, dtype=np.bool_)
        self.transition_rates = np.zeros(N)
        
        # Set the default way to generate an initial state
        self.initial = lambda: self.equilibrium_configuration()
        
        return None
    
    
    # Sets the state of the model
    def update_state(self, state):
        """ Updates the state of the system.
        
        Parameters:
            state (numpy array): the input state
        """
        
        self.state = state
        return True
    
    
    # Calculate the transition rates
    def update_transition_rates(self, idx, initial=False):
        """ Updates the transition rates for the current system state.
        
        Parameters:
            idx (int): Identifier of how the state was previously updated.
            initial: State whether all the flip rates must be calculated.
                     (default: false)
        """
        
        # What sites must be calculated
        if initial is False:
            idxs = [idx]
        else:
            idxs = list(range(0, self.size))
        
        # Update each index
        for i in idxs:
            self.transition_rates[i] = (1-self.c)**self.state[i] * self.c**(1-self.state[i])
        
        return True
    
    
    # Perform the transition into a new configuration
    def transition(self, idx):
        """ Transitions the system into a new configuration.
        
        Parameters:
            idx (int): Identifier for the system to transition.
        """
        
        # Update the system
        self.state[idx] = 1 - self.state[idx]
        
        return True
    
    
    # Generate a configuration from equilibrium
    def equilibrium_configuration(self):
        """ Generates a configuration sampled from equilibrium. """
        
        # Generate list of random numbers
        rs = np.random.rand(self.size)
        
        # Find the config
        config = rs < self.c
        config.astype(np.bool_)
            
        return config


    
    