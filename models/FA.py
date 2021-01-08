"""
FA model with N spins.
Same as spin model but with a unique constraint
"""

import numpy as np
import copy
from models.spin import spin

class famodel(spin):
    
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
            if idx == 0:
                idxs = [0, 1]
            elif idx == self.size - 1:
                idxs = [idx, idx-1]
            else:
                idxs = [idx-1, idx, idx+1]
        else:
            idxs = list(range(0, self.size))
        
        # Update each index
        for i in idxs:
            self.transition_rates[i] = (1-self.c)**self.state[i] * self.c**(1-self.state[i])
            if i == 0:
                self.transition_rates[i] *= self.state[i+1].astype(np.float_)
            elif i == self.size - 1:
                self.transition_rates[i] *= self.state[i-1].astype(np.float_)
            else:
                self.transition_rates[i] *= self.state[i-1].astype(np.float_) + self.state[i+1].astype(np.float_)
                
        
        return True
    

    
    