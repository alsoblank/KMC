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
        
        # Construct full state
        state = np.zeros(self.size+2, dtype=np.float_)
        state[1:self.size+1] = self.state
        
        # If initial, calculate all
        if initial is not False:
            # Calculate constraint
            self.transition_rates = state[0:self.size] + state[2:self.size+2]
            self.transition_rates *= (1-self.c)**self.state * self.c**(1-self.state)
        else:
            if idx == 0:
                idxs = [idx, idx+1]
            elif idx == self.size - 1:
                idxs = [idx, idx-1]
            else:
                idxs = [idx-1, idx, idx+1]
            
            for i in idxs:
                self.transition_rates[i] = (1-self.c)**self.state[i] * self.c**(1-self.state[i])
                self.transition_rates[i] *= state[i].astype(np.float_) + state[i+2].astype(np.float_)
            
        
        return True
    

    
    