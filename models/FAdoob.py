"""
FA model with N spins.
Same as spin model but with a unique constraint
"""

import numpy as np
import copy
from models.doob import doob
import lib.matrixproducts_FA as mp

class fadoob(doob):
    
    # Initiate the class
    def __init__(self, N, c, s, psi):
        """ Creates a model for N two-level systems.
        
        Parameters:
            N (int): The number of spins
            c (float): The bias for up spins, must be between 0 and 1
            s (float): Activity bias
            psi (MPS): Probability density vector
        """
        
        # Call doob constructor
        super().__init__(N, c, s, psi)
        
        
        # Set the default way to generate an initial state
        self.initial = lambda: self.equilibrium_configuration()
        
        return None
    
    
    # Calculate the original transition rates
    def update_original_transition_rates(self, idx, initial=False):
        """ Updates the original transition rates for the current system state.
        
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
            self.original_transition_rates = state[0:self.size] + state[2:self.size+2]
            self.original_transition_rates *= (1-self.c)**self.state * self.c**(1-self.state)
        else:
            if idx == 0:
                idxs = [idx, idx+1]
            elif idx == self.size - 1:
                idxs = [idx, idx-1]
            else:
                idxs = [idx-1, idx, idx+1]
            
            for i in idxs:
                self.original_transition_rates[i] = (1-self.c)**self.state[i] * self.c**(1-self.state[i])
                self.original_transition_rates[i] *= state[i].astype(np.float_) + state[i+2].astype(np.float_)
            
        
        return True
    

    
    