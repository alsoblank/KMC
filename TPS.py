""" 
Builds on the KMC to perform TPS calculations

Author: Luke Causer
"""

import numpy as np
import copy
from KMC import KMC

class TPS(KMC):
    
    # Initialize the simulator
    def __init__(self, model):
        """ A simulator to perform kinetic monte carlo simulations.
        
        Parameters:
           model (class): A model built on a template (basis) with the
                          transition rules defined.
        """
        
        # Call the KMC constructor
        super().__init__(model)
        
        return None
    
    
    # Run the algorithm
    