""" Contains the observers for a spin system. """

import numpy as np
import copy

def escape_rate(configuration, KMC = None):
    """ This observer will measure the escape rate of configurations.
    
    Parameters:
        config
        KMC: KMC Class
    """
    
    return np.sum(KMC.model.transition_rates)


def original_escape_rate(configuration, KMC = None):
    """ This observer will measure the original escape rate of configurations.
    
    Parameters:
        config
        KMC: KMC Class
    """
    
    return np.sum(KMC.model.original_transition_rates)


def left_components(configuration, KMC = None):
    """ This observer will measure the left components of configurations.
    
    Parameters:
        config
        KMC: KMC Class
    """
    
    return KMC.model.lc

    