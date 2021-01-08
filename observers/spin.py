""" Contains the observers for a spin system. """

import numpy as np
import copy

### OBSERVER FILE LATER ###


# Resamples the points in a trajectory to match a given timeset
def resample(trajectory, times, ts):
    """ Resamples a trajectory so that it is in a given list of time points.
    
    Parameters:
        trajectory: the full list of states
        times:  full list of transition times
        ts: times to sample auto-correlator
        
    Returns:
        traj: resampled trajectory
    """
    
    traj = []
    # Loop through each ts
    for t in ts:
        # Find the last time in times smaller than or equal to t
        idx = np.argmax(np.where(times <= t))
        traj.append(trajectory[idx])
    
    
    return traj


###########################

# An observer to measure activity
def activity(trajectory, times, KMC = None):
    """ This observer will measure the activity of a trajectory.
    
    Parameters:
        trajectory: the full list of states
        times:  full list of transition times
        KMC: KMC Class
    """
    
    return np.size(times)-1


# An observer to measure the occupations
def occupations(configuration, KMC = None):
    """ This observer will measure the occupation of a configuration.
    
    Parameters:
        config
        KMC: KMC Class
    """
    
    return copy.deepcopy(configuration.astype(np.int_))


# Measure the auto-correlation function
def autocorrelator(trajectory, times, KMC, ts):
    """ This observer measures the auto-correlator at fixed times.
    
    Parameters:
        trajectory: the full list of states
        times:  full list of transition times
        KMC: KMC Class
        ts: times to sample auto-correlator
    """
    # Resample
    traj = np.array(resample(trajectory, times, ts))
    
    # Create matrix of initial state
    initial = np.array([trajectory[0],]*np.size(ts))
    
    # Find the correlation
    correl = np.mean(np.array(traj) * initial, axis=1)
    
    
    return correl.astype(np.float_)

    