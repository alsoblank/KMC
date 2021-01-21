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

# Calculate the escape rate
def escape_rate(configuration, KMC = None):
    """ This observer will measure the escape rate of configurations.
    
    Parameters:
        config
        KMC: KMC Class
    """
    
    return np.sum(KMC.model.transition_rates)


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
    
    return copy.deepcopy(np.asarray(configuration, dtype=np.float_))


# Measure the auto-correlation function
def autocorrelator(trajectory, times, KMC, ts):
    """ This observer measures the auto-correlator at fixed times.
    
    Parameters:
        trajectory: the full list of states
        times:  full list of transition times
        KMC: KMC Class
        ts: times to sample auto-correlator
    """
    
    # Create matrix of initial state
    initial = np.array([trajectory[0],]*len(trajectory))
    
    # Find the correlation
    correl = np.mean(np.array(trajectory) * initial, axis=1)
    
    # Resample
    correl = resample(np.array(correl), times, ts)    
    
    return np.asarray(correl)


# Measure the integrated autocorrelator
def int_autocorrelator(trajectory, times, KMC, tmax):
    """ This observer measures the integral of the auto-correlator.
    
    Parameters:
        trajectory: the full list of states
        times:  full list of transition times
        KMC: KMC Class
        tmax: max time
    """

    # Create matrix of initial state
    initial = np.array([trajectory[0],]*len(trajectory))
    
    # Find the correlation
    correl = np.mean(np.array(trajectory) * initial, axis=1)
    
    # Create a list of times including tmax and find the difference
    ts = np.append(times, tmax)
    t_diffs = ts[1:] - ts[0:np.size(ts)-1]
    
    # Integrate
    integ = np.tensordot(t_diffs, np.array(correl), axes=(0,0))
    
    return integ
    

# Measure the auto-correlation function
def time_occupations(trajectory, times, KMC, ts):
    """ This observer measures the occupations at fixed times.
    
    Parameters:
        trajectory: the full list of states
        times:  full list of transition times
        KMC: KMC Class
        ts: times to sample auto-correlator
    """
    # Resample
    traj = resample(np.asarray(trajectory, dtype=np.int_), times, ts)
    
    return np.asarray(traj, dtype=np.int_)

    