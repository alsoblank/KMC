""" 
Builds on the KMC to perform TPS calculations

Author: Luke Causer
"""

import numpy as np
import copy
from KMC import KMC
import time as timelib
import models.FA

class TPS(KMC):
    
    # Initialize the simulator
    def __init__(self, model, criterion):
        """ A simulator to perform kinetic monte carlo simulations.
        
        Parameters:
           model (class): A model built on a template (basis) with the
                          transition rules defined.
        """
        
        # Call the KMC constructor
        super().__init__(model)
        self.criterion = criterion
        
        return None
    
    
    # Calculate all trajectory observables
    def calculate_trajectory_observables(self, trajectory, times):
        """ Calculates the trajectory observables for a trajectory.
        
        Parameters:
            trajectory: list of configurations
            times: list of times which transitions occur
            
        Returns:
            observables_traj: List of measures of observables
        """
        
        # Loop through the trajectory observables
        observables_traj = []
        for idx in self.obs_traj:
            measure = self.observables[idx][1](trajectory, times, self)
            observables_traj.append(measure)
        
        return observables_traj
            
    
    # Split a trajectory and keep the decided portion
    def split(self, trajectory, times, max_time, obs, time_split, choice):
        """ Will split a trajectory at the given time, discard the relevent
        part.
        
        Parmeters:
            trajectory: list of configurations
            times: list of times which transitions occur
            max_time: Time of trajectories
            obs: List of measures of observables for each config
            time_split: Where to split
            choice: First (0) or last (1) portion.
            
        Returns:
             new_trajectory: list of partial configurations
             new_times: list of partial times
             new_obs: list of partial observables
        """
        
        # Find the index of the last time before time_split
        idx = np.argwhere(np.asarray(times) < time_split)[-1][0]
        
        # Keep the relvent portions
        if choice == 0:
            # First portion
            idx1 = 0
            idx2 = idx+1
        else:
            # second portion
            idx1 = idx
            idx2 = np.size(times)
        
        new_trajectory = copy.deepcopy(trajectory[idx1:idx2])
        new_times = copy.deepcopy(times[idx1:idx2])
        new_obs = []
        for observable in obs:
            new_obs.append(copy.deepcopy(observable[idx1:idx2]))
        
        return [new_trajectory, new_times, new_obs]
    
    
    # Shifts a list of times
    def shift(self, times, time_split, max_time, choice):
        """ Shifts a list of times by shift:
            
            Parameters:
                times: list of times
                time_split: time to split
                max_time: trajectory time
                choice: right (0) or left (1)
            
            Returns:
                new_times: shifted times
        """
        
        if choice == 0:
            new_times = list(np.asarray(times) + max_time - time_split)
        else:
            new_times = list(np.asarray(times) - time_split)
            new_times[0] = 0
        
        return new_times
    
    
    # Reverse a trajectory and it's observables
    def reverse(self, trajectory, times, max_time, observables):
        """ Reverses a trajectory and it's measured observables.
        
         Parmeters:
            trajectory: list of configurations
            times: list of times which transitions occur
            max_time: Time of trajectories
            observables: List of measures of observables for each config
        
        Returns:
            new_trajectory: reversed trajectory
            new_times: reversed times
            new_observables: reversed times
        """
        
        # Reverse the trajectory and observables
        new_trajectory = copy.deepcopy(trajectory[::-1])
        new_observables = []
        for obs in observables:
            new_observables.append(obs[::-1])
        
        # Reverse time
        new_times = [0]
        if len(times) > 1:
            new_times = new_times + list(max_time - np.asarray(times[1:][::-1]))
        
        return [new_trajectory, new_times, new_observables]
    
    
    # Join two lists, removing the first element of the second
    def join(self, list1, list2):
        """ Joins two lists, while removing the first element of the second.
        
        Parameters:
            list1: first list
            list2: second list
        
        Returns:
            list3: joined list
        """
        
        if len(list2) <= 1:
            return list1
        else:
            return list1 + list2[1:]
        
        
    # Propose a new trajectory given the previous
    def propose(self, trajectory, times, max_time, obs):
        """ Calculates the trajectory observables for a trajectory.
        
        Parameters:
            trajectory: list of configurations
            times: list of times which transitions occur
            max_time: Time of trajectories
            obs_config: List of measures of observables for each config
        """
        
        # Choose what to cut
        time_cut = np.random.rand()*max_time # Time to cut at
        split = np.random.randint(2) # First portion or last
        #split = 0
        
        # Split and shift the trajectory
        p_trajectory, p_times, p_observables = self.split(trajectory, times,
                                               max_time, obs, time_cut,
                                               split)
        p_times = self.shift(p_times, time_cut, max_time, split)
        
        # Find the new initial state for the rejuvinated trajectory
        if split == 0:
            initial = trajectory[0]
            time = max_time - time_cut
        else:
            initial = trajectory[-1]
            time = time_cut
        
        # Run a simulation
        t1 = timelib.time()
        [p2_idxs, p2_ts, p2_observables] = self.simulation(initial,
                                                    time, self.obs_config)
        t1 = timelib.time()
        [p2_trajectory, p2_times] = self.reconstruct(initial, p2_idxs, p2_ts)
        
        # Merge the two trajectories
        if split == 0:
            # Reverse the new trajectory
            [p2_trajectory, p2_times, p2_observables] = self.reverse(p2_trajectory,
                                                        p2_times, time, p2_observables)
            
            # Join the two trajectories
            new_trajectory = self.join(p2_trajectory, p_trajectory)
            new_times = self.join(p2_times, p_times)
            new_observables = []
            for i in range(len(self.obs_config)):
                new_observables.append(self.join(p2_observables[i], p_observables[i]))
        else:
            # Add the time
            p2_times = list(np.asarray(p2_times) + max_time - time_cut)
            
            # Join the two trajectories
            new_trajectory = self.join(p_trajectory, p2_trajectory)
            new_times = self.join(p_times, p2_times)
            new_observables = []
            for i in range(len(self.obs_config)):
                new_observables.append(self.join(p_observables[i], p2_observables[i]))
        
        return [new_trajectory, new_times, new_observables]
    
    
    # Metropolis
    def metropolis(self, g1, g2):
        """ Chooses whether to accept or reject g2.
            
            Parameters:
                g1: original probability
                g2: new probability
            
            Returns:
                accept: True (1) or False (0)
        """
        
        accept_rate = np.exp(g2 - g1)
        r = np.random.rand()
        if r < accept_rate:
            return True
        else:
            return False
     
    
    # Run the algorithm
    def run(self, num_sim, max_time, warm_up=0, save=False, quiet=False):
        """ Runs TPS.
        
        Parameters:
            num: Number of simulations
            warmp_up: How many warm up runs to do (default: 0)
            max_time
            save: Directory to save data (default = False)
            quiet: Hide messages? (default = False)
        """
        
        # Store the acceptance rate
        acceptance = 0
        
        # Generate an initial trajectory and reconstruct it, find properties
        initial = self.model.initial()
        [idxs, ts, obs_config] = self.simulation(initial, max_time, self.obs_config)
        [trajectory, times] = self.reconstruct(initial, idxs, ts)
        obs_traj = self.calculate_trajectory_observables(trajectory, times) 
        prob = self.criterion(obs_config, obs_traj, self)    
        
        
        # Loop through for the total number of simulations
        for i in range(num_sim+warm_up):
            # Propose a new trajectory and calculate properties
            [new_trajectory, new_times, new_obs_config] = self.propose(trajectory,
                                                           times, max_time, obs_config)
            new_obs_traj = self.calculate_trajectory_observables(new_trajectory, new_times)
            
            # Accept or reject
            t1 = timelib.time()
            new_prob = self.criterion(new_obs_config, new_obs_traj, self) 
            accept = self.metropolis(prob, new_prob)
            if accept:
                # Update the variables
                trajectory = new_trajectory
                times = new_times
                obs_config = new_obs_config
                obs_traj = new_obs_traj
                prob = new_prob  
                acceptance += 1
            
            # Store observables
            if i >= warm_up:
                j = 0
                for idx in self.obs_config: # Configurations
                    measure = self.time_integrate(obs_config[j], times, max_time)
                    self.update_observable(idx, measure)
                    j += 1
                j = 0
                for idx in self.obs_traj: # Trajectories
                    self.update_observable(idx, obs_traj[j])
                    j += 1
                
                self.num_sims += 1
            
            # Save
            if not quiet:
                print("Simulation "+str(i)+"/"+str(num_sim+warm_up)+" complete.")
        if not quiet:
            print("Acceptance rate: "+str(acceptance/(num_sim+warm_up)))
        
        return True


    