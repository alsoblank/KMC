""" 
This module creates a general framework to run kinetic monte carlo
simulations.

Author: Luke Causer
"""

import numpy as np
import copy
import time as timelib

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
        self.observables = []
        self.measures = []
        self.obs_config = []
        self.obs_traj = []
        
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
    def simulation(self, initial, max_time, obs_idxs = None):
        """ Runs a simulation.
        
        Parameters:
            initial: The initial state of the system
            max_time: Time to run the simulation for
            obs_idxs: Identifiers for observables (default: None)
            
        Returns:
            idxs: List of identifiers which tell how the system transitions
            times: List of times when transitions occur
            observables: list of measured observables (optional)
        """
        
        # Update the model with the IS, and calculate initial TRs
        self.model.update_state(copy.deepcopy(initial))
        self.model.update_transition_rates(0, True)        
        
        # Variables to keep track of system variables
        t = 0
        idxs = []
        times = []
        if obs_idxs != None:
            observables = []
            for idx in obs_idxs:
                measure = self.observables[idx][1](self.model.state, self)
                observables.append([measure])
                
        
        # Loop until the max time is passed
        while t < max_time:
            # Pick a configuration to transition into
            idx = self.random_choice(self.model.transition_rates)
            
            # Randomly pick a jump time
            escape_rate = self.escape_rate(self.model.transition_rates)
            time = self.transition_time(escape_rate)
            t += time
            
            # Check to see we haven't surpassed max_time
            if t <= max_time:
                # Update the system
                self.model.transition(idx)
                self.model.update_transition_rates(idx)
                
                # Measure observables
                if obs_idxs != None:
                    i = 0 # Count the indexs
                    for obj_idx in obs_idxs:
                        measure = self.observables[obj_idx][1](self.model.state, self)
                        observables[i].append(measure)
                    i += 1
                
                # Store new trajectory information
                idxs.append(idx)
                times.append(t)
        
        # Return data
        if obs_idxs == None:
            return [idxs, times]
        else:
            return [idxs, times, observables]
    
    
    # Uses the data from KMC to reconstruct the whole trajectory
    def reconstruct(self, initial, idxs, times):
        """ Reconstructs a trajectory with the minimum information needed.
        
        Parameters:
            initial: initial state
            idxs: list of identifiers for transitions
            times: list of times which transitions occur
            
        Returns:
            trajectory: list of configs in trajectory
            ts: times of transitions
        """
        
        # Create the lists
        trajectory = [initial]
        ts = [0]
        config = copy.deepcopy(initial) # Create a copy to edit
        self.model.update_state(config) # Set the state
        
        # Loop through each transition
        for i in range(np.size(idxs)):
            # Update configuration
            self.model.transition(idxs[i])
            config = self.model.state
            
            # Store in lists
            trajectory.append(copy.deepcopy(config))
            ts.append(copy.deepcopy(times[i]))
        
        return [trajectory, ts]
    
    
    # Save the data
    def save(self, initial, idxs, times, directory):
        """ Save the data to the given file directory
        
        Parameters:
            initial: initial state
            idxs: list of identifiers for transitions
            times: list of times which transitions occur
            directory: the filename where to save the data
        """
        
        # Create a data list and save it
        data = np.array([], dtype=np.object_)
        np.save(directory, data, True)
        
        return True
    
    
    # Run a number of simulations
    def run(self, num, max_time, save=False, quiet=False):
        """ Runs a number of simulation.
        
        Parameters:
            num: Number of simulations
            max_time
            save: Directory to save data (default = False)
            quiet: Hide messages? (default = False)
        """                        
        
        # Loop through the number of simulations
        for i in range(num):
            # Run a simulation
            initial = self.model.initial() # Generate initial state
            [idxs, times, observables] = self.simulation(initial, max_time, self.obs_config) # Simulate
            
            # Measure observables
            [trajectory, ts] = self.reconstruct(initial, idxs, times)
            j = 0
            for idx in self.obs_config: # Configurations
                measure = self.time_integrate(observables[j], ts, max_time)
                self.update_observable(idx, measure)
                j += 1
            for idx in self.obs_traj: # Trajectories
                measure = self.observables[idx][1](trajectory, ts, self)
                self.update_observable(idx, measure)
            
            # Update the number of simulations
            self.num_sims += 1
            
            # Save data
            if save is not False:
                directory = save + str(i) + '.npy' # Create directory name
                self.save(initial, idxs, times, directory) # Save it
            
            # Print out message
            if quiet != True:
                print("Simulation "+str(i+1)+"/"+str(num)+" completed.")
                     
        return True
    
    
    # Add an observable to measure
    def observer(self, name, func, act='configuration'):
        """ Add an observable to measure.
        
        Parameters:
            name (string): Give a name to the observable to reference it by.
            func: Pass through a lambdea function which takes the KMC class
                  as a parameter.
            act: 'configuration' acts on local configurations over a time int,
                  'trajectory' is an observable which acts on the whole traj.
        """
        
        # Ensure the name has not already been used
        for obs in self.observables:
            if obs[0] == name:
                print("Observables must be given unique identifiers.")
                return False
        
        # Check the type is valid
        if act != 'configuration' and act != 'trajectory':
            print("The act must be 'configuration' or 'trajectory'.")
            return False
        
        # Store the observable
        self.observables.append([name, func, act])
        self.measures.append([])
        
        # Add to relevent list
        if act == 'configuration':
            self.obs_config.append(len(self.observables)-1)
        else:
            self.obs_traj.append(len(self.observables)-1)
        return True
    
    
    # Update an observable
    def update_observable(self, idx, data):
        """ Update an observable.
        
        Parameters:
            idx: identifier for the observable
            data: data to update
        """
        
        # Update
        if self.num_sims == 0:
            self.measures[idx] = data
        else:
            self.measures[idx] += data
        
        return True
    
    
    
    # Calculates the time integration over an observable
    def time_integrate(self, observable, times, time_max):
        """ Calculates the time integration over an observable, measured
        across a trajectory.
        
        Parameters:
            observable: list of observable
            times: list of transition times
            time_max: maximum trajectory time
        """
        
        observable = np.array(observable)
        
        # Create a list of times including tmax and find the difference
        ts = np.append(times, time_max)
        t_diffs = ts[1:] - ts[0:np.size(ts)-1]
        
        # Integrate
        integ = np.tensordot(t_diffs, np.array(observable), axes=(0,0))
        
        return integ
    
        
    # Find the index of a named observable
    def get_observable_idx(self, name):
        """ Returns the index where a named observable is stored.
        
        Parameters:
            name: string for name
        
        Returns:
            idx: identifer for index (None if does not exist)
        """
        
        idx = 0
        for observable in self.observables:
            if observable[0] == name:
                return idx
            idx += 1
        
        return None
    
    
    # Return an observable
    def get_measure(self, name):
        """ Returns the average observable measure.
        
        Parameters:
            name: string for name
        
        Returns:
            data
        """
        
        idx = self.get_observable_idx(name)
        if idx == None:
            print("Observable not defined.")
            return False
    
        return self.measures[idx] / self.num_sims
        
"""          
act = lambda trajectory, times, KMC: activity(trajectory, times, KMC)
occ = lambda configuration, KMC: occupations(configuration, KMC)
Fa = famodel(100, 0.5)
sim = KMC(Fa)
sim.observer("Activity", act, 'trajectory')
#sim.observer("Occupations", occ, 'configuration')
sim.run(10000, 10)        
"""           