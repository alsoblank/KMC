""" 
This module creates a general framework to run metropolis-hastings monte carlo
simulations.

Author: Luke Causer
"""

import numpy as np
import copy
import time as timelib

class MHMC():
    
    # Initialize the simulator
    def __init__(self, model, probability):
        """ A simulator to perform kinetic monte carlo simulations.
        
        Parameters:
           model (class): A model built on a template (basis) with the
                          transition rules defined.
           probability: A lambda function to a function which calculates the
                        probability of a configuration
        """
        
        # Store the model and set variables
        self.model = model
        self.probability = probability
        self.num_sims = 0
        self.observables = []
        self.measures = []
        
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
    
    
    # Propose a new configuration
    def propose(self, config):
        """ Save the data to the given file directory
        
        Parameters:
            config: configuration
        
        Returns:
            new_config: new configuration
        """
        
        # In the transition rates, find non zeros
        idxs = np.where(self.model.transition_rates != 0)[0]
        
        # Pick one randomly
        idx = idxs[self.random_choice(np.ones(np.size(idxs)))]
    
        
        return idx
    

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
    
    
    # Run a number of simulations
    def run(self, num, start_config, warm_up=0, save_freq=1, quiet=False):
        """ Runs a number of simulation.
        
        Parameters:
            num: Number of simulations
            start_config: a good guess for a start configuration
            warm_up: number of warm-up runs (default = 0)
            save_freq: the number of proposed configurations before saving
                       (default = 1)
            save: Directory to save data (default = False)
            quiet: Hide messages? (default = False)
        """                        
        
        # Update config and transtion rates
        config = copy.deepcopy(start_config)
        prob = self.probability(config)
        self.model.update_state(copy.deepcopy(config))
        self.model.update_transition_rates(0, True)
        
        # Loop through the number of simulations
        num_sims = num*save_freq + warm_up
        configs = []
        for i in range(num_sims):
            # Propose a new configuration and find it's probability
            idx = self.propose(copy.deepcopy(config))
            self.model.transition(idx)
            new_config = copy.deepcopy(self.model.state)
            new_prob = self.probability(new_config)
            
            # Choose to accept or reject
            accept = self.metropolis(prob, new_prob)
            if accept == True:
                config = copy.deepcopy(new_config)
                prob = copy.deepcopy(new_prob)
                self.model.update_transition_rates(idx)
            else:
                self.model.update_state(copy.deepcopy(config))
            
            # Update the number of simulations
            if i >= warm_up:
                if (i - warm_up) % save_freq == 0:
                    # Calculate observables
                    j = 0
                    for obs in self.observables:
                        self.update_observable(j, obs[1](config, self))
                        j += 1
                                        
                    # Add the configuration
                    self.num_sims += 1
                    configs.append(config)
            
                    # Save data
                
                
                    # Print out message
                    if quiet != True:
                        print("Simulation "+str(self.num_sims)+"/"+str(num)+" completed.")
                     
        return configs
    
    
    # Add an observable to measure
    def observer(self, name, func):
        """ Add an observable to measure.
        
        Parameters:
            name (string): Give a name to the observable to reference it by.
            func: Pass through a lambdea function which takes the KMC class
                  as a parameter.
        """
        
        # Ensure the name has not already been used
        for obs in self.observables:
            if obs[0] == name:
                print("Observables must be given unique identifiers.")
                return False
        
        # Store the observable
        self.observables.append([name, func])
        self.measures.append([])
        
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
        