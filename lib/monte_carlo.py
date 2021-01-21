import numpy as np
#import matplotlib.pyplot as plt
import scipy.special as sp
import sys
import os
import time

class data_store():
    # Initiate class
    def __init__(self, activity = True, occupations = True, AC = False,
                 AC_times = False, timescale = False, AC_int = False,
                 AC_int_timescale = False, correlations = True, 
                 correlations_range = 10):
        self.num_sims = 0
        
        self.count_activity = activity
        if activity == True:
            self.activity = 0

        self.count_occupations = occupations
        if occupations == True:
            self.occupations = 0

        self.count_AC = AC
        if AC == True:
            self.AC_times = AC_times
            self.AC = 0

        self.count_timescale = timescale
        if timescale == True:
            self.timescale = 0

        self.count_AC_int = AC_int
        if AC_int:
            self.AC_int = 0

        self.count_AC_int_timescale = AC_int_timescale
        if AC_int_timescale:
            self.AC_int_timescale = 0

        self.count_correlations = correlations
        if correlations:
            self.correlations = np.zeros(correlations_range)
        
    
    # Add to class
    def update(self, num_sims = 0, activity = 0, occupations = 0, AC = 0, timescale = 0,
               AC_int = 0, AC_int_timescale = 0, correlations = 0):
        # Calculate Ratiop
        r1 = self.num_sims / (self.num_sims + num_sims)
        r2 = num_sims / (self.num_sims + num_sims)
        
        # Update parameters
        if self.count_activity == True:
            self.activity = r1 * self.activity + r2*activity
        if self.count_occupations == True:
            self.occupations = r1 * self.occupations + r2*occupations
        if self.count_AC == True:
            self.AC = r1 * self.AC + r2 * AC
        if self.count_timescale == True:
            self.timescale = r1 * self.timescale + r2 *timescale
        #if self.count_AC_int:
        #    self.AC_int = r1 * self.AC_int + r2 * AC_int
        #if self.count_AC_int_timescale:
        #    self.AC_int_timescale = r1 * self.AC_int_timescale + r2 * AC_int_timescale   
        if self.count_correlations:
            self.correlations = r1 * self.correlations + r2*correlations
        
        # Update sims
        self.num_sims += num_sims
        
        


# XOR-FA class for calculations of non-tilted stochastic calculations
class Monte():
    
    # Initiate the class
    def __init__(self, num_sites, tmax, activity = True, occupations = True,
                  AC = False, AC_low = -2, AC_points = 1000, timescale = False,
                  AC_int = False, AC_int_timescale = False, correlations = True,
                  correlations_range = 10):
        # Store the number of sites time for each trajectory
        self.num_sites = num_sites
        self.tmax = tmax
        
        # Create relevent storage 
        self.num_sims = 0
         
        # Count the activity
        self.count_activity = activity
        self.activity = 0
        
        # Count the occupations
        self.count_occupations = occupations
        self.occupations = 0
        
        # Count the AC
        self.count_AC = AC
        self.AC_times = np.zeros(AC_points)
        self.AC_times[1:] = np.logspace(AC_low, np.log10(tmax), AC_points-1)
        self.AC = np.zeros(AC_points)
        
        # Count the TI AC
        self.count_timescale = timescale
        self.timescale = 0

        # Count the correlations
        self.count_correlations = correlations
        self.correlations = np.zeros(correlations_range)
        self.correlations_range = correlations_range

        # Count the AC int
        self.count_AC_int = AC_int
        self.AC_int = np.zeros(AC_points)

        # Count the TI AC
        self.count_AC_int_timescale = AC_int_timescale
        self.AC_int_timescale = 0

      
    
    # Makes a choice according to the correct probabilities
    # takes a probability vector and returns an index
    def choose_probability(self, prob_vec):
        # Calculate the cumulative sum
        cum_sum = np.cumsum(prob_vec)
        
        # Generate a random number
        r = np.random.rand() * cum_sum[-1]
        
        # Find the index which is bigger
        idx = np.argmax(cum_sum > r)
        
        # return the idx
        return idx
         
       
    # Generate an inital state
    def initial_state(self):
        # Make num_sites random numbers
        rs = np.random.rand(self.num_sites) > 0.5
        return rs
        
    
    # Calculate flip rates for each site
    def flip_rates(self, state, previous_flip=None, fliprates=False):        
        # Create a uniform flip rate for each site
        flip_rates = np.ones(self.num_sites)
        return flip_rates
    
    
    # Calculate escape rate of flip rates
    def escape_rate(self, flip_rates):
        return np.sum(flip_rates)
    
    
    def jump_time(self, escape_rate):
        # Generate a random number
        r = np.random.rand()
        
        # Jump time is calculated
        jumptime = - np.log(r) / escape_rate
        
        return jumptime
    
    
    def choose_flip(self, flip_rates):
        # Randomly pick a site to flip
        return self.choose_probability(flip_rates)
    
    
    # Changes the configuration
    def flip_site(self, state, site):
        state[site] = np.abs(state[site]-1)
        return state
    
    
    # Run a KMC simulation
    def KMC(self, initial, tmax):        
        # Set the time to zero and state
        t = 0
        state = np.copy(initial)
        
        # Get the initial flip rates
        flip_rates = self.flip_rates(state)
        
        # Set variables to store the flips and fliptimes
        flips = []
        times = []
        
        # Set the last flipped site by default to zero
        site = 0

        # Loop until maximum time is reached
        while t < tmax:
            # Calculate the updated flip rates
            flip_rates = self.flip_rates(state, site, flip_rates)
            
            # Calculate the escape rate
            escape_rate = self.escape_rate(flip_rates)
            
            # Calculate the jump time
            jumptime = self.jump_time(escape_rate)
            
            # Randomly pick a site to flip
            site = self.choose_flip(flip_rates)
            
            # Update time
            t += jumptime
            
            # If we're not about tmax, add the flip to the trajectory
            if t <= tmax:
                # Add to trajectory
                flips.append(site)
                times.append(t)
                
                # Update the state
                state = self.flip_site(state, site)
            else:
                # Simulation finished
                break
        
        return [initial, np.array(flips), np.array(times)]
    
    
    # Run multiple KMC configurations and save them
    def KMC_simulations(self, num_sim, filesave = None):     
            
        # Loop through all simulations
        for i in range(num_sim):
            self.configs = []
            
            # Generate initial state
            initial = self.initial_state()
            
            # Run a KMC
            [initial, flips, times] = self.KMC(initial, self.tmax)
            
            # Reconstruct trajectory
            [times, trajectory] = self.reconstruct_trajectory(initial, flips, times)
            
            # Add to simulation count 
            self.num_sims += 1
            
            # Add to activity
            if self.count_activity == True:
                self.activity += self.ti_activity(times, trajectory)
            
            # Add to occupations
            if self.count_occupations == True:
                self.occupations += self.ti_occupations(times, trajectory)

            # Add to correlations
            if self.count_correlations == True:
                for j in range(self.correlations_range):
                    self.correlations[j] += self.ti_correlations(times, trajectory, j+1)
            
            # Add to AC
            if self.count_AC == True:
                [ts, AC] = self.auto_correlator(times, trajectory)
                self.AC += AC
            
            # Add to time integrated AC
            if self.count_timescale == True:
                self.timescale += self.ti_timescale(times, trajectory)

            # Add to AC int
            if self.count_AC_int == True:
                [ts, AC_int] = self.auto_correlator_int(times, trajectory)
                self.AC_int += AC_int
            
            # Add to AC int tiemscale
            if self.count_AC_int_timescale == True:
                self.AC_int_timescale += self.auto_correlator_int_ts(times, trajectory)
            
            # If directory is specified, save it
            if filesave != None:
                # Generate the data structure
                data = data_store(self.count_activity, self.count_occupations, self.count_AC, self.AC_times, self.count_timescale, 
                                  self.count_AC_int, self.count_AC_int_timescale, self.count_correlations)
                data.update(self.num_sims, self.activity / (self.num_sims * self.num_sites), self.occupations / self.num_sims, self.AC / self.num_sims,
                                  self.timescale / self.num_sims, self.AC_int / self.num_sims, self.AC_int_timescale / self.num_sims,
                                  self.correlations / self.num_sims)
                
                # Save the data
                np.save(filesave, [data], allow_pickle=True)

            print('Simulation '+str(i+1)+'/'+str(num_sim)+' completed.')
    
    
   # Reconstruct a trajectory from it's inital state, flips and times
    def reconstruct_trajectory(self, initial, flips, fliptimes):
        # Add the inital to the trajectory
        trajectory = [initial]
        
        # Copy the initial to a changable state
        state = np.copy(initial)
        
        # Loop through each flip
        for i in range(np.size(flips)):
            # Change the state
            state[flips[i]] = np.abs(state[flips[i]] - 1)
            
            # Add it to trajectory
            trajectory.append(np.copy(state))
        
        times = np.zeros(np.size(flips)+1)
        times[1:] = fliptimes
        
        return [times, np.array(trajectory)]
            
    
    # Compress the trajectory into slices in time
    def compress_trajectory(self, ts, traj):
        # Get the times
        times = self.AC_times

        # Add the inital config to the trajectory
        trajectory = []
                  
        # Loop through all the changes and add to trajectory where relavent
        for time in times:
            # Find times which are higher            
            if ts[-1] < time:
                idx = -1
            else:
                idx = np.argmax(ts > time) - 1
                
            trajectory.append(traj[idx, :])
        
        return [times, np.array(trajectory)]
    


    # Returns the average number of flips for trajectories
    def ti_activity(self, times, trajectory):
        return (np.size(times) - 1) / self.tmax
    
    
    # Calculate the time integrated occupations of a trajectory
    def ti_occupations(self, times, trajectory):
         # Calculate the time that each configuration lasts for
        time_diff = np.zeros(np.size(times))
        time_diff[0:np.size(time_diff)-1] = times[1:np.size(time_diff)] - times[0:np.size(time_diff)-1]
        time_diff[-1] = self.tmax-times[-1]
        
        # Do the time integration
        occupation = np.tensordot(time_diff, trajectory, axes=(0, 0))
        
        # Divide by time
        occupation = occupation / self.tmax
        
        return occupation


    # Calculate the time integrated excitation density
    def ti_excitation(self, times, trajectory):
        return np.mean(self.ti_occupation(time, trajectory))


    # Calculate the time-integrated correlations
    def ti_correlations(self, times, trajectory, dist=1):
        # Calculate the time that each configuration lasts for
        time_diff = np.zeros(np.size(times))
        time_diff[0:np.size(time_diff)-1] = times[1:np.size(time_diff)] - times[0:np.size(time_diff)-1]
        time_diff[-1] = self.tmax-times[-1]
        
        # Calculate the correlation at each point in time
        correlations = trajectory[:, 0:np.size(trajectory, axis=1) - dist] * trajectory[:, dist:np.size(trajectory, axis=1)]
        correlations = np.mean(correlations, axis=1)
        
        # Do the time-integration
        correlation = np.tensordot(time_diff, correlations, axes=(0, 0))
        
        # Divide by time
        correlation = correlation / self.tmax
        
        return correlation
    
    
    # Calculate the auto correlator
    def auto_correlator(self, times, trajectory):
        # Compress trajectory accordingly
        [times, trajectory] = self.compress_trajectory(times, trajectory)
        
        # Get the inital configuration and make trajectory of it
        initial_config = trajectory[0, :]
        initial_traj = np.tile(initial_config, (np.size(times), 1))
        
        # Calculate the correlations at each point with initial state
        correlation = np.mean(initial_traj * trajectory, axis=1)

        return [times, correlation]
    
    
    # Calculate the auto-correlator averaged by time
    def auto_correlator_int(self, times, trajectory):
        # Get the inital configuration and make trajectory of it
        initial_config = trajectory[0, :]
        initial_traj = np.tile(initial_config, (np.size(times), 1))

        # Calculate the correlations at each point with initial state
        correlation = np.mean(initial_traj * trajectory, axis=1)

        # Calculate the time that each configuration lasts for
        times = np.append(times, self.tmax) # Append tmax to the end
        time_diff = times[1:] - times[0:np.size(times)-1]

        # Calculate the cumulative sum of time diff * correlation and time-average
        correlation_int = np.cumsum(time_diff * correlation) / times[1:] 
        correlation_int = np.insert(correlation_int, 0, correlation[0])
        correlation = np.append(correlation, correlation[-1])
        # Loop through each time in AC and calculate the AC at that point
        correlation_return = []
        i = 0
        for time in self.AC_times:
            # Find times which are higher            
            if times[-1] <= time:
                idx = -1
            else:
                idx = np.argmax(times > time) - 1
            
            if time == 0:
                correlation_return.append(correlation_int[0])
            else:
                correl = (correlation_int[idx] * times[idx] + (time - times[idx])*correlation[idx]) / time
                correlation_return.append(correl)
            
            i += 1
                
        return [self.AC_times, np.array(correlation_return)]

    
    # Calculate the integrated auto-correlator averaged by time
    def auto_correlator_int_ts(self, times, trajectory):
        # Get the inital configuration and make trajectory of it
        initial_config = trajectory[0, :]
        initial_traj = np.tile(initial_config, (np.size(times), 1))

        # Calculate the correlations at each point with initial state
        correlation = np.mean(initial_traj * trajectory, axis=1)

        # Calculate the time that each configuration lasts for
        times = np.append(times, self.tmax) # Append tmax to the end
        time_diff = times[1:] - times[0:np.size(times)-1]

        # Calculate the cumulative sum of time diff * correlation and time-average
        correlation_int = np.cumsum(time_diff * correlation) / times[1:] 

        # Calculate the first segment
        timescale = correlation[0] * time_diff[0]

        # Add the rest of the segments
        correlation = np.append(correlation, correlation[-1])
        A = (correlation_int[0:np.size(correlation_int)-1] - correlation[2:])*times[1:np.size(times)-1]*np.log(times[2:] / times[1:np.size(times)-1])
        A += correlation[2:] * (times[2:] - times[1:np.size(times)-1])
        timescale += np.sum(A)
        
        return timescale

    
    
    # calculate the time integrated AC of a trajectory
    def ti_timescale(self, times, trajectory):
        # Calculate the time that each configuration lasts for
        time_diff = np.zeros(np.size(times))
        time_diff[0:np.size(time_diff)-1] = times[1:np.size(time_diff)] - times[0:np.size(time_diff)-1]
        time_diff[-1] = self.tmax-times[-1]
                    
        # Get the inital configuration and make matrix of original
        initial_config = trajectory[0, :]
        initial_mat = np.tile(initial_config, (np.size(times), 1))
        
        # Calculate the correlation at each site and then average over sites
        correlations = np.mean(initial_mat * trajectory, axis=1)
        
        # Do the time integration
        tau = np.sum(correlations * time_diff)
        
        return tau

    
    
    # Returns whether to accept (true) or reject(false)
    def metropolis_criterion(self, probability):
        if probability >= 1:
            return 1
        else:
            # Return a random number
            r = np.random.rand()
            
            if r < probability:
                return 1
            else:
                return 0
    
