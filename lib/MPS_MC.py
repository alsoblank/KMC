import numpy as np
# import matplotlib.pyplot as plt
import scipy.special as sp
import sys
import os
import lib.monte_carlo as mc
import lib.matrixproducts_FA as mp
import copy


class MPS_MC():    
    
    # Calcuate all the right contractions of an MPS
    def right_blocks(self, psi, stop_idx = 0):
        # Get the size of the MPS
        num_sites = psi.length
        
        # If the stop idx is all the way to the right, return nothing
        if stop_idx == num_sites - 1:
            return []
        
        # list to store right blocks
        rights = []
        
        # Get the last site
        A = psi.get_matrix(num_sites - 1)
        
        # Reshape it 
        A = np.reshape(A, (np.size(A, axis=0), np.size(A, axis=1)))
        
        # Contract with it's self to get a (DxD matrix)
        right = np.tensordot(A, A, axes=(1, 1))
        rights.append(right)
        
        # Loop through the remaining
        for i in range(num_sites-1-stop_idx):
            # Which site is it
            site = num_sites-1-i
            
            # Get the matrix
            A = psi.get_matrix(site-1)
            
            # Contract it with right block twice and then over physical dim
            right = np.tensordot(A, right, axes=(2, 1))
            right = np.tensordot(A, right, axes=(2, 2))
            right = np.trace(right, axis1=1, axis2=3)
            
            # Add to rights
            rights.append(right)
        
        return rights
    
    
    # Calculate all the left contractions of an MPS
    
    # Initiates a left block from MPS tensor A and local MPO O
    def start_left_block(self, A, O):
        # Reshape A to cut out dim 0
        A = np.reshape(A, (np.size(A, axis=1), np.size(A, axis=2)))
        
        # Contract the first with local MPO
        left = np.tensordot(A, O, axes=(0, 0))
        
        # Contract with A
        left = np.tensordot(left, A, axes=(1, 0))
        
        return left
    
    
    # Grows a left block using an MPS tensor A and local MPO O
    def grow_left_block(self, left, A, O):
        # Contract with A
        left = np.tensordot(left, A, axes=(0, 0))
        
        # Contract with O
        left = np.tensordot(left, O, axes=(1, 0))
        
        # Contract with A
        left = np.tensordot(left, A, axes=(0, 0))
        
        # Trace over physical dimension
        left = np.trace(left, axis1=1, axis2=2)
        
        # Return the new left block
        return left
        
    
    # Contract a left block with a right one
    def contract_blocks(self, left, right):
        # Contract left with right
        prod = np.tensordot(left, right, axes=(0,0))
        
        # Trace over
        prod = np.trace(prod)
        
        # Return the product
        return prod
    
    
    # Generate a configuration from the probability vector
    def configuration(self, psi):
        # Get the system size
        N = psi.length
        
        # Store config
        config = []
        
        # Get all the right blocks
        rights = self.right_blocks(psi)
        
        # Construct local MPO tensors
        zero = np.array([[1, 0], [0, 0]])
        one = np.array([[0, 0], [0, 1]])
        
        # Get the starting probability
        prob = rights[-1]
        left = 1
        
        for site in range(N):        
            # Get the site
            A = psi.get_matrix(site)
            
            # Grow (or create the left)
            if site == 0:
                # Create the first left block for one
                left_one = self.start_left_block(A, one)
            else:
                # Grow it
                left_one = self.grow_left_block(left, A, one)
            
            # Contract it with the right block
            if site != N - 1:
                right = rights[N-2-site]
                prob_1 = self.contract_blocks(left_one, right)
            else:
                prob_1 = left_one
            
            # Make a choice
            r = np.random.rand()*prob
            if r < prob_1:
                config.append(1)
                prob = prob_1
                left = left_one
            else:
                config.append(0)
                prob = prob - prob_1
                if site == 0:
                    # Create the first left block for one
                    left_zero = self.start_left_block(A, zero)
                else:
                    # Grow it
                    left_zero = self.grow_left_block(left, A, zero)
                left = left_zero

        # Make sure config isn't full of zeros    
        config = np.array(config).astype(int)
        if np.sum(config) == 0:
            config = self.configuration(psi)

        return config
    
    
    def blocks(self, psi, state, left_stop, right_stop):
        # Create the left blocks
        left = []
        if left_stop > 0:
            # Count which site we're at
            site = 0
            
            # Get the first matrix
            matrix = psi.get_matrix(site)
            
            # Take the appropiate index
            prod = matrix[:, state[site], :]
            
            # Add it to the left blocks
            left.append(prod)
            
            site += 1
            while site < left_stop and site < psi.length - 1:
                # Get the next matrix
                matrix = psi.get_matrix(site)
                
                # Calculate the next step in the block
                prod = np.tensordot(prod, matrix[:, state[site], :], axes=(1, 0))
                
                # Add it to the left blocks
                left.append(prod)
                site += 1
                
        # Create the left blocks
        right = []
        if right_stop < psi.length - 1:
            # Count which site we're at
            site = psi.length - 1
            
            # Get the first matrix
            matrix = psi.get_matrix(site)
            
            # Take the appropiate index
            prod = matrix[:, state[site], :]
            
            # Add it to the left blocks
            right.append(prod)
            
            site += -1
            while site > right_stop and site > 0:
                # Get the next matrix
                matrix = psi.get_matrix(site)
                
                # Calculate the next step in the block
                prod = np.tensordot(matrix[:, state[site], :], prod, axes=(1, 0))
                
                # Add it to the left blocks
                right.append(prod)
                site += -1
                
        return left, right
    
    
    # Compute single MPS contractions
    def contract_single_blocks(self, left, A, right):
        # Contract over left and A
        prod = np.tensordot(left, A, axes=(1, 0))
        
        # Contract over right
        prod = np.tensordot(prod, right, axes=(1, 0))    
        
        return np.float(prod)
    

# XOR-FA class for calculations of non-tilted stochastic calculations
class MC_MPS(mc.Monte, MPS_MC):
    
    def __init__(self, c, s, psi, num_sites, tmax, activity = True, occupations = True,
                  AC = False, AC_low = -2, AC_points = 1000, timescale = False,
                  AC_int = False, AC_int_timescale = False, correlations = True,
                  correlations_range = 10):
        # Set parameters
        self.c = c
        self.num_sites = psi.length
        self.s = s
        
        # Probability distribtion
        self.psi = psi
        
        # Left vectors
        Qdag_local = np.zeros((1, 2, 2, 1))
        Qdag_local[0, :, :, 0] = np.array([[np.sqrt(1-c)**-1, 0], [0, np.sqrt(c)**-1]])
        Qdag = mp.mpo(2, self.num_sites, 1, Qdag_local)
        self.left = mp.apply_operator(copy.deepcopy(self.psi), Qdag, conjugate=True)     
        
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

        # Count the correlations
        self.count_correlations = correlations
        self.correlations = np.zeros(correlations_range)
        self.correlations_range = correlations_range
        
        # Count the AC
        self.count_AC = AC
        self.AC_times = np.zeros(AC_points)
        self.AC_times[1:] = np.logspace(AC_low, np.log10(tmax), AC_points-1)
        self.AC = np.zeros(AC_points)
        
        # Count the TI AC
        self.count_timescale = timescale
        self.timescale = 0

        # Count the AC int
        self.count_AC_int = AC_int
        self.AC_int = np.zeros(AC_points)

        # Count the TI AC
        self.count_AC_int_timescale = AC_int_timescale
        self.AC_int_timescale = 0
        
        self.configs = []
        self.escaperates = []
        self.original_escaperates = []

    
            
                    
    # Sample a monte carlo state
    def initial_state(self):
        # Return sampling form MC class
        return self.configuration(self.psi)
        
    
    # Check the constraint on a site:
    def constraint(self, state, idx):
        # Ensure idx is int
        idx = int(idx) 
        
        # Get the number of sites
        num_sites = np.size(state)
        
        # OBC; different for end sites
        if idx == 0:
            return state[idx+1]
        elif idx == num_sites - 1:
            return state[idx-1]
        else:
            return state[idx+1] + state[idx-1]
    
    
    def original_flip_rates(self, state, previous_flip):
        # Get system size
        num_sites = np.size(state)

        
        # Check to see if we're given information to make more efficient
        if previous_flip != None:
            if previous_flip == 0:
                sites = [previous_flip,  previous_flip+1]
            elif previous_flip == num_sites - 1:
                sites = [previous_flip - 1, previous_flip]
            else:
                sites = [previous_flip - 1, previous_flip, previous_flip+1]
            flip_rates = self.original_fliprates
        else:
            sites = range(num_sites)
            flip_rates = np.zeros(num_sites)

        
        # Loop through each site
        for site in sites:
            if state[site] == 0:
                flip_rates[site] = self.constraint(state, site) * self.c
            else:
                flip_rates[site] = self.constraint(state, site) * (1-self.c)
        
        # return flip rates
        return flip_rates
    
    
    # Find the left component of a configuration
    def left_component(self, configuration):
        # Get the first tensor in l
        left = self.left.get_matrix(0)
        
        # Pick the dimension which corresponds to configuration
        left = left[:, configuration[0], :]
        
        # loop through the rest and contract
        for i in range(np.size(configuration) - 1):
            site = i + 1
            
            # Get tensor
            A = self.left.get_matrix(site)
            A = A[:, configuration[site], :]
            
            # Contract
            left = np.tensordot(left, A, axes=(1, 0))
        
        return np.abs(float(left))
    
    
    # Calculate the average excitation density
    def excitation_density(self):
        # Get the number of sites
        num_sites = self.num_sites
        
        # Define local operators
        I = np.zeros((1, 2, 2, 1))
        I[0, :, :, 0] = np.identity(2)
        n = np.zeros((1, 2, 2, 1))
        n[0, :, :, 0] = np.array([[0, 0], [0, 1]])
        
        # Set the excitation to zero
        excitation = 0
        
        # Loop through each site
        for site in range(num_sites):
            # Construct an MPO with n at site and I everywhere else
            MPO = mp.mpo(2, num_sites, 1, I)
            MPO.edit_structure(site, n)
            
            # Find the expection wrt psi
            excitation += mp.expectation(MPO, self.psi)
        
        # Divide by the number of sites
        excitation = excitation / num_sites
        
        # Return excitation
        return excitation  


    # Calculate the average excitation density
    def excitation_density_sq(self):
        # Get the number of sites
        num_sites = self.num_sites
        
        # Define local operators
        I = np.zeros((1, 2, 2, 1))
        I[0, :, :, 0] = np.identity(2)
        n = np.zeros((1, 2, 2, 1))
        n[0, :, :, 0] = np.array([[0, 0], [0, 1]])
        
        # Set the excitation to zero
        excitation = 0
        
        # Loop through each site
        for site in range(num_sites):
            # Construct an MPO with n at site and I everywhere else
            MPO = mp.mpo(2, num_sites, 1, I)
            MPO.edit_structure(site, n)
            
            # Find the expection wrt psi
            excitation += mp.expectation(MPO, self.psi)**2
        
        # Divide by the number of sites
        excitation = excitation / num_sites
        
        # Return excitation
        return excitation
                
    
    # Calculate flip rates for each site
    def flip_rates(self, state, previous_flip=None, fliprates=False):
        # Get system size
        num_sites = np.size(state)
        
        # Calculate the left-component of this configuration
        l_c = self.left_component(state)
        
        # Update the original fliprates and make a copy
        self.original_fliprates = self.original_flip_rates(state, previous_flip)  
        flip_rates = copy.deepcopy(self.original_fliprates)
        
        # Find which flips are allowed
        allowed_flips = np.where(self.original_fliprates != 0)[0]
        
        # Contract left block up to the most right index, and right up to the
        # most left
        l = allowed_flips[-1]
        r = allowed_flips[0]
        lefts, rights = self.blocks(self.left, state, l, r)
        
        # Loop through each possible flip
        for flip in allowed_flips:
            # Get the local tensor and choose the config with the changed site
            A = self.left.get_matrix(flip)
            A = A[:, np.abs(state[flip]-1), :]
            
            # Contract over blocks
            if flip == 0:
                l_c2 = np.tensordot(A, rights[num_sites-1-flip-1], axes=(1, 0))
            elif flip == num_sites - 1:
                l_c2 = np.tensordot(lefts[flip - 1], A, axes=(1, 0))
            else:
                l_c2 = np.tensordot(lefts[flip - 1], A, axes=(1, 0))
                l_c2 = np.tensordot(l_c2, rights[num_sites-1-flip-1], axes=(1, 0))
            
            # Make it a positive float
            l_c2 = np.abs(float(l_c2))
            
            # Update the fliprate
            flip_rates[flip] = flip_rates[flip]*(l_c2 / l_c) * np.exp(-self.s)
        
        return flip_rates
    
    
    # Changes the configuration; overrides to store configurations in class
    def flip_site(self, state, site):
        # Change the state
        new_state = copy.deepcopy(state)
        new_state[site] = np.abs(new_state[site]-1)
        
        # Store it
        self.configs.append(new_state)
        
        # Return the new state
        return new_state
    
    
    # Calculated the escape rate; override to store all escape rates in class
    def escape_rate(self, flip_rates):
        # Calculate it
        escape = np.sum(flip_rates)
        
        # Store it
        self.escaperates.append(escape)
        
        # Do the same for the original flip rates
        self.original_escaperates.append(np.sum(self.original_fliprates))
        
        # Return the escape rate
        return escape
    
    
    
    # Evolve a trajectory forward in time
    def fast_forward(self, initial, flips, times, configs, tmax, tsplit, escape, originalescape):
        # Find the final configuration the state is in
        if np.size(configs) == 0:
            e_initial = initial
        else:
            e_initial = configs[-1].astype(np.int)
          
        # Run a new KMC simulation
        self.configs = []
        self.escaperates = []
        self.original_escaperates = []
        [e_initial, e_flips, e_times] = self.KMC(e_initial, tsplit)
        
        e_configs = np.array(self.configs)
        e_escape = np.array(self.escaperates)
        e_originalescape = np.array(self.original_escaperates)
        
        # Join together the new trajectory
        p_flips = np.append(flips, e_flips)
        p_times = np.append(times, e_times + tmax)
        if np.size(configs) == 0:
            p_configs = e_configs
        elif np.size(e_configs) == 0:
            p_configs = configs
        else:
            p_configs = np.append(configs, e_configs, axis = 0)
        if np.size(e_escape) == 1:
            p_escape = escape
            p_originalescape = originalescape
        else:
            # Don't double add the inbetween state
            p_escape = np.append(escape, e_escape[1:])
            p_originalescape = np.append(originalescape, e_originalescape[1:])
        
        
        # Subtract the time off to reshift to zero
        p_times = p_times - tsplit
        
        # Find the first change in trajectory after tsplit
        if np.size(p_times) == 0:
            # No flips at all happen, just the same state
            p_initial = initial
        elif p_times[-1] < 0:
            # tsplit is after last flip
            p_initial = p_configs[-1]
            p_times = np.array([])
            p_flips = np.array([])
            p_configs = []
            p_escape = p_escape[-1]
            p_originalescape = p_originalescape[-1]
        else:
            # Find the first flip after tsplit
            idx = np.argmax(p_times > 0)
            
            p_flips = p_flips[idx:]
            p_times = p_times[idx:]
            
            # Get the state before the flip
            if idx == 0:
                p_initial = initial
            else:
                p_initial = p_configs[idx-1]
            
            p_configs = p_configs[idx:]
            p_escape = p_escape[idx:]
            p_originalescape = p_originalescape[idx:]
                
        return [p_initial.astype(np.int), p_flips, p_times, p_configs, p_escape, p_originalescape]
    
    
    
    # Evolve a trajectory backwards in time
    def rewind(self, initial, flips, times, configs, tmax, tsplit, escape, originalescape):
        # Run a new KMC simulation from the initial state
        self.configs = []
        self.escaperates = []
        self.original_escaperates = []
        [e_initial, e_flips, e_times] = self.KMC(initial, tmax-tsplit)
        e_configs = self.configs
        e_escape = np.array(self.escaperates)
        e_originalescape = np.array(self.original_escaperates)
        
        # Flip the trajectory
        if e_configs != []:
            # Take the last configuration as the initial
            e_initial = e_configs[-1]
            
            # Flip the ordering of flips
            e_flips = np.flip(e_flips)
            e_times = -np.flip(e_times)
            
            # Flip the configurations around
            e_configs = np.flip(np.array(e_configs[0:-1]), axis=0)

            # Flip the escape rates
            e_escape = np.flip(e_escape)
            e_originalescape = np.flip(e_originalescape)
            
            # Put the initial state into the same form
            e_init = np.zeros((1, np.size(e_initial)))
            e_init[0, :] = e_initial
            
            # Check to see if there's anything left in configs after removing last
            if np.size(e_configs) == 0:
                e_configs = e_init
            else:
                e_configs = np.append(e_configs, e_init, axis=0)
            
        # Put the two trajectories together
        p_initial = e_initial
        p_times = np.append(e_times, times)
        p_flips = np.append(e_flips, flips)
        
        if np.size(configs) == 0:
            p_configs = e_configs
        elif np.size(e_configs) == 0:
            p_configs = configs
        else:
            p_configs = np.append(e_configs, configs, axis=0)
        
        # Don't double count state inbetween
        if np.size(escape) == 1:
            p_escape = e_escape
            p_originalescape = e_originalescape
        else:
            p_escape = np.append(e_escape, escape[1:])
            p_originalescape = np.append(e_originalescape, originalescape[1:])
        
        # Shift the times by tmax-tsplit
        p_times = p_times + (tmax-tsplit)
        
        # Find the last flip which occurs before tmax
        if np.size(p_flips) != 0:
            # There are flips
            check = p_times < tmax
            
            if np.sum(check) != 0:
                # Flips do occur before tmax, find which ones do
                idxs = np.where(check)[0]
                
                # Select the final one
                idx = idxs[-1]
                
                # Only take the required flips
                p_times = p_times[0:idx+1]
                p_flips = p_flips[0:idx+1]
                p_configs = p_configs[0:idx+1]
                p_escape = p_escape[0:idx+2]
                p_originalescape = p_originalescape[0:idx+2]
                
        # Return the new trajectory
        return [p_initial.astype(np.int), p_flips, p_times, p_configs, p_escape, p_originalescape]
                
        
    # Calculate the time-integrated escape rate for a trajectory given the times
    # and escape rates
    def TIER(self, vector, ts, tmax):
        # Add tmax onto the end
        ts = np.append(ts, tmax)
        
        # Find the time differences
        diff = ts[1:]-ts[0:-1]
        
        return np.sum(diff*vector)  
    
    
    # Transition Path Sampling Method
    def TPS(self, num_sim, num_sites, tmax, directory, warm_up = 0):
        # Create an empty vector to store the state at each flip in trajectory
        self.configs = []
        self.escaperates = []
        self.original_escaperates = []
        
        # Store activity
        activity = 0
        
        # Generate an initial trajectory
        [initial, flips, times] = self.KMC(self.initial_state(), tmax)
        
        # Fetch the configs for each point
        configs = np.array(self.configs)
        escape = np.array(self.escaperates)
        originalescape = np.array(self.original_escaperates)
        
        # Calculate the left component of the end points
        linitial = self.left_component(initial)
        if np.size(configs) == 0:
            lfinal = self.left_component(initial)
        else:
            lfinal = self.left_component(configs[-1])
        
        # Calculate the time integrated escape rate
        tier = self.TIER(escape-originalescape, times, tmax)
        
        # Measure the acceptence rate
        acceptance_rate = 0
        
        # Continiously propose perturbations, and accept of reject
        for i in range(num_sim + warm_up):
            # Pick a time to split the trajectory
            tsplit = np.random.rand()*tmax
            
            # Pick which half to keep
            r = np.random.randint(2)
            
            if r == 0:
                # Keep the first portion; rewind time
                [p_initial, p_flips, p_times, p_configs, p_escape, p_originalescape] = self.rewind(initial,
                                           flips, times, configs, tmax, tsplit, escape, originalescape)
            elif r == 1:
                # Keep the second; fast forward time
                [p_initial, p_flips, p_times, p_configs, p_escape, p_originalescape] = self.fast_forward(initial,
                                           flips, times, configs, tmax, tsplit, escape, originalescape)
                
            # Calculate the new left components and TIER
            p_linitial = self.left_component(p_initial)
            if np.size(p_configs) == 0:
                p_lfinal = self.left_component(p_initial)
            else:
                p_lfinal = self.left_component(p_configs[-1].astype(np.int))

            p_tier = self.TIER(p_escape-p_originalescape, p_times, tmax)
            
            # Calculate the g-factors for acception/rejection
            #g = (linitial / lfinal) * np.exp(tier) / self.initial_probability(initial)
            #p_g = (p_linitial / p_lfinal) * np.exp(p_tier) / self.initial_probability(p_initial)
            g = np.exp(tier)
            p_g = np.exp(p_tier)
            
            # Calculate the acceptance proability
            #prob = p_g / g
            prob = np.exp(p_tier - tier)
            print(prob)
            
            # Accept or reject using metropolis criterion
            if self.metropolis_criterion(prob):
                # Accepted, make new state
                initial = p_initial.astype(np.int)
                flips = p_flips
                times = p_times
                configs = p_configs
                escape = p_escape
                originalescape = p_originalescape
                linitial = p_linitial
                lfinal = p_lfinal
                tier = p_tier
                
                # Add to acceptence rate
                if i >= warm_up:
                    acceptance_rate += 1
            
            # Decided whether to save or not
            if i >= warm_up:
                # Make sure directory exists
                if not os.path.exists(directory + 'initial/'):
                     os.makedirs(directory + 'initial/')
                if not os.path.exists(directory + 'flips/'):
                     os.makedirs(directory + 'flips/')
                if not os.path.exists(directory + 'times/'):
                     os.makedirs(directory + 'times/')
                 
                # Save the trajectory
                np.save(directory + 'initial/' + str(i) + '.npy', initial.astype(bool),
                        allow_pickle = True)
                np.save(directory + 'flips/' + str(i) + '.npy', flips.astype(np.int16),
                        allow_pickle = True)
                np.save(directory + 'times/' + str(i) + '.npy', times,
                        allow_pickle = True)
                
                # Increase activity
                activity += np.size(times)
                
                
            
            print('Simulation '+str(i+1)+'/'+str(num_sim+warm_up)+' completed.')
        print('Activity: '+ str(activity/(num_sim*self.tmax)))
        self.activity = activity/(num_sim*self.tmax)
        print('Acceptance Rate is '+str(acceptance_rate / num_sim))
    
    
    # Randomly sample configurations, and their assosiated left components
    def left_component_sampling(self, num_configs):
        # Store configs and left components
        configs = []
        lefts = []
        
        # Loop through num_configs
        for i in range(num_configs):
            config = self.configuration(self.psi)
            left = self.left_component(config)
            
            configs.append(config)
            lefts.append(left)
        
        return [np.array(configs).astype(int), np.array(lefts)]
    
