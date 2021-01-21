"""
FA model with N spins.
Same as spin model but with a unique constraint
"""

import numpy as np
import copy
from models.spin import spin
import lib.matrixproducts_FA as mp

class doob(spin):
    
    # Initiate the class
    def __init__(self, N, c, s, psi):
        """ Creates a model for N two-level systems.
        
        Parameters:
            N (int): The number of spins
            c (double): The bias for up spins, must be between 0 and 1
                        (default = 0.5)
        """
        
        # Validate inputs
        if not isinstance(N, int):
            print("N must be an integer.")
            return None
        
        # Construct left vector
        Qdag_local = np.zeros((1, 2, 2, 1))
        Qdag_local[0, :, :, 0] = np.array([[np.sqrt(1-c)**-1, 0], [0, np.sqrt(c)**-1]])
        Qdag = mp.mpo(2, N, 1, Qdag_local)
        left = mp.apply_operator(copy.deepcopy(psi), Qdag, conjugate=True)   
        
        # Store the state of the system
        self.size = N
        self.c = c
        self.s = s
        self.psi = psi
        self.left = left
        self.state = np.zeros(N, dtype=np.bool_)
        self.transition_rates = np.zeros(N)
        self.original_transition_rates = np.zeros(N)
        
        # Tensor blocks
        self.left_blocks = [[] for i in range(N)]
        self.right_blocks = [[] for i in range(N)]
        
        return None
    
    
    # Build up a left block from the previous
    def update_left_block(self, site):
        """ Builds a partial contract with the left vector and current state
        using the previous block.
        
        Parameters:
            site: which site to build at
        """
        
        # Get the left matrix and site state
        state = self.state[site]
        left = self.left.get_matrix(site)
        
        if site == 0:
            # Return just the left at the given site
            block = left[:, state, :]
        else:
            # Fetch previous block and contract
            prev = self.left_blocks[site-1]
            block = np.tensordot(prev, left[:, state, :], axes=(1, 0))
        
        # Update
        self.left_blocks[site] = block
        
        return True
    
    
    # Build up a right block from the previous
    def update_right_block(self, site):
        """ Builds a partial contract with the left vector and current state
        using the previous block.
        
        Parameters:
            site: which site to build at
        """
        
        # Get the left matrix and site state
        state = self.state[site]
        left = self.left.get_matrix(site)
        
        if site == self.size-1:
            # Return just the left at the given site
            block = left[:, state, :]
        else:
            # Fetch previous block and contract
            prev = self.right_blocks[site+1]
            block = np.tensordot(left[:, state, :], prev, axes=(1, 0))
        
        # Update
        self.right_blocks[site] = block
        
        return True
        
        
    # Calculate a left component
    def calculate_left_component(self, site, current=False):
        """ Calcualtes the left component for a configuration by contracting a
        left block, middle at site, and right block.
        
        Parameters:
            site: site which is flipped
            current: calculate current state (true) or transition state (false)
                     (default: false)
         
        Returns:
            lc: left component
        """
        
        # Get the left MPS at site
        if current == False:
            state = 1 - self.state[site]
        else:
            state = self.state[site]
        M = self.left.get_matrix(site)[:, state, :]
        
        if site == 0:
            # Contract site with right
            right = self.right_blocks[site+1]
            lc = np.tensordot(M, right, axes=(1, 0))
        elif site == self.size - 1:
            # Contract site with left
            left = self.left_blocks[site-1]
            lc = np.tensordot(left, M, axes=(1, 0))
        else:
            # Contract site with left
            left = self.left_blocks[site-1]
            right = self.right_blocks[site+1]
            lc = np.tensordot(left, M, axes=(1, 0))
            lc = np.tensordot(lc, right, axes=(1, 0))
        
        return np.abs(float(lc))
            
    
    # Calculate the transition rates
    def update_transition_rates(self, idx, initial=False):
        """ Updates the transition rates for the current system state.
        
        Parameters:
            idx (int): Identifier of how the state was previously updated.
            initial: State whether all the flip rates must be calculated.
                     (default: false)
        """

        # Update original transition rates
        self.update_original_transition_rates(idx, initial)
        
        # Find the first and last sites which can flip
        idx_left = np.where(self.original_transition_rates != 0)[0][0]
        idx_right = np.where(self.original_transition_rates != 0)[0][-1]
        
        if initial is not False:
            # Start from edges
            idx_left_start = 0
            idx_right_start = self.size - 1
        else:
            # Start from idx
            idx_left_start = idx
            idx_right_start = idx
            
        # Buid up blocks
        for i in range(idx_right - idx_left_start):
            self.update_left_block(idx_left_start + i)
        for i in range(idx_right_start - idx_left):
            self.update_right_block(idx_right_start - i)
            
        
        # Update left component
        self.lc = self.calculate_left_component(idx_left, True)
            
        # Loop through each site and calculate the transition rates
        transition_rates = copy.deepcopy(self.original_transition_rates)
        for site in range(self.size):
            if transition_rates[site] != 0:
                lc = self.calculate_left_component(site)
                transition_rates[site] *= np.exp(-self.s) * lc / self.lc
        
        self.transition_rates = transition_rates
        
        return True
    
    
    # Calculate initial state
    def equilibrium_configuration(self):
        """ Randomly generates a configuration from equilibrium. 
        
        Returns:
            configuration
        """
        
        # Build up right blocks
        right_blocks = [[] for i in range(self.size)]
        M = self.psi.get_matrix(self.size - 1) # Get the matrix
        block = np.tensordot(M, M, axes=(1, 1)) # Contract with self
        block = np.trace(block, axis1=1, axis2=3) # Trace away extra dims
        right_blocks[self.size-1] = block
        for site in range(self.size - 2):
            M = self.psi.get_matrix(self.size - 2 - site) # Get the matrix
            block = np.tensordot(M, block, axes=(2, 0)) # Contract with block
            block = np.tensordot(M, block, axes=(2, 2)) # Contract with block
            block = np.trace(block, axis1=1, axis2=3) # Trace dimensions
            right_blocks[self.size - 2 - site] = block
        
        # Store sites and local MPO measures
        sites = []
        left_blocks = []
        mpo_zero = np.zeros((2, 2))
        mpo_zero[0, 0] = 1
        mpo_one = np.zeros((2, 2))
        mpo_one[1, 1] = 1
        
        # First site
        M = self.psi.get_matrix(0) # Get the matrix
        left_zero = np.tensordot(M, mpo_zero, axes=(1, 0)) # Contract with MPO
        left_zero = np.tensordot(left_zero, M, axes=(2, 1)) # Contract with M
        left_zero = np.trace(left_zero, axis1=0, axis2=2) # Trace out
        p0 = np.tensordot(left_zero, right_blocks[1], axes=(0, 0)) # Right block
        p0 = np.trace(p0, axis1=0, axis2=1) # Trace out extra dim
        
        left_one = np.tensordot(M, mpo_one, axes=(1, 0)) # Contract with MPO
        left_one = np.tensordot(left_one, M, axes=(2, 1)) # Contract with M
        left_one = np.trace(left_one, axis1=0, axis2=2) # Trace out
        p1 = np.tensordot(left_one, right_blocks[1], axes=(0, 0)) # Right block
        p1 = np.trace(p1, axis1=0, axis2=1) # Trace out extra dim
        
        r = np.random.rand()*(p0+p1)
        if r < p0:
            left_blocks.append(left_zero)
            sites.append(0)
        else:
            left_blocks.append(left_one)
            sites.append(1)
        
        # Loop through middle sites
        for site in range(self.size - 2):
            M = self.psi.get_matrix(site + 1) # Get the matrix
            left = np.tensordot(left_blocks[site], M, axes=(0, 0))
            
            left_zero = np.tensordot(left, mpo_zero, axes=(1, 0))
            left_zero = np.tensordot(left_zero, M, axes=(0, 0))
            left_zero = np.trace(left_zero, axis1=1, axis2=2)
            p0 = np.tensordot(left_zero, right_blocks[site+2], axes=(0, 0)) # Right block
            p0 = np.trace(p0, axis1=0, axis2=1) # Trace out extra dim
            
            left_one = np.tensordot(left, mpo_one, axes=(1, 0))
            left_one = np.tensordot(left_one, M, axes=(0, 0))
            left_one = np.trace(left_one, axis1=1, axis2=2)
            p1 = np.tensordot(left_one, right_blocks[site+2], axes=(0, 0)) # Right block
            p1 = np.trace(p1, axis1=0, axis2=1) # Trace out extra dim
            
            r = np.random.rand()*(p0+p1)
            if r < p0:
                left_blocks.append(left_zero)
                sites.append(0)
            else:
                left_blocks.append(left_one)
                sites.append(1)
        
        # Final site
        M = self.psi.get_matrix(self.size - 1) # Get the matrix
        left = np.tensordot(left_blocks[self.size - 2], M, axes=(0, 0))
        
        left_zero = np.tensordot(left, mpo_zero, axes=(1, 0))
        left_zero = np.tensordot(left_zero, M, axes=(0, 0))
        left_zero = np.trace(left_zero, axis1=1, axis2=2)
        p0 = float(left_zero)
        
        left_one = np.tensordot(left, mpo_one, axes=(1, 0))
        left_one = np.tensordot(left_one, M, axes=(0, 0))
        left_one = np.trace(left_one, axis1=1, axis2=2)
        p1 = float(left_one)
        
        r = np.random.rand()*(p0+p1)
        if r < p0:
            left_blocks.append(left_zero)
            sites.append(0)
        else:
            left_blocks.append(left_one)
            sites.append(1)
        
        return np.asarray(sites, dtype=np.int_)
            
        
            
            
        
    

    
    