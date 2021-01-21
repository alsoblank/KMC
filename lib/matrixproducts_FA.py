"""
States of high-dimensional systems can be seperated into matrix product
states. These are local states which are related to their neighbours
through index contraction. This module is designed to deal with matrix product
states.

author: @Luke Causer
"""

import numpy as np
import copy
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import svds
import time

class mps:
       
    def __init__(self, dim, length, bond_dim = 0, input_state = None):
        """Initiate an MPS.
        dim = dimension of system
        length = length of atom chain
        bond_dim = bond_dimension of the system"""
        
        # Assign the parameters to the MPS
        self.dim = dim
        self.length = length
        self.bond_dim = bond_dim
        
        # Deal with no bond dim
        if self.bond_dim == 0:
            self.bond_dim = int(self.dim**(self.length / 2))
        
        # Make sure the bond dimension isn't bigger than it needs to be
        max_dim = int(self.dim**(self.length/2))
        if self.bond_dim > max_dim:
            self.bond_dim = max_dim
        
        # Create the structure
        self.structure = self.create_structure()
        
        # If an input state is given, generate an MPS for it
        if input_state is not None:
            self.state_to_mps(input_state)
    
    
    def create_structure(self):
        """ Creates the structure for the MPS """
        
        # Create a zeros object with the length of the system
        structure = np.zeros(self.length, dtype='object')
        
        # Loop through each atom in the chain and generate it's structure
        for i in range(self.length):
            # Dimension sizes
            D1 = self.bond_dim
            D2 = self.dim
            D3 = self.bond_dim   
            if i < int(self.length / 2):
                D1 = int(min(self.bond_dim, self.dim**(i)))
                D3 = int(min(self.bond_dim, self.dim**(i+1)))
            else:
                D1 = int(min(self.bond_dim, self.dim**(self.length-i)))
                D3 = int(min(self.bond_dim, self.dim**(self.length-i-1)))
                
            # Store the shape
            structure[i] = np.zeros((D1, D2, D3))
        
        # Set the structure
        return structure

            
            
    def edit_structure(self, site, matrix):
        """ Edits the matrix at the given site
        site = Site to edit the matrix
        Matrix = new matrix """
        
        # Reset the structure to zero
        r1, d, r2 = np.shape(self.structure[site])
        self.structure[site][0:r1, 0:d, 0:r2] = np.zeros((r1, d, r2)) 
        
        # Get the shape of matrix
        r1, d, r2 = np.shape(matrix)
        
        # Insert the matrix into the structure
        self.structure[site][0:r1, 0:d, 0:r2] = matrix

    
    def get_matrix(self, site):
        """ Returns the matrix at a given site
        Inputs:
            site = site to return
        Outputs:
            matrix = matrix to return
        """
        
        return copy.copy(self.structure[site])
    
    
    def random_mps(self):
        """ Generate random matrix entries for the MPS """
        
        # Loop through each component
        for i in range(self.length):
            # Get the shape of the matrix at site i
            shape = np.shape(self.structure[i])
            
            # Generate as many random elements as necersary
            random = np.random.rand(np.prod(shape), 1)
            
            # Store them in each matrix
            self.structure[i] = np.reshape(random, shape)
            
            
    def copy_mps(self, MPS):
        """ Copies a MPS with a lower bond dimension """
        
        for site in range(self.length):
            self.edit_structure(site, MPS.get_matrix(site))

    
    
    def state_to_mps(self, state):
        """ Take a vector input state on our L sites and put it into MPS
        form. """
        
        # We start by reshaping the vector into a (2, d^(L-1)). We are
        # essentially writing it in terms of the 2 dimensions of the first
        # site.
        D = np.reshape(state, (self.dim, self.dim**(self.length-1)))
        
        # Recursively apply SVD, reshape and store
        for site in range(self.length-1):
            # Apply SVD
            U, S, V = svd(D)
            
            # Calculate the minimum of the bond dimension and the number of
            # singular values
            r = min(np.size(S, axis=0), self.bond_dim)
            
            # If bond_dim > S, we surpress it to the higher singular values
            # as they contribute the most to the state
            U = U[:, 0:r]
            S = S[0:r, 0:r]
            V = V[0:r, :]
            
            # Calculate the inner and outer dimensions of the matrix.
            # Naturally, if site < length / 2 then the they will be
            # (d^i, d^(i+1)). If site > length / 2 then (d^(L-i), d^(L-i-1))
            # We adjust them according to the bond dimension.
            if site < self.length / 2:
                r1 = min(self.dim**site, self.bond_dim)
                r2 = min(self.dim**(site+1), self.bond_dim)
            else:
                r1 = min(self.dim**(self.length-site), self.bond_dim)
                r2 = min(self.dim**(self.length-site-1), self.bond_dim)
            
            # Store U into the structure  
            A = np.reshape(U, (r1, self.dim, r2))
            self.edit_structure(site, A)
            
            # Calculate D to be the product of S and V, and reshape it
            D = np.matmul(S, V)
            D = np.reshape(D, (self.dim*r, self.dim**(self.length - 2- site)))
        
        # We finally reshape the remaining D to be the last entry
        D = np.reshape(D, (self.dim, self.dim, 1))
        self.edit_structure(site + 1, D)
    
    
    def to_state(self):
        """ Returns the MPS in a vector state form """
        
        # Get the first matrix and reshape the first dimension (1) out
        state = self.get_matrix(0)
        state = np.reshape(state, (self.dim, np.size(state, axis=2)))
        
        # Loop through, taking the tensor product across states
        for site in range(self.length-1):
            matrix = self.get_matrix(site + 1)
            state = np.tensordot(state, matrix, axes=(1+site, 0))
            
        # Reshape into a (d^L, 1) vector
        return np.reshape(state, (self.dim**self.length, 1))
    
    
    def mean_field(self, vector):
        """ Turns the MPS into a mean field given an individual vector """
        
        matrix = np.zeros((1, self.dim, 1))
        matrix[0, :, 0] = vector[:, 0]
        
        for site in range(self.length):
            self.edit_structure(site, matrix)
    
    
    def left_canonical(self, terminate = None):
        """ Moves the MPS into it's left canonical, up to the given site
        Inputs:
            terminate = site to move upto left canonical form (optional) """
        
        # If no site is given, just do it for the whole MPS
        if terminate is None:
            terminate = self.length - 1
        
        # Retrieve the first matrix
        M1 = self.get_matrix(0)
        
        # Reshape the first dimension out
        M1 = np.reshape(M1, (self.dim, np.size(M1, axis=2)))
        
        # Apply SVD
        A, S, V = svd(M1)
        
        # Reshape A into the correct form and update it
        s = np.size(S, axis=0)
        A = np.reshape(A, (1, self.dim, s))
        self.edit_structure(0, A)
        
        # Recursively update the middle entries
        for site in range(terminate - 1):
            # We need to absorb S and V into the next matrix
            M = self.get_matrix(site + 1)
            M = np.tensordot(np.matmul(S, V), M, axes=(1, 0))
            
            # Reshape M by compounding the first index with the physical index
            M = np.reshape(M, (s*self.dim, np.size(M, axis=2)))
            
            # Apply SVD to M
            A, S, V = svd(M)
            
            # Get the correct dimensions
            olds = s
            s = np.size(S, axis=0)
            
            # Update the matrix at site
            A = np.reshape(A, (olds, self.dim, s))
            self.edit_structure(site+1, A)
       
        # Absorb S and V into the last matrix  and update it
        ML = self.get_matrix(terminate)
        ML = np.tensordot(np.matmul(S, V), ML, axes=(1, 0)) 
        self.edit_structure(terminate, ML)
        
        
    def right_canonical(self, terminate = None):
        """ Moves the MPS into it's right canonical, up to the given site
        Inputs:
            terminate = site to move upto right canonical form (optional) """
        
        # If no site is given, just do it for the whole MPS
        if terminate is None:
            terminate = 0
        
        # Retrieve the last matrix
        ML = self.get_matrix(self.length-1)
        
        # Reshape the last dimension out
        ML = np.reshape(ML, (np.size(ML, axis=0), self.dim))
        
        # Apply SVD
        U, S, B = svd(ML)
        
        # Reshape B into the correct form and update it
        s = np.size(S, axis=0)
        B = np.reshape(B, (s, self.dim, 1))
        self.edit_structure(self.length-1, B)
        
        # Recursively update the middle entries
        for site in range(self.length - 2 - terminate):
            # We need to absorb U and S into the previous matrix
            M = self.get_matrix(self.length-2-site)
            M = np.tensordot(M, np.matmul(U, S), axes=(2, 0))
            
            # Reshape M by compounding the last index with the physical index
            M = np.reshape(M, (np.size(M, axis=0), self.dim*s))
            
            # Apply SVD to M
            U, S, B = svd(M)
            
            # Get the correct dimensions
            olds = s
            s = np.size(S, axis=0)
            
            # Update the matrix at site
            B = np.reshape(B, (s, self.dim, olds))
            self.edit_structure(self.length-2-site, B)
       
        # Absorb U and S into the matrix and update it
        M1 = self.get_matrix(terminate)
        M1 = np.tensordot(M1, np.matmul(U, S), axes=(2, 0)) 
        self.edit_structure(terminate, M1)
        
        
    def mixed_canonical(self, site):
        """ Puts the MPS into a mixed canonical state around the input site.
        Inputs:
            site = site to put into mixed canonical form around """
        
        # If we're at the first site, just put it in right canonical.
        # Same with last site and left canonical. Otherwise, left canonical
        # to the left of the site and right canonical to the right of site
        if site == 0:
            self.right_canonical()
        elif site == self.length - 1:
            self.left_canonical()
        else:
            self.left_canonical(site)
            self.right_canonical(site)
            
            
            

class mpo:
    
    def __init__(self, dim, length, bond_dim, local_operator = None):
        """Initiate an MPO.
        Inputs:
            dim = dimension of system
            length = length of atom chain
            bond_dim = bond_dimension of the operator
            local_operator = Operator which acts locally on sites
        """
        
        # Set the parameters
        self.dim = dim
        self.length = length
        self.bond_dim = bond_dim
        
        # Create the structure
        self.structure = self.create_structure()
        
        # If a local operator is provided, use it to build the structure
        if local_operator is not None:
            self.build_structure(local_operator)
        
        
    def create_structure(self):
        """ Creates the structure for the MPS """
        
        # Create a zeros object with the length of the system
        structure = np.zeros(self.length, dtype='object')
        
        # Loop through each atom in the chain and generate it's structure
        for i in range(self.length):
            # Dimension sizes
            D1 = self.bond_dim
            D2 = self.dim
            D3 = self.bond_dim            
            if i == 0:
                D1 = 1
            elif i == self.length - 1:
                D3 = 1
            
            structure[i] = np.zeros((D1, D2, D2, D3))
        
        # Set the structure
        return structure
    
    
    def edit_structure(self, site, matrix):
        """ Edits the matrix at the given site
        site = Site to edit the matrix
        Matrix = new matrix """
        
        # Get the shape of matrix
        r1, d1, d2, r2 = np.shape(matrix)
        
        # Insert the matrix into the structure
        self.structure[site][0:r1, 0:d1, 0:d2, 0:r2] = matrix
    
    
    def build_structure(self, local_operator):
        """ Takes a lower diagonial local operator and repeats it over all
        the sites
        Inputs:
            local_operator = Operator which acts locally on sites
        """
        
        # The first site just takes the bottom row
        M1 = local_operator[-1:, :, :, :]
        self.edit_structure(0, M1)
        
        # The middle sites are the whole operator
        for site in range(self.length - 2):
            self.edit_structure(site + 1, local_operator)
        
        # The end sites are the first column of the operator
        ML = local_operator[:, :, :, 0:1]
        self.edit_structure(self.length-1, ML)
        
    
    def get_matrix(self, site):
        """ Returns the matrix at a given site
        Inputs:
            site = site to return
        Outputs:
            matrix = matrix to return
        """
        
        return np.copy(self.structure[site])
    

def truncate(mps1, bond_dim):
    """ Edits the MPS to one with a lower bond dimension via truncation.
    It keeps the largest singular values.
    Inputs:
        bond_dim = new bond dimension
    """
    
    # move into right canonical
    mps1.right_canonical()
    
    # Make a new mps
    mps2 = mps(mps1.dim, mps1.length, bond_dim)
    
    # Get the first tensor
    A = mps1.get_matrix(0)
    
    # Loop through each bond dimension
    for site in range(mps1.length - 1):  
        # Get the next tensor
        B = mps1.get_matrix(site + 1)
        
        # Get the shape
        [m1, d, m2] = np.shape(A)
        
        # Reshape the site, putting physical and virtual dimensions together
        A = group_indices(A, 0, 1)

        
        if min(m1*d, m2) > bond_dim:
            # Apply sparse SVD to find largest bond_dim singular values
            U, S, V = svds(A, min(bond_dim, m1*d), which='LM')
            S = np.diag(S)
        else:
            # Apply normal SVD
            U, S, V = svd(A)
                

        # Renormalize S
        renorm = np.sqrt(np.trace(S*S))
        S = S / renorm
               
        D = min(bond_dim, m2)
        
        # Ungroup indices in U
        U = moveindex(U, 0) # Moves grouped to end
        U = np.reshape(U, (np.size(U, 0), m1, d)) # Reshape
        U = moveindex(U, 0) # Put indices into correct order
        
        #
        A = U
        
        # Contract S, V
        V = np.tensordot(S, V, axes=(1, 0))
        
        # Contract V with next tensor
        B = np.tensordot(V, B, axes=(1, 0))
            
        # Update tensors
        mps2.structure[site] = A
        A = B
            
    # Update final tensor
    mps2.structure[site+1] = A
    
    return mps2


def svd(matrix):
    """ Apply singular value decomposition. The rank is r=min(m, n)       
    Inputs:
        matrix = Input an mxn matrix
    Outputs:
        U = mxr matrix
        S = rxr matrix
        V = rxn matrix
    """
    
    # Get the shape of the matrix
    m, n = np.shape(matrix)
    r = min(m, n)
    
    # Apply SVD
    U, S, V = np.linalg.svd(matrix)
    
    # Return the correct size matrices
    U = U[0:m, 0:r]
    V = V[0:r, 0:n]
    S = np.diag(S)
    return U, S, V


def tensor_size(tensor1):
    """ Returns how many indices (or axes) a tensor has """
    
    tensor = copy.copy(tensor1)
    
    # Get the shape
    shape = np.shape(tensor)
    
    # Get the size of the shape
    return np.size(shape)


def moveindex(tensor1, axis, position=-1):
    """ Moves the index in the tensor to the given position. If position = 0
    then move it to the end."""
    
    tensor = copy.copy(tensor1)
    
    # Check to see if position is 0, if so get the end axis position
    if position == -1:
        position = tensor_size(tensor) - 1
    
    # Loop through from it's current position to the desired position, swapping
    # axes
    for i in range(np.abs(position - axis)):
        if position > axis:
            tensor = np.swapaxes(tensor, axis1 = axis+i, axis2 = axis+i+1)
        elif position < axis:
            tensor = np.swapaxes(tensor, axis1 = axis-i, axis2 = axis-i-1)
    
    return tensor
            


def group_indices(tensor1, axis1, axis2):
    """ Groups together indices by taking them to the end of the tensor,
    grouping them and putting them back to the position of the first axis. """
    
    tensor = copy.copy(tensor1)
    size = tensor_size(tensor)
    
    # Move axis 2 to the end and then axis 1 to the end and swap axis 1 and 2
    tensor = moveindex(tensor, axis2)
    tensor = moveindex(tensor, axis1)
    tensor = np.swapaxes(tensor, axis1=size-2, axis2=size-1)
    
    # Get the shape  of the tensor
    shape = np.shape(tensor)
    
    # Calculate the new shape, where the last two indices are grouped
    newshape = np.zeros((size - 1), dtype='int')
    for i in range(size - 2):
        newshape[i] = shape[i]
    newshape[size-2] = shape[size-2]*shape[size-1]
    
    # Reshape the tensor
    tensor = np.reshape(tensor, newshape)
    
    # Move the new index back to axis 1 position
    tensor = moveindex(tensor, size-2, axis1)
    
    return tensor
    

def product(tensor1, tensor2):
    """ Takes two tensors and contracts the indices. 
    MPS x MPS = scalar
    MPO x MPS = MPS x MPO = MPS
    MPO x MPO = MPO """
    
    if isinstance(tensor1, mps) and isinstance(tensor2, mps):
        # Perform the dot product where we tranpose tensor1
        return dot(tensor1, tensor2)
    elif isinstance(tensor1, mpo) and isinstance(tensor2, mpo):
        # If we have an mpo of bond dimension D1 and mpo of bond dimension D2
        # then we get an mpo of bond dimension D1*D2
        return operator_product(tensor1, tensor2)
    elif isinstance(tensor1, mpo) and isinstance(tensor2, mps):
        # If we have an mpo of bond dimension D1 and mps of bond dimension D2
        # then we get an mps of bond dimension D1*D2
        return apply_operator(tensor1, tensor2)
    elif isinstance(tensor1, mps) and isinstance(tensor2, mpo):
        # If we have an mps of bond dimension D1 and mpo of bond dimension D2
        # then we get an mps of bond dimension D1*D2. We tranpose the mps
        return apply_operator(tensor1, tensor2, conjugate=True)


def dot(mps1, mps2):
    """ Takes two MPS's and calculates their inner product."""
    
    # Retrieve the first sites, and contract over the physical indices
    A1 = mps1.get_matrix(0)
    A1 = np.conj(np.reshape(A1, (mps1.dim, np.size(A1, axis=2))))
    A2 = mps2.get_matrix(0)
    A2 = np.reshape(A2, (mps2.dim, np.size(A2, axis=2)))
    prod = np.tensordot(A1, A2, axes=(0, 0))
    
    # Now contract with further matrices recursively
    for site in range(mps1.length - 1):
        # Get next matrices
        A1 = np.conj(mps1.get_matrix(site+1))
        A2 = mps2.get_matrix(site+1)        
        
        # Contract with them
        prod = np.tensordot(prod, A1, axes=(0, 0))
        prod = np.tensordot(prod, A2, axes=(0, 0))
        
        # Trace over physical indices
        prod = np.trace(prod, axis1=0, axis2=2)
    # Return a d^L vector
    return np.sum(prod)


def apply_operator(tensor1, tensor2, conjugate=False):
    """ Calculates the product between a tensor and an opeator. If conjugate
    is set to false, the tensor order must be mpo then mps. If true, the order
    is mps then mpo and we contract over different physical indices.
    
    -------------------Needs updating!---------------------------------"""
    
    # Define a new MPS, with bond dimension D1 x D2
    dim = tensor1.dim
    length = tensor1.length
    bond_dim = tensor1.bond_dim * tensor2.bond_dim
    MPS = mps(dim, length, bond_dim)
    
    # Loop through each site
    for site in range(length):
        # Retrieve the sites, and contract over physical indices
        if conjugate == False:
            M = tensor1.get_matrix(site)
            A = tensor2.get_matrix(site)
        else:
            # We also swap the axes so we're contracting over the correct ones
            M = np.swapaxes(tensor2.get_matrix(site), axis1=1, axis2=2)
            A = tensor1.get_matrix(site)
        
        # Take the product
        prod = np.tensordot(A, M, axes=(1, 1))
                
        # Swap axes and reshape
        prod = np.swapaxes(prod, axis1=1, axis2=2)
        prod = np.swapaxes(prod, axis1=2, axis2=3)
        prod = group_indices(prod, 3, 4)
        prod = group_indices(prod, 0, 1)
        
        # Update it to MPS
        #MPS.edit_structure(site, prod)
        MPS.structure[site] = prod
        
    return MPS
                

def operator_product(mpo1, mpo2):
    """ Takes two MPOs and computes their product """
    
    # Define a new MPO, with a bond dimension of D1 x D2
    dim = mpo1.dim
    length = mpo1.length
    bond_dim = mpo1.bond_dim * mpo2.bond_dim
    MPO = mpo(dim, length, bond_dim)
    
    # Loop through each site
    for site in range(length):
        # Get each site
        M1 = mpo1.get_matrix(site)
        M2 = mpo2.get_matrix(site)
        
        # Take the product
        prod = np.tensordot(M1, M2, axes=(2, 1))
        
        # Group bond dimensions together
        prod = moveindex(prod, 3, 1)
        prod = group_indices(prod, 0, 1)
        prod = moveindex(prod, 2, 3)
        prod = group_indices(prod, 3, 4) 
        
        MPO.edit_structure(site, prod)
    
    return MPO


def expectation(MPO, MPS):
    """ Calculates some expectation value of some MPO w.r.t some MPS """
    
    # Retrieve the first sites and resize them
    M = np.resize(MPO.get_matrix(0), (MPO.dim, MPO.dim, MPO.bond_dim))
    A = MPS.get_matrix(0)
    A = np.resize(A, (MPS.dim, np.size(A, axis=2)))
    
    # Take contractions
    prod = np.tensordot(A, M, axes=(0, 0))
    prod = np.tensordot(prod, np.conj(A), axes=(1, 0))
    
    # Loop through the remaining sites
    for site in range(MPO.length-1):
        # Retrieve the sites
        M = MPO.get_matrix(site+1)
        A = MPS.get_matrix(site+1)
        
        # Contract with first MPS
        prod = np.tensordot(prod, A, axes=(0, 0))
        # Contract with MPO
        prod = np.tensordot(prod, M, axes=(0, 0))
        # Take the trace
        prod = np.trace(prod, axis1=1, axis2=3)
        # Contract with second MPS
        prod = np.tensordot(prod, np.conj(A), axes=(0, 0))
        # Take the trace
        prod = np.trace(prod, axis1=1, axis2=3)
    
    return np.sum(prod)


def reverse_mps(MPS):
    # Make a copy of the MPS
    mps2 = copy.deepcopy(MPS)
    
    # Get mps size
    length = MPS.length
    
    # Loop through each site
    for i in range(length):
        # Get the tensor
        A = MPS.get_matrix(i)
        
        # Reverse the indices
        A = moveindex(A, 0, 2)
        A = moveindex(A, 1, 0)
        
        # Update structure
        mps2.edit_structure(length - 1 - i, A)
        
    return mps2


def add_mps(MPS1, MPS2):
    # Create a new MPS
    MPS3 = mps(MPS1.dim, MPS1.length, MPS1.bond_dim + MPS2.bond_dim)
    
    for i in range(MPS1.length):
        M1 = MPS1.get_matrix(i)
        M2 = MPS2.get_matrix(i)
        
        (D1, d, D2) = np.shape(M1)
        (D3, d, D4) = np.shape(M2)
        if i == 0:
            tensor = np.zeros((1, d, D2+D4))
            tensor[0:1, :, 0:D2] = M1
            tensor[0:1, :, D2:D2+D4] = M2
        elif i == MPS1.length - 1:
            tensor = np.zeros((D1+D3, d, 1))
            tensor[0:D1, :, 0:1] = M1
            tensor[D1:D1+D3, :, 0:1] = M2
        else:
            tensor = np.zeros((D1+D3, d, D2+D4))
            tensor[0:D1, :, 0:D2] = M1
            tensor[D1:D1+D3, :, D2:D2+D4] = M2
        
        MPS3.structure[i] = tensor
        
    return MPS3
            
            

def effective_operator_add_left_block(left, operator, state):
    """ For an effective operator, we add one site to the left block.
    Inputs:
        left = the left block we're adding too, a (D, Do, D) tensor
        operator = the operator matrix at site i to add. (Do, d, d, Do) tensor
        state = the state matrix at site i to add. (D, d, D) tensor
    Output:
        newleft = new left block, a (D, Do, D) tensor """
    newleft = np.tensordot(left, state, axes=(0, 0))  # (Do, D, d, D)
    newleft = np.tensordot(newleft, operator, axes=(0, 0)) # (D, d, D, d, d, Do)
    newleft = np.trace(newleft, axis1=1, axis2=3) # (D, D, d, Do)
    newleft = np.tensordot(newleft, np.conj(state), axes=(0, 0)) # (D, d, Do, d, D)
    newleft = np.trace(newleft, axis1=1, axis2=3)
    #newleft[np.abs(newleft) < 10**-10] = 0
    return newleft


def effective_operator_add_right_block(right, operator, state):
    """ For an effective operator, we add one site to the right block.
    Inputs:
        right = the right block we're adding too, a (D, Do, D) tensor
        operator = the operator matrix at site i to add. (Do, d, d, Do) tensor
        state = the state matrix at site i to add. (D, d, D) tensor
    Output:
        newright= new right block, a (D, Do, D) tensor """
        
    newright = np.tensordot(state, right, axes=(2, 0))  # (D, d, Do, D)
    newright = np.tensordot(operator, newright, axes=(3, 2)) # (Do, d, d, D, d, D)
    newright = np.trace(newright, axis1=1, axis2=4) # (Do, d, D, D)
    newright = np.tensordot(np.conj(state), newright, axes=(2, 3)) # (D, d, Do, d, D)
    newright = np.trace(newright, axis1=1, axis2=3) # (D, Do, D)
    #newright[np.abs(newright) < 10**-10] = 0
    return newright


def effective_operator(site, MPO, MPS, left, right, direction):
    """ Will update the left and right blocks at the appropiate points for the
    given site we're optimizing. It will then construct the effective operator.
    Inputs:
        site = Which site we're optimizing
        mpo = the operator we're calculating an effect operator for
        mps = the mps we're optimizing
        left = the left blocks up to this point
        right = the right blocks up to this point
        direction = which way we're sweeping. 0 is right, 1 is left
    outputs:
        left = the new, updated left blocks
        right = the new, updated right blocks
        Oeff = the effective operator acting on the site"""
        
    # Retrieve properties of the system
    length = MPS.length
    dim = MPS.dim
    
    # Calculate the updates to the relavent blocks
    if site == 1 and direction == 0:
        # We're at the second site, and we need to update the first left block
        # There is not block to contract the first sites with, so we calculate
        # it explicitly
        A = MPS.get_matrix(0)
        A = np.reshape(A, (dim, np.size(A, axis=2)))
        M = np.reshape(MPO.get_matrix(0), (dim, dim, MPO.bond_dim))
        
        # Contract sites
        leftblock = np.tensordot(A, M, axes=(0, 0))
        leftblock = np.tensordot(leftblock, np.conj(A), (1, 0))
        
        # Update the block
        left[0] = leftblock
    elif site == length - 2 and direction == 1:
        # We're at the second to last site, and we need to update the
        # last right block. There is not block to contract the first sites
        # with, so we calculate it explicitly.
        A = MPS.get_matrix(length - 1)
        A = np.reshape(A, (np.size(A, axis=0), dim))        
        M = np.reshape(MPO.get_matrix(length - 1), (MPO.bond_dim, dim, dim))
        
        # Contract first sites; ends in (Ds, Do, Ds) form
        rightblock = np.tensordot(A, M, axes=(1, 1))
        rightblock = np.tensordot(rightblock, np.conj(A), (2, 1))
        
        # Update the block
        right[length - 1] = rightblock
    elif direction == 0 and site != 0:
        # In a middle site and moving right
        left[site - 1] = effective_operator_add_left_block(left[site - 2],
             MPO.get_matrix(site - 1), MPS.get_matrix(site - 1))
    elif direction == 1 and site == length - 1:
        left[site - 1] = effective_operator_add_left_block(left[site - 2],
             MPO.get_matrix(site - 1), MPS.get_matrix(site - 1))
    elif direction == 1 and site != length - 1:
        # In a middle site and moving left
        right[site + 1] = effective_operator_add_right_block(right[site + 2],
             MPO.get_matrix(site + 1), MPS.get_matrix(site + 1))
    elif direction == 0 and site == 0:
        right[site + 1] = effective_operator_add_right_block(right[site + 2],
             MPO.get_matrix(site + 1), MPS.get_matrix(site + 1))
            
    # Now calculate the effective operator acting on the site by joining the
    # blocks with the remaining operator block
    if site == 0:        
        # Create a linear opeator which joins the blocks with the site and
        # MPO
        
        fun = lambda v: join_eff_op_blocks_right(v, right[1], MPO.get_matrix(0))
        D = MPS.dim * np.size(right[1], axis=2)
        Oeff = LinearOperator((D, D), matvec=fun)
    elif site == length - 1:        
        fun = lambda v: join_eff_op_blocks_left(v, left[site-1], MPO.get_matrix(site))
        D = MPS.dim * np.size(left[site-1], axis=0)
        Oeff = LinearOperator((D, D), matvec=fun)
    else:
        fun = lambda v: join_eff_op_blocks(v, left[site-1], right[site+1], MPO.get_matrix(site))
        D = MPS.dim * np.size(right[site+1], axis=0)*np.size(left[site-1], axis=0)
        Oeff = LinearOperator((D, D), matvec=fun)
    return left, right, Oeff



def effective_vector_add_left_block(left, constraint, state):
    """ For an effective vector, we add one site to the left block.
    Inputs:
        left = the left block we're adding too, a (D, Dc) tensor
        constraint = the constraint matrix at site i to add. (Dc, d, Dc) tensor
        state = the state matrix at site i to add. (D, d, D) tensor
    Output:
        newleft = new left block, a (D, Dc) tensor """
     
    newleft = np.tensordot(left, state, axes=(0, 0))
    newleft = np.tensordot(newleft, np.conj(constraint), axes=(0, 0))
    newleft = np.trace(newleft, axis1=0, axis2=2)
    #newleft[np.abs(newleft) < 10**-10] = 0
    return newleft 


def effective_vector_add_right_block(right, constraint, state):
    """ For an effective vector, we add one site to the right block.
    Inputs:
        right = the right block we're adding too, a (D, Dc) tensor
        constraint = the constraint matrix at site i to add. (Dc, d, Dc) tensor
        state = the state matrix at site i to add. (D, d, D) tensor
    Output:
        newright = new right block, a (D, Dc) tensor """
    newright = np.tensordot(np.conj(constraint), right, axes=(2, 1))
    newright = np.tensordot(state, newright, axes=(2, 2))
    newright = np.trace(newright, axis1=1, axis2=3)
    #newright[np.abs(newright) < 10**-10] = 0
    return newright


def effective_vector(site, constraint, MPS, left, right, direction):
    """ Will update the left and right blocks at the appropiate points for the
    given site we're optimizing. It will then construct the effective vector.
    Inputs:
        site = Which site we're optimizing
        constraint = the constraint we're calculating an effect vector for
        mps = the mps we're optimizing
        left = the left blocks up to this point
        right = the right blocks up to this point
        direction = which way we're sweeping. 0 is right, 1 is left
    outputs:
        left = the new, updated left blocks
        right = the new, updated right blocks
        Ceff = the effective vector acting on the site"""
    
    # Retrieve properties of the system
    length = MPS.length
    dim = MPS.dim
        
    # Calculate the updates to the relavent blocks
    if site == 1 and direction == 0:
        # We update the first left block, which is just a manual contraction
        # of tensors
        A = MPS.get_matrix(0)
        A = np.reshape(MPS.get_matrix(0), (dim, np.size(A, axis=2)))
        B = constraint.get_matrix(0)
        B = np.reshape(B, (dim, np.size(B, axis=2)))
        
        # Contract sites
        leftblock = np.tensordot(A, np.conj(B), axes=(0, 0))
        
        # Update the block
        left[0] = leftblock
    elif site == length - 2 and direction == 1:
        # We update the first right block, which is just a manual contraction
        # of tensors
        A = MPS.get_matrix(length - 1)
        A = np.reshape(A, (np.size(A, axis=0), dim))
        B = constraint.get_matrix(length - 1)
        B = np.reshape(B, (np.size(B, axis=0), dim))
        
        # Contract sites
        rightblock = np.tensordot(A, np.conj(B), axes=(1, 1))
        
        # Update the block
        right[length - 1] = rightblock
    elif direction == 0 and site != 0:
        # We're in a middle site, moving right; update the last left block
        left[site - 1] = effective_vector_add_left_block(left[site-2],
                         constraint.get_matrix(site-1),
                         MPS.get_matrix(site-1))     
    elif direction == 0 and site == 0:
        right[site + 1] = effective_vector_add_right_block(right[site+2],
                         constraint.get_matrix(site+1),
                         MPS.get_matrix(site+1))    
    elif direction == 1 and site != length - 1:
        # We're in a middle site moving left; update the last right block
        right[site + 1] = effective_vector_add_right_block(right[site+2],
                         constraint.get_matrix(site+1),
                         MPS.get_matrix(site+1))   
    elif direction == 1 and site == length - 1:
        left[site - 1] = effective_vector_add_left_block(left[site-2],
                         constraint.get_matrix(site-1),
                         MPS.get_matrix(site-1)) 
    
    # We calculate the effective vector by joining left and right blocks with
    # the remaining constraint site
    if site == 0:
        # Contract the right block with the remaining constraint site at 0
        B = constraint.get_matrix(0)
        B = np.reshape(B, (dim, np.size(B, axis=2)))
        prod = np.tensordot(np.conj(B), right[1], axes=(1, 1))
        prod = np.reshape(prod, (1, dim*np.size(right[1], axis=0)))
        
        fun = lambda v: join_eff_vec_blocks(v, prod)
        D = dim*np.size(right[1], axis=0)
        Ceff = LinearOperator((D, D), matvec=fun)
    elif site == length - 1:
        # Contract the left block with the remaining constraint site at L - 1
        B = constraint.get_matrix(site)
        B = np.reshape(B, (np.size(B, axis=0), dim))
        prod = np.tensordot(left[site - 1], np.conj(B), axes=(1, 0))
        prod = np.reshape(prod, (1, np.size(left[site - 1], axis=0)*dim))
        
        fun = lambda v: join_eff_vec_blocks(v, prod)
        D = np.size(left[site - 1], axis=0)*dim
        Ceff = LinearOperator((D, D), matvec=fun)
    else:
        # Contract the left and right block with the remaining site
        B = constraint.get_matrix(site)
        prod = np.tensordot(left[site-1], np.conj(B), axes=(1, 0))
        prod = np.tensordot(prod, right[site+1], axes=(2, 1))
        prod = np.reshape(prod, (1, np.size(left[site - 1], axis=0)*dim*
                                 np.size(right[site + 1], axis=0)))
        
        fun = lambda v: join_eff_vec_blocks(v, prod)
        D = np.size(left[site - 1], axis=0)*dim*np.size(right[site + 1], axis=0)
        Ceff = LinearOperator((D, D), matvec=fun)
    return left, right, Ceff


def initiate_effective_operator(MPO, MPS):
    """ Creates the correct shapes for the left and right blocks of an 
    effective operator, and calculates all the right blocks at each point.
    Inputs: 
        MPO = the mpo to calculate the blocks for
        mps = the mps to caluclate the blocks for
    """
    
    # Get the system properties
    length = MPS.length
    dim = MPS.dim
    
    # Store the left and right blocks
    left = []
    right = []
    
    # Create the structure for the blocks
    for site in range(length):
        A = MPS.get_matrix(site)
        M = MPS.get_matrix(site)
        
        left.append([np.zeros((np.size(A, axis = 2), MPO.bond_dim,
                              np.size(A, axis = 2)))])
        right.append([np.zeros((np.size(A, axis = 0), MPO.bond_dim,
                              np.size(A, axis = 0)))])
    
    # Starting from the right, calculate the right block for each
    for site in range(length - 1):
        # Retrieve the sites
        A = MPS.get_matrix(length - 1 - site)
        M = MPO.get_matrix(length - 1 - site)
        
        if site == 0:
            A = np.reshape(A, (np.size(A, axis=0), dim))
            M = np.reshape(M, (MPO.bond_dim, dim, dim))
            
            # Contract first sites; ends in (Ds, Do, Ds) form
            rightblock = np.tensordot(A, M, axes=(1, 1))
            rightblock = np.tensordot(rightblock, np.conj(A), (2, 1))
            
            right[length - 1] = rightblock
        else:
            right[length - 1 - site] = effective_operator_add_right_block(
                    right[length - site], M, A)
            
    return left, right 


def initiate_effective_vector(constraint, MPS):
    """ Creates the correct shapes for the left and right blocks of an 
    effective vector, and calculates all the right blocks at each point.
    Inputs: 
        constraint = the constraint to calculate the blocks for
        mps = the mps to caluclate the blocks for
    """
    
    # Get the system properties
    length = MPS.length
    dim = MPS.dim
    
    # Store the left and right blocks
    left = []
    right = []
    
    # Create the structure for the blocks
    for site in range(length):
        A = MPS.get_matrix(site)
        B = constraint.get_matrix(site)
        
        left.append([np.zeros((np.size(A, axis = 2), np.size(B, axis = 2)))])
        right.append([np.zeros((np.size(A, axis = 0), np.size(B, axis = 0)))])
    
    # Starting from the right, calculate the right block for each
    for site in range(length - 1):
        # Retrieve the sites
        A = MPS.get_matrix(length-1-site)
        B = constraint.get_matrix(length-1-site)
        
        if site == 0:
            # Reshape out last dimension
            A = np.reshape(A, (np.size(A, axis=0), dim))
            B = np.reshape(B, (np.size(B, axis=0), dim))
            
            # Contract the first sites; ends up in (D, Dc) form
            rightblock = np.tensordot(A, np.conj(B), axes=(1, 1))
            
            # Update the site
            right[length - 1] = rightblock
        else:
            right[length - 1 - site] = effective_vector_add_right_block(
                                       right[length - site], B, A)
    return left, right


def move_canonical(MPS, site, direction):
    # Get the matrix we wish to normalize
    M = MPS.get_matrix(site)
    
    D1, D2, D3 = np.shape(M)
    
    if direction == 0:
        M = np.reshape(M, (D1*D2, D3))
        U, S, V = svd(M)
        s = np.size(S, axis=0)
        U = np.reshape(U, (D1, D2, s))
        D = np.tensordot(np.matmul(S, V), MPS.get_matrix(site+1), axes=(1, 0))
        MPS.edit_structure(site, U)
        MPS.edit_structure(site+1, D)
    elif direction == 1:
        M = np.reshape(M, (D1, D2*D3))
        U, S, V = svd(M)
        s = np.size(S, axis=0)
        V = np.reshape(V, (s, D2, D3))
        D = np.tensordot(MPS.get_matrix(site-1), np.matmul(U, S), axes=(2, 0))
        MPS.edit_structure(site, V)
        MPS.edit_structure(site-1, D)
        
    
    return MPS   


def sector_operator(length):
    I = np.identity(2)
    n = np.array([[0, 0], [0, 1]])
    
    O = np.zeros(((3, 2, 2, 3)))
    O[0, :, :, 0] = I
    O[1, :, :, 0] = n
    O[2, :, :, 1] = I - n
    O[2, :, :, 2] = I
    
    O1 = np.copy(O[-1:, :, :, :])
    O1[0:, :, :, 0] = n
    
    sec = mpo(2, length, 3, O)
    sec.edit_structure(0, O1)     
    
    return sec
            
 
def variational_sweep(H, initial, constraint = None,
                      constraint_multiplier = 100, epsilon=10**-10,
                      min_sweeps=2, max_sweeps = 100, k=1):
    """ Searches for the ground, or excited states of a given hamiltonian. It
    sweeps through each site in a MPS, and optimizes by calculating an
    effective Hamiltonian on that site, and diagonalizes it. We can use
    previous calculated states as constraints to search for higher states.
    Inputs:
        H = Hamiltonian
        initial = Initial MPS guess
        constraint = states to not search for (optionial)
        epsilon = Uncertainity excpetion in varience (optional)
        min_sweeps = The minimum amount of times to sweep through the sites
        k = the number of eigenvalues to find for each optimization. k=1 is
            recommended for finding ground states. k = 5 is recommended for
            finding excited states.
    Outputs:
        E = Energy of state
        state = ground (or excited) state """
    
    # Print a message to let the user know the variational sweep has begun
    print('----------Calculating the energy using variational sweep----------')
    
    # Make a copy of the initial state
    psi = copy.deepcopy(initial)
    if constraint is not None:
        const = copy.deepcopy(constraint)
        
    # If constraint multiplier is zero, change it to 1
    if np.abs(constraint_multiplier) < 10**-10:
        constraint_multiplier = 1
        
    
    # Get parameters of system
    length = H.length
    dim = H.dim
    
    # Make a list of one sweep; it starts from the left, moves to the right, 
    # and back to the start
    sites1 = np.linspace(0, length - 2, length-1, dtype='int')
    sites2 = np.linspace(length-1, 1, length-1, dtype='int')
    
    # Define H^2 operator
    H2 = product(H, H)
    
    # Put the initial state into left-canonical
    try:
        psi.right_canonical()
    except Exception:
        print("Couldn't move into right canonical, continue anyway")
    
    # Initiate left and right blocks
    left, right = initiate_effective_operator(H, psi)
    # If there is a constraint, initiate the left and right blocks
    if constraint is not None:
        print('yes')
        const_left, const_right = initiate_effective_vector(const, psi)
    
    
    # Dummie variable for the varience
    variance = epsilon
    
    # Loop through sweeps until we reach a convergence in variance
    sweeps = 0  # Keep track of the number of sweeps
    Hexp = 0
    nochange = 0
    
    direction = 0
    cont = True
    convergence = False
    while cont or sweeps < min_sweeps:
        if direction == 0:
            sites = sites1
        else:
            sites = sites2
        # Loop through each site in our list
        for site in sites:    
            # Calculate the effective hamiltonian
            left, right, Heff = effective_operator(site, H, psi, left, right,
                                                   direction)
                        
            # If there is a constraint, calculate the effective vector
            if constraint is not None:
                # Calculate the effect vector
                const_left, const_right, Ceff = effective_vector(site,
                                                const, psi, const_left,
                                                const_right, direction)
                
                
                # Add this to the hamiltonian
                Heff = Heff + constraint_multiplier * Ceff
                #Heff = Ceff
            
            # Get the old site as an initial guess
            oldA  = psi.get_matrix(site)
            D1, D2, D3 = np.shape(oldA)
            oldA = np.resize(oldA, (D1*D2*D3, 1))
            

            try:
                # Calculate the eigenvalues and eigenvectors of Heff
                eig, vec = eigsh(Heff, which='SA', k = 1, v0=oldA)
                
                # Make them real as H is hermitian (they will be real, but python
                # has slight error which will make them look complex)
                eig = np.real(eig)
                vec = np.real(vec)
                
                # Find the lowest eigenvalue
                idx = np.argmin(eig)
                
                # Calculate the new update for this site
                newA = vec[:, idx]
                    
                # Re-size it appropiately
                if site == 0:
                    newA = np.reshape(newA, (D1, D2, D3))
                elif site == length - 1:
                    newA = np.reshape(newA, (D1, D2, D3))
                else:
                    newA = np.reshape(newA, (D1, D2, D3))
                
                # Update the site
                psi.edit_structure(site, newA)
            except Exception:
                print('eigs couldnt converge; move onto next site')
                pass
            
            # Move into mixed canonical i.e. move the normalisation right/left
            psi = move_canonical(psi, site, direction)
            
        # Reverse the list of sites and direction
        direction = np.abs(direction - 1)
        
        # Calculate the variance of the energy
        Hexp = expectation(H, psi)
        H2exp = expectation(H2, psi)
        lastvariance = variance
        variance = (H2exp / (Hexp**2)) - 1
        if sweeps != 0:
            relativediff = np.abs(variance-lastvariance) / variance
            if relativediff < 0.01:
                nochange += 1
                if nochange >= 3:
                    convergence = True
            else:
                nochange = 0
        var = H2exp - Hexp**2
        
        sweeps += 1
        cont = variance > epsilon and sweeps < max_sweeps and not convergence
        cont = cont and var > 10**-20
        #if sweeps % 10 == 0:
            #epsilon = epsilon * 10
        print('Sweep ' + str(sweeps) + ': The energy is ' + str(Hexp) + 
              ' with varience ' + str(variance))
        
    Hexp = expectation(H, psi)
    H2exp = expectation(H2, psi)
    Hvar = H2exp - Hexp**2
    
    return [psi, Hexp, Hvar]
    
def join_eff_op_blocks(vector, left, right, matrix):
    vector = np.reshape(vector, (np.size(left, axis=2), np.size(matrix, axis=1), np.size(right, axis=2)))
    prod = np.tensordot(left, vector, axes=(2, 0)) 
    prod = np.tensordot(prod, matrix, axes=(1, 0))
    prod = np.trace(prod, axis1=1, axis2=3)
    prod = np.tensordot(prod, right, axes=(1, 0))
    prod = np.trace(prod, axis1=2, axis2=3)
    D1, D2, D3 = np.shape(prod)
    return np.reshape(prod, (D1*D2*D3, 1))


def join_eff_op_blocks_right(vector, right, matrix):
    vector = np.reshape(vector, (np.size(matrix, axis=1), np.size(right, axis=2)))
    matrix = np.reshape(matrix, (np.size(matrix, axis=1),np.size(matrix, axis=2),
                                 np.size(matrix, axis=3)))
    prod = np.tensordot(vector, right, axes=(1, 0))
    prod = np.tensordot(matrix, prod, axes=(2, 1))
    prod = np.trace(prod, axis1=1, axis2=2)
    D1, D2 = np.shape(prod)
    prod = np.reshape(prod, (D1*D2, 1))
    return prod


def join_eff_op_blocks_left(vector, left, matrix):
    vector = np.reshape(vector, (np.size(left, axis=2), np.size(matrix, axis=1)))
    prod = np.tensordot(left, vector, axes=(2, 0))
    prod = np.tensordot(prod, matrix, axes=(1, 0)) 
    prod = np.trace(prod, axis1=1, axis2=3)
    prod = np.reshape(prod, (np.size(left, axis=2)*np.size(matrix, axis=1), 1))
    return prod
    

def join_eff_vec_blocks(vector, prod):
    vector = np.reshape(vector, (np.size(vector), 1))
    prod = np.dot(prod, vector)*np.transpose(prod)
    return prod
        
