import numpy as np
import scipy.special as sp
import sys
import os
import lib.matrixproducts_FA as mp
import copy


    
# Calculate the occupation of each site
def occupation(MPS):
    # Get the number of sites
    sites = MPS.length
    
    # Store occupations
    occupations = np.zeros(sites)
    
    # Loop through each site
    for site in range(sites):
        # Create the MPO
        I = np.array([[1, 0], [0, 1]])
        n = np.array([[0, 0], [0, 1]])
        local = np.zeros((1, 2, 2, 1))
        local[0, :, :, 0] = I
        ni = mp.mpo(2, sites, 1, local)
        nlocal = np.zeros((1, 2, 2, 1))
        nlocal[0, :, :, 0] = n
        ni.edit_structure(site, nlocal)
        
        # Calculate occupation
        occupations[site] = mp.expectation(ni, MPS)
    
    return occupations / mp.dot(MPS, MPS)

# Calculate the average excitation density
def excitation(MPS):
    #Get the number of sites
    sites = MPS.length

    # Create the MPO
    I = np.array([[1, 0], [0, 1]])
    n = np.array([[0, 0], [0, 1]])
    local = np.zeros((2, 2, 2, 2))
    local[0, :, :, 0] = I
    local[1, :, :, 0] = n
    local[1, :, :, 1] = I
    ni = mp.mpo(2, sites, 2, local)
        
    # Calculate excitation
    excitation = mp.expectation(ni, MPS) / sites
    
    return excitation / mp.dot(MPS, MPS)


def correlation(MPS, distance=1):
    # Get the number of sites 
    sites = MPS.length
    
    # Operators
    I = np.array([[1, 0], [0, 1]])
    n = np.array([[0, 0], [0, 1]])
    
    # Find the amount of pairs
    pairs = sites - distance
    
    # Store correlation
    correlation = 0
    
    # Loop through each pair
    for site in range(pairs):
        # Create an MPO
        local = np.zeros((1, 2, 2, 1))
        local[0, :, :, 0] = I
        mpo = mp.mpo(2, sites, 1, local)
        
        # Add the n operator to the correct sites
        nlocal = np.zeros((1, 2, 2, 1))
        nlocal[0, :, :, 0] = n
        mpo.edit_structure(site, nlocal)
        mpo.edit_structure(site+distance, nlocal)
        
        # Calculate the expectation
        correlation += mp.expectation(mpo, MPS)
    
    return correlation / pairs


def correlation_disconnected(MPS, distance=1):
    # Get the number of sites 
    sites = MPS.length
    
    norm = mp.dot(MPS, MPS)
    # Operators
    I = np.array([[1, 0], [0, 1]])
    n = np.array([[0, 0], [0, 1]])
    
    # Find the amount of pairs
    pairs = sites - distance
    
    # Store correlation
    correlation = 0
    
    # Get occupations
    occupations = occupation(MPS)
    
    # Loop through each pair
    for site in range(pairs):
        # Create an MPO
        local = np.zeros((1, 2, 2, 1))
        local[0, :, :, 0] = I
        mpo = mp.mpo(2, sites, 1, local)
        
        # Add the n operator to the correct sites
        nlocal = np.zeros((1, 2, 2, 1))
        nlocal[0, :, :, 0] = n
        mpo.edit_structure(site, nlocal)
        mpo.edit_structure(site+distance, nlocal)
        
        # Calculate the expectation
        correlation += mp.expectation(mpo, MPS) / norm
        
        correlation -= occupations[site]*occupations[site+distance]
    
    return correlation / pairs


# Calculate the DW occupation of each site
def DW_occupation(MPS):
    # Get the number of sites
    sites = MPS.length
    
    # Store occupations
    occupations = np.zeros(sites + 1)
    
    norm = mp.dot(MPS, MPS)
    
    # Local operators
    I = np.array([[1, 0], [0, 1]])
    n = np.array([[0, 0], [0, 1]])
    
    
    # Loop through each site
    for site in range(sites + 1):
        if site == 0:
            # Create local identity operator
            local = np.zeros((1, 2, 2, 1))
            local[0, :, :, 0] = I
            
            # Create local n operator
            nlocal1 = np.zeros((1, 2, 2, 1))
            nlocal1[0, :, :, 0] = n 
            
            # Create MPO 
            ni = mp.mpo(2, sites, 1, local)
            ni.edit_structure(site, nlocal1)
            
            # Calculate occupation
            occupations[site] = mp.expectation(ni, MPS)
        elif site == sites:
            # Create local identity operator
            local = np.zeros((1, 2, 2, 1))
            local[0, :, :, 0] = I
            
            # Create local n operator
            nlocal1 = np.zeros((1, 2, 2, 1))
            nlocal1[0, :, :, 0] = n 
            
            # Create MPO 
            ni = mp.mpo(2, sites, 1, local)
            ni.edit_structure(site-1, nlocal1)
            
            # Calculate occupation
            occupations[site] = mp.expectation(ni, MPS)
        else:
            # Create local identity operator
            local = np.zeros((1, 2, 2, 1))
            local[0, :, :, 0] = I
            
            # Create local n operator
            nlocal1 = np.zeros((1, 2, 2, 1))
            nlocal1[0, :, :, 0] = n 
            
            # Create local 1-n operator
            nlocal2 = np.zeros((1, 2, 2, 1))
            nlocal2[0, :, :, 0] = I - n 
            
            # Create MPO 
            ni1 = mp.mpo(2, sites, 1, local)
            ni1.edit_structure(site-1, nlocal1)
            ni1.edit_structure(site, nlocal2)
            
            ni2 = mp.mpo(2, sites, 1, local)
            ni2.edit_structure(site-1, nlocal2)
            ni2.edit_structure(site, nlocal1)
            
            # Calculate occupation
            occupations[site] += mp.expectation(ni1, MPS)
            occupations[site] += mp.expectation(ni2, MPS)
            
    
    return occupations / norm

def DW_correlation_disconnected(MPS, distance=1):
    # Get the number of sites 
    sites = MPS.length
    
    norm = mp.dot(MPS, MPS)
    
    # Operators
    I = np.array([[1, 0], [0, 1]])
    n = np.array([[0, 0], [0, 1]])
    
    # Find the amount of pairs
    pairs = sites + 1 - distance
    
    # Store correlation
    correlation = 0
    
    # Get occupations
    occupations = DW_occupation(MPS)
    
    # Loop through each pair
    for site in range(pairs):
        if distance == 1:
            if site == 0:
                # Create an MPO to measure
                local = np.zeros((1, 2, 2, 1))
                local[0, :, :, 0] = I
                mpo = mp.mpo(2, sites, 1, local)
            
                site1 = np.zeros((1, 2, 2, 1))
                site1[0, :, :, 0] = n
                
                site2 = np.zeros((1, 2, 2, 1))
                site2[0, :, :, 0] = I - n
                
                mpo.edit_structure(site, site1)
                mpo.edit_structure(site+1, site2)
            elif site == sites - distance:
                # Create an MPO to measure
                local = np.zeros((1, 2, 2, 1))
                local[0, :, :, 0] = I
                mpo = mp.mpo(2, sites, 1, local)
            
                site1 = np.zeros((1, 2, 2, 1))
                site1[0, :, :, 0] = I - n
                
                site2 = np.zeros((1, 2, 2, 1))
                site2[0, :, :, 0] = n
                
                mpo.edit_structure(site-1, site1)
                mpo.edit_structure(site, site2)
            else:
                # Create an MPO to measure
                local = np.zeros((2, 2, 2, 2))
                local[0, :, :, 0] = I
                local[1, :, :, 1] = I
                mpo = mp.mpo(2, sites, 2, local)
                
                if site == 1:
                    site1 = np.zeros((1, 2, 2, 2))
                    site1[0, :, :, 0] = n
                    site1[0, :, :, 1] = I - n
                else:
                    site1 = np.zeros((2, 2, 2, 2))
                    site1[0, :, :, 0] = n
                    site1[1, :, :, 1] = I - n
                
                site2 = np.zeros((2, 2, 2, 2))
                site2[0, :, :, 0] = I - n
                site2[1, :, :, 1] = n
                
                if site == sites - 1 - distance:
                    site3 = np.zeros((2, 2, 2, 1))
                    site3[0, :, :, 0] = n
                    site3[1, :, :, 0] = I - n
                else:
                    site3 = np.zeros((2, 2, 2, 2))
                    site3[0, :, :, 0] = n
                    site3[1, :, :, 1] = I - n
                
                if site != 1:
                    local = np.zeros((1, 2, 2, 2))
                    local[0, :, :, 0] = I
                    local[0, :, :, 1] = I
                    mpo.edit_structure(0, local)
                if site != sites - 1 - distance:
                    local = np.zeros((2, 2, 2, 1))
                    local[0, :, :, 0] = I
                    local[1, :, :, 0] = I
                    mpo.edit_structure(sites-1, local)
                    
                mpo.edit_structure(site-1, site1)
                mpo.edit_structure(site, site2)
                mpo.edit_structure(site+1, site3)
                
                
        else:
            if site == 0:
                # Create an MPO to measure
                local = np.zeros((2, 2, 2, 2))
                local[0, :, :, 0] = I
                local[1, :, :, 1] = I
                mpo = mp.mpo(2, sites, 2, local)
            
                site1 = np.zeros((1, 2, 2, 2))
                site1[0, :, :, 0] = n
                site1[0, :, :, 1] = n
                
                site2 = np.zeros((2, 2, 2, 2))
                site2[0, :, :, 0] = I - n
                site2[1, :, :, 1] = n
                
                site3 = np.zeros((2, 2, 2, 2))
                site3[0, :, :, 0] = n
                site3[1, :, :, 1] = I - n
                
                site4 = np.zeros((2, 2, 2, 1))
                site4[0, :, :, 0] = I
                site4[1, :, :, 0] = I
                
                mpo.edit_structure(site, site1)
                mpo.edit_structure(site+distance-1, site2)
                mpo.edit_structure(site+distance, site3)
                mpo.edit_structure(sites-1, site4)
            elif site == sites - distance:
                # Create an MPO to measure
                local = np.zeros((2, 2, 2, 2))
                local[0, :, :, 0] = I
                local[1, :, :, 1] = I
                mpo = mp.mpo(2, sites, 2, local)
                
                site1 = np.zeros((2, 2, 2, 2))
                site1[0, :, :, 0] = n
                site1[1, :, :, 1] = I - n
                
                site2 = np.zeros((2, 2, 2, 2))
                site2[0, :, :, 0] = I - n
                site2[1, :, :, 1] = n
                
                site3 = np.zeros((2, 2, 2, 1))
                site3[0, :, :, 0] = n
                site3[1, :, :, 0] = n
                
                site4 = np.zeros((1, 2, 2, 2))
                site4[0, :, :, 0] = I
                site4[0, :, :, 1] = I
                
                mpo.edit_structure(0, site4)
                mpo.edit_structure(site-1, site1)
                mpo.edit_structure(site, site2)
                mpo.edit_structure(site-1+distance, site3)
            else:
                # Create an MPO to measure
                local = np.zeros((4, 2, 2, 4))
                local[0, :, :, 0] = I
                local[1, :, :, 1] = I
                local[2, :, :, 2] = I
                local[3, :, :, 3] = I
                mpo = mp.mpo(2, sites, 4, local)
                
                if site == 1:
                    site1 = np.zeros((1, 2, 2, 4))
                    site1[0, :, :, 0] = n
                    site1[0, :, :, 1] = I - n
                    site1[0, :, :, 2] = n
                    site1[0, :, :, 3] = I - n
                else:
                    site1 = np.zeros((4, 2, 2, 4))
                    site1[0, :, :, 0] = n
                    site1[1, :, :, 1] = I - n
                    site1[2, :, :, 2] = n
                    site1[3, :, :, 3] = I - n
                
                site2 = np.zeros((4, 2, 2, 4))
                site2[0, :, :, 0] = I - n
                site2[1, :, :, 1] = n
                site2[2, :, :, 2] = I - n
                site2[3, :, :, 3] = n
                
                site3 = np.zeros((4, 2, 2, 4))
                site3[0, :, :, 0] = n
                site3[1, :, :, 1] = I - n
                site3[2, :, :, 2] = I - n
                site3[3, :, :, 3] = n
                
                if site == sites - 1 - distance:
                    site4 = np.zeros((4, 2, 2, 1))
                    site4[0, :, :, 0] = I - n
                    site4[1, :, :, 0] = n
                    site4[2, :, :, 0] = n
                    site4[3, :, :, 0] = I - n
                else:
                    site4 = np.zeros((4, 2, 2, 4))
                    site4[0, :, :, 0] = I - n
                    site4[1, :, :, 1] = n
                    site4[2, :, :, 2] = n
                    site4[3, :, :, 3] = I - n
                
                
                if site != 1:
                    local = np.zeros((1, 2, 2, 4))
                    local[0, :, :, 0] = I
                    local[0, :, :, 1] = I
                    local[0, :, :, 2] = I
                    local[0, :, :, 3] = I
                    mpo.edit_structure(0, local)
                if site != sites - 1 - distance:
                    local = np.zeros((4, 2, 2, 1))
                    local[0, :, :, 0] = I
                    local[1, :, :, 0] = I
                    local[2, :, :, 0] = I
                    local[3, :, :, 0] = I
                    mpo.edit_structure(sites-1, local)
                
                mpo.edit_structure(site-1, site1)
                mpo.edit_structure(site, site2)
                mpo.edit_structure(site-1+distance, site3)
                mpo.edit_structure(site+distance, site4)
        
        
        correl = mp.expectation(mpo, MPS) / norm
        correlation += correl - occupations[site]*occupations[site+distance]
    return correlation / pairs