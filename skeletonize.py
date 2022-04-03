"""
A numba-based implementation of the Lee et. al 1994 medial axis thinning algorithm.
https://www.sci.utah.edu/devbuilds/biomesh3d/FEMesher/references/lee94-3dskeleton.pdf
"""

__author__    = 'Jacob Bumgarner <jrbumgarner@mix.wvu.edu>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright 2022 by Jacob Bumgarner'
__webpage__   = 'https://jacobbumgarner.github.io/VesselVio/'
__download__  = 'https://jacobbumgarner.github.io/VesselVio/Downloads'

import numpy as np
from numba import njit, prange, objmode

####################
### Binarization ###
####################
@njit(cache=False)
def binarize(volume):
    return (volume > 0).astype(np.uint8)

###########################
### Directional Filters ###
###########################
@njit(cache=False)
def init_filters():
    """Directional nomenclature derived from Lee '94 Fig. 8
    """
    North = [0,1,1]
    South = [2,1,1]
    East = [1,1,2]
    West = [1,1,0]
    Up = [1,0,1]
    Down = [1,2,1]
    # return np.array([North, South, East, West, Up, Down])-1
    return np.array([West, East, Down, Up, South, North])-1

######################
### Euler Variance ###
######################
@njit(cache=False)
def load_Euler_LUT():
    """See Table 2, column 3 of Lee '94
    """
    Euler_table = [1, -1, -1, 1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 
                   1, 1, -1, -3, -1, 3, 1, 1, -1, 3, 1, -1, 1, 1, 
                   -1, 3, 1, 1, -1, -3, 3, -1, 1, 1,3, -1, 1, -1, 
                   1, 1, -1, 3, 1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1, 
                   -1, 1, 1, -1, 3, 1, 1, -1, -7, -1, -1, 1, -3, 
                   -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, -1, 
                   3, 1, 1, -1, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1, 
                   -3, 3, -1, 1, 1, 3, -1, 1, -1, 1, 1, -1, 3, 1, 
                   1, -1, 1, 3, 3, 1, 5, 3, 3, 1, -1, 1, 1, -1, 3, 
                   1, 1, -1]
    LUT = np.zeros(256, np.int8)
    LUT[1::2] = Euler_table
    # LUT = np.array(Euler_table, dtype=np.int8)
    return LUT


def load_Euler_octants():
    """For an explanation of octants/Euler characteristics, 
        see Lee'94 and Lobrecht '80
    The octants are rotations of the cube with the ordering described in 
        Lee '94 and Lobrecht '80, such that cube[1,1,1] (v) is octant[1,1,1].
        IAC uses a different scheme. 
        To avoid confusion, I've stuck with the paper's numbering order
    """
    key = build_Euler_key()
    NEU = key[[8, 5, 16, 13, 7, 4, 15]]
    NWU = key[[2, 1, 11, 10, 5, 4, 13]]
    SEU = key[[6, 7, 14, 15, 3, 4, 12]]
    SWU = key[[0, 3, 9, 12, 1, 4, 10]]
    
    NEB = key[[25, 24, 16, 15, 22, 21, 13]]
    NWB = key[[19, 22, 11, 13, 18, 21, 10]]
    SEB = key[[23, 20, 14, 12, 24, 21, 15]]
    SWB = key[[17, 18, 9, 10, 20, 21, 12]]
    return np.array([SWU, NWU, SEU, NEU, SWB, NWB, SEB, NEB])

def build_Euler_key():
    """Create a key that corresponds to the numerical indices shown in the 
    cube in Lee '94
    """
    # Create the cube and orient it as shown in figure 1
    indices = np.arange(0,27)
    indices [14:] = indices[13:26]
    indices [13] = 0
    cube = indices.reshape([3,3,3])
    cube = np.rot90(cube, axes=(1,2))
    cube = np.rot90(cube, axes=(0,1))
    
    ## Dev
    ## Check to make sure that the flattened cube steps as expected
    # flat = cube.flatten()
    # neighborhood = flat.copy()
    # index = 0
    # for i in range(9):
    #     for k in range(3):
    #         neighborhood[index] = flat[i+k*9]
    #         index += 1
    
    # Ravel the cube, arg sort based on the flattened indices to get the key
    flat = cube.ravel()
    key = np.argsort(flat)
    key = np.delete(key, 1)
    return key
    
    # Could've just used np.argsort() lol
    ind = np.arange(0,27).tolist()
    indexed = zip(flat, ind)
    
    # Sort the indexed cube based on the flattened 0-26 indices. 
        # This way, when we flatten the neighborhood for Euler analysis, 
        # we can access the elements as described in the original ms, 
        # rather than trying to re-index everyting ourselves (hello bugs...)
    indexed = sorted(list(indexed), key=lambda x: x[0])
    # Sample of indexed...
        # (0, 0), (0, 13), (1, 9), (2, 18), (3, 1), (4, 10), (5, 19)
        # 0 at 0th index, 1 at 9th index, 2 at 18th index, etc.
    # Remove v (i.e., (0,13))from the cube to get N26(v) element locations
    del(indexed[1]) 
    key = np.array([e[1] for e in indexed])    
    return key

E_octant = load_Euler_octants()

#######################
### Octree Labeling ###
#######################
# Octree incidence for the N26v labeling - only label those that haven't been labeled yet. 
@njit(cache=False) 
def load_o_incidence():
    # All indices above 12 have a label += 1.
    # This prevents reshaping the N26v during labeling.
    # It abandons the naming convention used in the euler tree, but it saves time.
    o_1 = set([0, 1, 3, 4, 9, 10, 12])
    o_2 = set([2, 5, 11, 14])
    o_3 = set([6, 7, 15, 16])
    o_4 = set([8, 17])
    o_5 = set([18, 19, 21, 22])
    o_6 = set([20, 23])
    o_7 = set([24, 25])
    o_8 = set([26])
    return [o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8]

@njit(cache=False) # All indices above 12 ha
def load_o_sets():
    # Shifted
    o_1 = set([0,1,3,4,9,10,12])
    o_2 = set([2,5,11,14])
    o_3 = set([6,7,15,16])
    o_4 = set([8,17])
    o_5 = set([18,19,21,22])
    o_6 = set([20,23])
    o_7 = set([24,25])
    o_8 = set([26])
    return [o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8]
osets = load_o_sets()

@njit(cache=False)
def load_octree():
    # Indices for the seven elements in each octet
        # in order of appearance in the Lee '94 Fig. 6 octree
    octets = [[0, 1, 3, 4, 9, 10, 12], # 1
               [1, 4, 10, 2, 5, 11, 13], # 2
               [3, 4, 12, 6, 7, 14, 15], # 3
               [4, 5, 13, 7, 15, 8, 16], # 4
               [9, 10, 12, 17, 18, 20, 21], # 5
               [10, 11, 13, 18, 21, 19, 22], # 6
               [12, 14, 15, 20, 21, 23, 24], # 7
               [13, 15, 16, 21, 22, 24, 25]] # 8
    octets = np.array(octets, dtype=np.uint8)
    octets += octets > 12 # Add one to indices above 13
    
    # Octree external membership for all 7 elements in each octree
    v_memberships = [[[0, 0, 0], [2, 0, 0], [3, 0, 0], [2, 3, 4], 
                     [5, 0, 0], [2, 5, 6], [3, 5, 7]], # Octree 1 families
                    [[1, 0, 0], [1, 3, 4], [1, 5, 6], [0,0,0], 
                     [4, 0, 0], [6, 0, 0], [4, 6, 8]], # Octree 2 families
                    [[1, 0, 0], [1, 2, 4], [1, 5, 7], [0, 0, 0],
                     [4, 0, 0], [7, 0, 0], [4, 7, 8]], # Octree 3 families
                    [[1, 2, 3], [2, 0, 0], [2, 6, 8], [3, 0, 0], 
                     [3, 7, 8], [0, 0, 0], [8, 0, 0]], # Octree 4 families
                    [[1, 0, 0], [1, 2, 6], [1, 3, 7], [0, 0, 0], 
                     [6, 0, 0], [7, 0, 0], [6, 7, 8]], # Octree 5 families
                    [[1, 2, 5], [2, 0, 0], [2, 4, 8], [5, 0, 0], 
                     [5, 7, 8], [0, 0, 0], [8, 0, 0]], # Octree 6 families
                    [[1, 3, 5], [3, 0, 0], [3, 4, 8], [5, 0 ,0], 
                     [5, 6, 8], [0, 0, 0], [8, 0, 0]], # Octree 7 families
                    [[2, 4, 6], [3, 4, 7], [4, 0, 0], [5, 6, 7], 
                     [6, 0 ,0], [7, 0, 0], [0, 0, 0]]] # Octree 8 families
    v_memberships = np.array(v_memberships)
    v_memberships -= 1
    return octets, v_memberships    

#############################
### Border identification ###
#############################
@njit(cache=False)
def identify_nonzero(volume):
    points = np.vstack(np.nonzero(volume)).T
    return points

########################
### Point Evaluation ###
########################
## Border point identification
@njit(parallel=True, cache=False)
def border_evaluation(volume, points, o):
    n_points = points.shape[0]
    border_filter = np.full(n_points, False)
    for n in prange(n_points):
        if not volume[points[n][0]+o[0], points[n][1]+o[1], points[n][2]+o[2]]:
            border_filter[n] = True
    border_points = points[border_filter]
    border_ids = np.nonzero(border_filter)[0]
    return border_points, border_ids

@njit(cache=False)
def is_endpoint(volume, point):
    z, y, x = point
    return np.sum(volume[z-1:z+2, y-1:y+2, x-1:x+2]) <= 2

### Euler evaluation
# Neighborhood retrieval for Euler and simple-point tests
@njit(cache=False)
def get_neighborhood(volume, p):
    z, y, x = p
    N26v = volume[z-1:z+2, y-1:y+2, x-1:x+2].flatten()
    return N26v

@njit(cache=False)
def Euler_evaluation(N26v, E_LUT, E_octants):
    e = 0
    for i in range(8):
        e_octant = 1
        for j in range(7):
            if N26v[E_octants[i][j]]:
                e_octant |= 1 << 7-j
        e += E_LUT[e_octant]
    return e == 0

@njit(cache=False)
def octree_labeling(N26v, label, octets, v_memberships, o_index):
    for v in range(7):
        if N26v[octets[o_index, v]] == 1:
            N26v[octets[o_index, v]] = label
            for k in range(3):
                if v_memberships[o_index, v, k] != -1:
                    octree_labeling(N26v, label, octets, v_memberships, 
                                    v_memberships[o_index, v, k])
    return

@njit(cache=False)
def N26v_labeling(N26v, octets, v_memberships, o_sets):
    label = 2
    for i in range(26):
        if i == 13:
            continue
        if N26v[i] == 1:
            for o_index in range(8): # Iterate through octets in octree
                if i in o_sets[o_index]:
                    octree_labeling(N26v, label, octets, v_memberships, o_index)
            label += 1
            if label - 2 >= 2: # There should only be one labeled component present
                return False
    return True

@njit(parallel=True, cache=False)
def is_simple_point(volume, points, o, 
                       E_LUT, E_octants,
                       octants, v_memberships, o_sets):
    n_points = points.shape[0]
    point_filter = np.full(n_points, False)
    for n in prange(n_points):
        if not volume[points[n][0]+o[0], points[n][1]+o[1], points[n][2]+o[2]]:
            if is_endpoint(volume, points[n]):
                continue
            N26v = get_neighborhood(volume, points[n])
            if  (N26v_labeling(N26v, octants, v_memberships, o_sets) and
                Euler_evaluation(N26v, E_LUT, E_octants)):
                point_filter[n] = True
                
    points = points[point_filter]
    border_ids = np.nonzero(point_filter)[0]
    return points, border_ids

## Final point removal for each subiteration
@njit(cache=False)
def set_zero(volume, points, border_ids,
             octants, v_memberships, o_set):
    final_filter = np.full(border_ids.shape[0], True)
    for n in range(points.shape[0]):
        N26v = get_neighborhood(volume, points[n])
        if N26v_labeling(N26v, octants, v_memberships, o_set):
            volume[points[n][0], points[n][1], points[n][2]] = 0
        else:
            final_filter[n] = False
    border_ids = border_ids[final_filter]
    return volume, border_ids

@njit(parallel=True, cache=False)
def label_skeleton(volume, points, label):
    for i in prange(points.shape[0]):
        z, y, x = points[i]
        volume[z, y, x] = label
        
    return volume

@njit(cache=False)
def skeletonize(volume, return_labeled=False, verbose=False):  
    """Skeletonize an input volume
    
    Parameters
    ----------
    volume: 3D np.array
        A binary padded volume must be loaded for skeletonization
        
    return_labeled: bool, optional
        If true, returns an np.int_ volume where the points are labeled in 
        order of their removal from the volume
    
    verbose: bool, optional
    
    Returns
    -------
    3D np.array
        The binary skeleton or the labeled volume with skeletonization steps
        
    """
    # Load the thinning orientations
    orientations = init_filters()
    
    # Load the Euler characteristic tables and Euler octants
    E_LUT = load_Euler_LUT()
    E_octants = E_octant
    
    # Load the octants for N26(v) component labeling
    o_sets = load_o_sets()
    octants, v_memberships = load_octree()
    
    # Identify the nonzero points in the volume
    points = identify_nonzero(volume)
    
    cycle = 0
    
    # Create an array for removal labeling
    if return_labeled:
        labeled_skeleton = np.zeros(volume.shape, dtype=np.uint16)
        step = 1
        
    while True:
        remaining_points = points.shape[0]
        for i in range(6):
            if verbose: # Not available for numba==0.53.1
                with objmode():
                    print (f"Cycle:{cycle}, Iteration:{i}, Points remaining: {points.shape[0]}")
            # First identify all candidate border points
            border_points, border_ids = is_simple_point(volume, points, 
                                                        orientations[i],
                                                        E_LUT, E_octants,
                                                        octants, v_memberships, 
                                                        o_sets)
        
            # Then eliminate all of of the simple points
            volume, border_ids = set_zero(volume, border_points, border_ids,
                                  octants, v_memberships, o_sets)
            
            # Update our points array
            keep_filter = np.full(points.shape[0], True)
            keep_filter[border_ids] = False
            
            if return_labeled:
                removed_filter = np.invert(keep_filter)
                removed_points = points[removed_filter]
                labeled_skeleton = label_skeleton(labeled_skeleton, 
                                                  removed_points, 
                                                  step)
                step += 1 # Update the step
            
            # Update points list
            points = points[keep_filter]
            
        if remaining_points == points.shape[0]:
            if return_labeled:
                labeled_skeleton = label_skeleton(labeled_skeleton, 
                                                  points,
                                                  step-1)
            break
        cycle += 1

    return volume.astype(np.uint16) if not return_labeled else labeled_skeleton


if __name__ == "__main__":
    cube = np.ones((3,3,3), dtype=np.uint8)
    cube = np.pad(cube, 1)
    
    skeleton = skeletonize(cube, return_labeled=False, verbose=True)