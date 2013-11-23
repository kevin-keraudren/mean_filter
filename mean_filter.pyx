import cython
cimport numpy as np
import numpy as np

cdef float integrate(np.ndarray[float, ndim=3,  mode="c"] sat,
                     int d0, int r0, int c0, int d1, int r1, int c1):
    """
    Taken from:
    https://github.com/scikit-image/scikit-image
    
    Using a summed area table / integral image, calculate the sum
    over a given window.

    This function is the same as the `integrate` function in
    `skimage.transform.integrate`, but this Cython version significantly
    speeds up the code.

    Parameters
    ----------
    sat : ndarray of float
        Summed area table / integral image.
    r0, c0 : int
        Top-left corner of block to be summed.
    r1, c1 : int
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : int
        Sum over the given window.
    """
    cdef float S = 0

    d1 = min(d1,sat.shape[0]-1)
    r1 = min(r1,sat.shape[1]-1)
    c1 = min(c1,sat.shape[2]-1)
    
    S += sat[d1, r1, c1]

    if (d0 - 1 >= 0):
        S -= sat[d0-1, r1, c1]
        
    if (r0 - 1 >= 0):
        S -= sat[d1, r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= sat[d1, r1, c0 - 1]    

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[d1, r0 - 1, c0 - 1]

    if (d0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[d0 - 1, r1, c0 - 1]

    if (d0 - 1 >= 0) and (r0 - 1 >= 0):
        S += sat[d0 - 1, r0 - 1, c1]

    if (d0 - 1 >= 0) and (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S -= sat[d0 - 1, r0 - 1, c0 - 1]         

    return S

@cython.boundscheck(False)
def mean_filter( np.ndarray[float, ndim=3, mode="c"] image,
                 int s0, int s1, int s2 ):
    image_sat = np.cumsum(np.cumsum(np.cumsum(image,2),1),0)

    filtered = np.zeros( (image.shape[0],
                          image.shape[1],
                          image.shape[2]),
                         dtype=np.float32 )

    cdef int s_0 = s0/2
    cdef int s_1 = s1/2
    cdef int s_2 = s2/2
    cdef int size = s0*s1*s2
    cdef int i, j, k
    cdef int i0, j0, k0
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            for k in xrange(image.shape[2]):
                i0 = i - s_0
                j0 = j - s_1
                k0 = k - s_2
                # subtract 1 because `i_end` and `j_end` are used for indexing into
                # summed-area table, instead of slicing windows of the image.
                i0_end = i0 + s0 - 1
                j0_end = j0 + s1 - 1
                k0_end = k0 + s2 - 1

                filtered[i,j,k] = integrate( image_sat,
                                             i0, j0, k0,
                                             i0_end, j0_end, k0_end)

    filtered /= size
    return filtered
