import numpy as np
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef extern from "KeplerSTM_C.h":
    int KeplerSTM_C(double* x0, double dt, double mu, double* x1, double epsmult)
    int KeplerSTM_C_vallado(double* x0, double dt, double mu, double* x1, double epsmult)

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def CyKeplerSTM(np.ndarray[DTYPE_t, ndim=1] x0, DTYPE_t dt, np.ndarray[DTYPE_t, ndim=1] mus, DTYPE_t epsmult):
    '''
    Kepler State Transition Matrix

    Cythonized version of the Kepler State Transition Matrix Calculations.
    This provides a single method to iteratively call the backend KeplerSTM_C function
    on inputs.  

    Args:
        x0 (ndarray):
            6n x 1 vector of stacked positions and velocities for n planets
        dt (float):
            Time step
        mus (ndarray):
            n x 1 vector of standard gravitational parameters mu = G(m+m_s) where m is 
            the planet mass, m_s is the star mass and G is the gravitational 
            constant
        epsmult (float):
            default multiplier on floating point precision, used as convergence 
            metric.  Higher values mean faster convergence, but sacrifice precision.
            
    Return:
        x1 (ndarray):
            Propagated orbital values (equivalent dimension to x0)
    
    Notes:
        All units must be complementary (i.e., if position is AU and velocity
        is AU/day, mu must be in AU^3/day^2).

        Code must be compiled before use: 
        > python CyKeplerSTM_setup.py build_ext --inplace

    '''
    cdef int N = mus.size

    assert (x0.dtype == DTYPE) and (x0.size == N*6), "Incompatible inputs." 


    #intialize output and intermediate arrays
    cdef np.ndarray[DTYPE_t, ndim=1] x1 =  np.zeros(6*N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] xin =  np.zeros(6, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] xout =  np.zeros(6, dtype=DTYPE)

    cdef int res,j, mucounter
    mucounter = 0
    for j from 0 <= j < x0.size by 6:
        xin = x0[j:j+6]
        res = KeplerSTM_C(<double*> xin.data, dt, mus[mucounter], <double*> xout.data, epsmult)
        if (res != 0):
            res = KeplerSTM_C_vallado(<double*> xin.data, dt, mus[mucounter], <double*> xout.data, epsmult)
            if (res != 0):
                raise Exception("Integration failed.")
        x1[j:j+6] = xout
        mucounter += 1
 
    return x1

