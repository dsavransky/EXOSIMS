import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import numbers

class RejectionSampler():
    '''
    Simple Rejection Sampler for arbitrary distributions defined 
    via a PDF encoded as a function (or lambda function)

    Args:
        f (function):
            Probability density function.  Must be able to operate
            on numpy ndarrays.  Function does not need to be 
            normalized over the sampling interval.
        xMin (float):
            Minimum of interval to sample (inclusive).
        xMax (float):
            Maximum of interval to sample (inclusive).

    Attributes:
        f, xMin, xMax
            As above


    Notes:
        If xMin == xMax, return values will all exactly equal xMin.
        To sample call the object with the desired number of samples.
    '''
    
    def __init__(self, f, xMin, xMax):

        #validate inputs
        assert isinstance(xMin,numbers.Number) and \
               isinstance(xMax,numbers.Number),\
                "xMin and xMax must be numbers."
        self.xMin = float(xMin)
        self.xMax = float(xMax)

        assert hasattr(f, '__call__'),\
                "f must be callable."
        self.f = f

        if xMin != xMax:
            self.M = self.calcM()
        


    def __call__(self, numTest=1, verb = False):
        '''
        A call to the object with the number of samples will 
        return the sampled distribution.
        '''

        assert isinstance(numTest,numbers.Number),\
                "numTest must be an integer."
        numTest = int(numTest)
        
        if self.xMin==self.xMax:
            return np.zeros(numTest)+self.xMin
        
        #initialize
        n = 0
        X = np.zeros(numTest)
        numIter = 0
        maxIter = 1000
        
        nSamp = max(2*numTest, 1000*1000)
        while n < numTest and numIter < maxIter:
            xd = np.random.random(nSamp) * (self.xMax - self.xMin) + self.xMin
            yd = np.random.random(nSamp) * self.M
            pd = self.f(xd)
            
            xd = xd[yd < pd]
            X[n:min(n+len(xd), numTest)] = xd[:min(len(xd),numTest-n)]
            n += len(xd)
            numIter += 1
        
        if numIter == maxIter:
            raise Exception("Failed to converge.")
        
        if verb:
            print('Finished in '+repr(numIter)+' iterations.')
        
        return X

    def calcM(self):
        '''
        Calculate the maximum bound of the distribution over the 
        sampling interval.
        '''
        #first do a coarse grid to get ic
        dx = np.linspace(self.xMin, self.xMax, 1000*1000)
        ic = np.argmax(self.f(dx))
        
        #now optimize
        g = lambda x: -self.f(x)
        M = fmin_l_bfgs_b(g,[dx[ic]],approx_grad=True,bounds=[(self.xMin,self.xMax)])
        M = self.f(M[0])
        
        return M


