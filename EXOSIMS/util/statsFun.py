import numpy as np
from scipy.optimize import fmin_l_bfgs_b

'''
Collection of useful statistics routines

12/22/11 Dmitry Savransky savransky1@llnl.gov
'''

def simpSample(f, numTest, xMin, xMax, verb = False):
    '''
    Use the rejection sampling method to generate a 
    probability distribution according to the given function f, 
    between some range xMin and xMax
    '''

    #find max value
    #first do a coarse grid to get ic
    dx = np.linspace(xMin,xMax,1000000)
    ic = np.argmax(f(dx))
    g = lambda x: -f(x)
    M = fmin_l_bfgs_b(g,[dx[ic]],approx_grad=True,bounds=[(xMin,xMax)])
    M = f(M[0])

    #initialize
    n = 0
    X = np.zeros(numTest);
    numIter = 0;
    maxIter = 1000;

    nSamp = np.max([2*numTest,1e6])
    while n < numTest and numIter < maxIter:
        xd = np.random.random(nSamp) * (xMax - xMin) + xMin
        yd = np.random.random(nSamp) * M
        pd = f(xd)

        xd = xd[yd < pd]
        X[n:min(n+len(xd), numTest)] = xd[:min(len(xd),numTest-n)]
        n += len(xd)
        numIter += 1

    if numIter == maxIter:
        raise Exception("Failed to converge.")

    if verb:
        print 'Finished in '+repr(numIter)+' iterations.'

    return X

def eqLogSample(f, numTest, xMin, xMax, bins=10):
    out = np.array([])
    bounds = np.logspace(np.log10(xMin),np.log10(xMax),bins+1)
    for j in np.arange(1,bins+1):
        out = np.concatenate((out,simpSample(f,numTest/bins,bounds[j-1],bounds[j])))

    return out
