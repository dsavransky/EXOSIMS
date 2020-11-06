"""evenlyDistributePointsOnSphere.py
This code creates a set of n points on a unit sphere which are approximately spaced as far as possible from each other.

#Written By: Dean Keithly
#Written On: 10/16/2018
"""

from numpy import pi, cos, sin, arccos, arange
import os
import matplotlib
if not 'DISPLAY' in os.environ and not 'indows' in os.environ['OS']: #Check environment for keys
    import matplotlib.pyplot as plt 
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d
#from pylab import * # Not sure if necessary
import numpy as np
from scipy.optimize import minimize

def secondSmallest(d_diff_pts):
    """For a list of points, return the value and ind of the second smallest
    args:
        d_diff_pts - numy array of floats of distances between points
    returns:
        secondSmallest_value - 
        secondSmallest_ind - 
    """
    tmp_inds = np.arange(len(d_diff_pts))
    tmp_inds_min0 = np.argmin(d_diff_pts)
    tmp_inds = np.delete(tmp_inds, tmp_inds_min0)
    tmp_d_diff_pts =np.delete(d_diff_pts, tmp_inds_min0)
    secondSmallest_value = min(tmp_d_diff_pts)
    secondSmallest_ind = np.argmin(np.abs(d_diff_pts - secondSmallest_value))
    return secondSmallest_value, secondSmallest_ind

def pt_pt_distances(xyzpoints):
    distances = list()
    closest_point_inds = list() # list of numpy arrays containing closest points to a given ind
    for i in np.arange(len(xyzpoints)):
        xyzpoint = xyzpoints[i] # extract a single xyz point on sphere
        diff_pts = xyzpoints - xyzpoint # calculate linear difference between point spacing
        d_diff_pts = np.linalg.norm(diff_pts,axis=1) # calculate linear distance between points
        ss_d, ss_ind = secondSmallest(d_diff_pts) #we must get the second smallest because the smallest is the point itself
        distances.append(ss_d)
        closest_point_inds.append(ss_ind)
    return distances, closest_point_inds

def f(vv):
    # This is the optimization problem objective function
    # We calculate the sum of all distances between points and 
    xx = vv[::3]
    yy = vv[1::3]
    zz = vv[2::3]
    xyzpoints = np.asarray([[xx[i], yy[i], zz[i]]for i in np.arange(len(zz))])
    #Calculates the sum(min(dij)**3.)
    distances, inds = pt_pt_distances(xyzpoints)
    return sum(1./np.asarray(distances))#-sum(np.asarray(distances)**2.) #squares and sums each point-to-closest point distances

def nlcon2(vvv,ind):
    """ This is the nonlinear constraint on each "point" of the sphere
    We require that the center-to-point distance be ~1.
    Args:
        vvv (numpy array) - dims of 3xn where [[x0,y0,z0],...,[xn,yn,zn]]
        ind (integer) - index of the constraint of the point
    """
    xxx = vvv[::3][ind] # this decodes the x vars, vvv[0] and every 3rd element after that
    yyy = vvv[1::3][ind] # this decodes the y vars, vvv[1] and every 3rd element after that
    zzz = vvv[2::3][ind] # this decodes the z vars, vvv[2] and every 3rd element after that
    xyzpoint = np.asarray([xxx,yyy,zzz])#[[xxx[i], yyy[i], zzz[i]]for i in np.arange(len(zzz))])
    return np.linalg.norm(xyzpoint) - 1. #I just want the length to be 1

def splitOut(aa):
    """Splits out into x,y,z, Used to simplify code
    Args:
        aa - dictionary spit out by  
    Returns:
        outkx, outky, outkz (numpy arrays) - arrays of the x, y, and values of the points
    """
    outkx = aa['x'][::3]
    outky = aa['x'][1::3]
    outkz = aa['x'][2::3]
    return outkx, outky, outkz

def plotAllPoints(x,y,z,f,x0,con):
    """
    Args:
        x- initial x points
        y- initial y points
        z- initial z points
        f- objective function for optimization
        x0- flattened initial values to be shoved into objective function
        con- list of dicts of constraints to be placed on the values
    """
    #plt.close(5006)
    fig = plt.figure(num=5006)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='blue')
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rc('font',weight='bold')
    plt.show(block=False)


    out01k = minimize(f,x0, method='SLSQP',constraints=(con), options={'ftol':1e-4, 'maxiter':100})
    out01kx, out01ky, out01kz = splitOut(out01k)
    ax.scatter(out01kx, out01ky, out01kz,color='purple')

    out1k = minimize(f,x0, method='SLSQP',constraints=(con), options={'ftol':1e-4, 'maxiter':1000})
    out1kx, out1ky, out1kz = splitOut(out1k)
    ax.scatter(out1kx, out1ky, out1kz,color='red')
    plt.show(block=False)

    out2k = minimize(f,x0, method='SLSQP',constraints=(con), options={'ftol':1e-4, 'maxiter':2000})
    out2kx, out2ky, out2kz = splitOut(out2k)
    ax.scatter(out2kx, out2ky, out2kz,color='green')
    plt.show(block=False)

    out4k = minimize(f,x0, method='SLSQP',constraints=(con), options={'ftol':1e-4, 'maxiter':4000})
    out4kx, out4ky, out4kz = splitOut(out4k)
    ax.scatter(out4kx, out4ky, out4kz,color='cyan')
    plt.legend(['Initial','100 iter.','1k iter.','2k iter.','4k iter.'],loc='uplter left')
    ax.set_xlabel('X',weight='bold')
    ax.set_ylabel('Y',weight='bold')
    ax.set_zlabel('Z',weight='bold')
    plt.title('Points Distributed on a Sphere',weight='bold')
    plt.show(block=False)

    # To Save this figure:
    # gca()
    # savefig('figurename.png')
    return fig, out01k, out1k, out2k, out4k

def setupConstraints(v,nlcon2):
    """ Sets Up all Constraints on each vector
    """
    con = list()
    for i in np.arange(len(v)):
        ctemp = {'type':'eq','fun':nlcon2,'args':(i,)}
        con.append(ctemp) 
    return con

def initialXYZpoints(num_pts=30):
    """ Quick and unprecise way of distributing points on a sphere
    """
    indices = arange(0, num_pts, dtype=float) + 0.5
    phi = arccos(1 - 2*indices/num_pts)
    theta = pi * (1 + 5**0.5) * indices
    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    v = np.asarray([[x[i], y[i], z[i]] for i in np.arange(len(x))]) # an array of each point on the sphere
    d = np.linalg.norm(v,axis=1) # used to ensure the length of each vector is 1
    return x, y, z, v


if __name__ == '__main__':
    """ The main function will produce a plot of points optimally spaced on a sphere 
    """
    #### Generate Initial Set of XYZ Points ###############
    x, y, z, v = initialXYZpoints(num_pts=30)
    #######################################################

    #### Define constraints on each point of the sphere #######
    con = setupConstraints(v,nlcon2)
    #### Define initial conditions of points on sphere
    x0 = v.flatten() # takes v and converts it into [x0,y0,z0,x1,y1,z1,...,xn,yn,zn]
    out1k = minimize(f,x0, method='SLSQP',constraints=(con), options={'ftol':1e-4, 'maxiter':1000}) # run optimization problem for 1000 iterations
    # out1k contains all of the relevant output information out1k['x'], out1k['fun'], out1k['success']

    #### Recast out1k to [[x0,y0,z0],...,[xn,yn,zn]] array
    out1kx, out1ky, out1kz = splitOut(out1k)
    out1kv = np.asarray([[out1kx[i], out1ky[i], out1kz[i]] for i in np.arange(len(out1kx))])

    
    #### Plot Points on Sphere for verying number of numerical optimizations
    fig, out01k, out1k, out2k, out4k = plotAllPoints(x,y,z,f,x0,con)

    #### Split out the xyz components of solution
    out01kx, out01ky, out01kz = splitOut(out01k)
    out1kx, out1ky, out1kz = splitOut(out1k)
    out2kx, out2ky, out2kz = splitOut(out2k)
    out4kx, out4ky, out4kz = splitOut(out4k)

    #### Combine xyz components into  vectors
    out01kv = np.asarray([[out01kx[i], out01ky[i], out01kz[i]] for i in np.arange(len(out01kx))])
    out1kv = np.asarray([[out1kx[i], out1ky[i], out1kz[i]] for i in np.arange(len(out1kx))])
    out2kv = np.asarray([[out2kx[i], out2ky[i], out2kz[i]] for i in np.arange(len(out2kx))])
    out4kv = np.asarray([[out4kx[i], out4ky[i], out4kz[i]] for i in np.arange(len(out4kx))])

    #### Calculate array of minimum distances between points
    dist01k, inds01k = pt_pt_distances(out01kv)
    dist1k, inds1k = pt_pt_distances(out1kv)
    dist2k, inds2k = pt_pt_distances(out2kv)
    dist4k, inds4k = pt_pt_distances(out4kv)

    #### Get minimum, maximum, and mean distances as function of number of iterations #########
    minimumDistances = np.asarray([min(dist01k), min(dist1k), min(dist2k), min(dist4k)])
    maximumDistances = np.asarray([max(dist01k), max(dist1k), max(dist2k), max(dist4k)])
    meanDistances = np.asarray([np.mean(dist01k), np.mean(dist1k), np.mean(dist2k), np.mean(dist4k)])

    fig2 = plt.figure(num=5007)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rc('font',weight='bold')
    plt.plot([100,1000,2000,3000],minimumDistances,marker='s',linestyle = 'None')
    plt.plot([100,1000,2000,3000],maximumDistances,marker='o',linestyle = 'None')
    plt.plot([100,1000,2000,3000],meanDistances,marker='v',linestyle = 'None')
    plt.ylabel('Distances (1 is the radius of the sphere)',weight='bold')
    plt.xlabel('Number of Optimization Iterations',weight='bold')
    plt.legend(['min(Dist)','max(Dist)','mean(Dist)'])
    plt.show(block=False)

