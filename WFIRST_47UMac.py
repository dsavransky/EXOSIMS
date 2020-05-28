#DmitryWantsAPony.py
# Detectability of 47 UMa c
# author: Dean Keithly, Gabriel Soto, Corey Spohn

import matplotlib.pyplot as plt 
from pylab import *
rc('axes', linewidth=2)
rc('font', weight='bold',size=14)

import os
import EXOSIMS.MissionSim
import numpy.random as rand
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Angle
from EXOSIMS.util.deltaMag import *
from EXOSIMS.util.eccanom import eccanom
from scipy import interpolate
import pandas as pd
import radvel.utils as rvu
import radvel.orbit as rvo
import keplertools.fun as fun


import EXOSIMS,os.path
scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','WFIRST_47UMac.json')
import EXOSIMS.MissionSim
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)

PPop = sim.PlanetPopulation
PPM = sim.PlanetPhysicalModel

def RfromM(m):
    '''
    Given masses m (in Earth masses) return radii (in Earth radii) \
    based on modified forecaster
    '''
    m = np.array(m,ndmin=1)
    R = np.zeros(m.shape)


    S = np.array([0.2790,0,0,0,0.881])
    C = np.array([np.log10(1.008), 0, 0, 0, 0])
    T = np.array([2.04,95.16,(u.M_jupiter).to(u.M_earth),((0.0800*u.M_sun).to(u.M_earth)).value])

    Rj = u.R_jupiter.to(u.R_earth)
    Rs = 8.522 #saturn radius

    S[1] = (np.log10(Rs) - (C[0] + np.log10(T[0])*S[0]))/(np.log10(T[1]) - np.log10(T[0]))
    C[1] = np.log10(Rs) - np.log10(T[1])*S[1]

    S[2] = (np.log10(Rj) - np.log10(Rs))/(np.log10(T[2]) - np.log10(T[1]))
    C[2] = np.log10(Rj) - np.log10(T[2])*S[2]

    C[3] = np.log10(Rj)

    C[4] = np.log10(Rj) - np.log10(T[3])*S[4]


    inds = np.digitize(m,np.hstack((0,T,np.inf)))
    for j in range(1,inds.max()+1):
        R[inds == j] = 10.**(C[j-1] + np.log10(m[inds == j])*S[j-1])

    return R
### !!!!!!!!!
### Make this value True if you want to output plots
IWantPlots = True

randomM0 = False #true if using randomM0
periastronM0 = True #true if trying to use M0 from periastron estimates
nYears = 0 #default 0, mission start

#From Dmitry's links
period = 2391 #days +100 -87
sma = 3.6 #+/-0.1
e = 0.098 #+0.047 -0.096
#Time periastron passage (days) 2452441 +628-825
#Longitude of Periastron (deg) 295 +114-160
mass = 0.54 #+.066 -.073 in jupiter mass Msin(i)
w = (295*u.deg).to('rad') #from https://plandb.sioslab.com/plandetail.php?name=47+UMa+c

#Host Star Aliases
#47 UMa     2MASS J10592802+4025485     BD+41 2147  Chalawan    GJ 407  HD 95128    HIP 53721   HR 4277     IRAS 10566+4041     SAO 43557   TYC 3009-02703-1    WISE J105927.66+402549.4
Bmag = 5.66 #(mag)
radius = 1.23 #r sun
star_d = 13.802083302115193#distance (pc) ±0.028708172014593
star_mass = 1.03 #0.05

n = 10**4 # Number of planets to simulate
time_steps = 2

##############################
# Times to consider
##############################
print('Creating a Fake Target List for HIP 53721')
from EXOSIMS.Prototypes.TargetList import TargetList
obs = sim.Observatory
missionStart = sim.TimeKeeping.missionStart  #Time Object
time_forward = 3*u.yr
mission_steps = np.linspace(0 * u.d, time_forward.to(u.d), num=time_steps) 
mission_times = missionStart + mission_steps

# =============================================================================
# keepout calculations
# =============================================================================
ra_input  = np.array([ Angle('10h59m27.97s').to('deg').value ])
dec_input = np.array([ Angle('40d25m48.9s').to('deg').value ])
fTL = TargetList(**{"ra_input":ra_input,"dec_input":dec_input,"star_dist":star_d,'modules':{"StarCatalog": "FakeCatalog_InputStars", \
                    "TargetList":"EclipticTargetList ","OpticalSystem": "Nemati", "ZodiacalLight": "Stark", "PostProcessing": " ", \
                    "Completeness": " ","BackgroundSources": "GalaxiesFaintStars", "PlanetPhysicalModel": " ", \
                    "PlanetPopulation": "KeplerLike1"}, "scienceInstruments": [{ "name": "imager"}],  \
                    "starlightSuppressionSystems": [{ "name": "HLC-565"}]   })

#s x 4 x 2 array where s is the number of starlight suppression systems [WE ONLY HAVE 1] as
# defined in the Optical System. Each of the remaining 4 x 2 arrays are system
# specific koAngles for the Sun, Moon, Earth, and small bodies (4), each with a 
# minimum and maximum value (2) in units of deg.    
koangles = np.array([ [40,180],[40,180],[40,180],[1,180]  ]).reshape(1,4,2)
obs.koAngles_SolarPanel = [53,124]*u.deg   # solar panel restrictions

# Mission times using Gabe's convention
dtRange = np.arange(0,time_forward.to(u.d).value, (time_forward/time_steps).to(u.d).value)*u.d
missionTimes = missionStart + dtRange
# star of interest
sInds = np.array([0])
# initializing arrays
koGood = np.zeros( missionTimes.size)
culprit = np.zeros( [1,1,missionTimes.size,12])
# calculating keepouts throguhout the year
for t,date in enumerate(missionTimes):
    koGood[t],r_body, r_targ, culprit[:,:,t,:], koangleArray = obs.keepout(fTL, sInds, date, koangles, returnExtra=True)
print('Done Generating Keepout')
observableDates = missionStart + dtRange[[bool(b) for b in koGood]]

### Plotting keepouts
if IWantPlots:
    keepoutString = """Sun KO = [%.0f,%.0f],
    Earth KO = [%.0f,%.0f],
    Moon KO = [%.0f,%.0f],
    Small Planet KO = [%.0f,%.0f],
    Solar Panel KO = [%.0f,%.0f]
    """ %(koangles[0,0,0],koangles[0,0,1], koangles[0,1,0],koangles[0,1,1], koangles[0,2,0],koangles[0,2,1], koangles[0,3,0],koangles[0,3,1],obs.koAngles_SolarPanel[0].value,obs.koAngles_SolarPanel[1].value)
    
    plt.figure(figsize=(10,8))
    plt.plot(dtRange , np.sum(culprit,axis=-1)[0,0],linewidth=3,label=keepoutString)
    plt.plot(dtRange[[bool(b) for b in koGood]] , np.zeros(observableDates.shape),'o',label='Observable Dates')
    plt.xlabel("Time (d)",labelpad=16, fontsize=14,fontweight='bold')
    plt.ylabel("Number of Culprits Causing Keepout",labelpad=16, fontsize=14,fontweight='bold')
    plt.title("Fraction of Time at which star is Observable = %0.2f" % (np.sum(koGood) / dtRange.size) )
    plt.legend(loc='best')

# =============================================================================
# Generating parameters according to MCMC chains
# =============================================================================

chains = pd.read_csv("95128_gamma_chains.csv")
chains = chains.drop(
    columns=[
        "Unnamed: 0",
        "per1",
        "k1",
        "tc1",
        "jit_apf",
        "jit_j",
        "jit_lick_a",
        "jit_lick_b",
        "jit_lick_c",
        "jit_lick_d",
        "jit_nea_2d",
        "jit_nea_CES",
        "jit_nea_ELODIE",
        "jit_nea_HRS",
        "secosw1",
        "sesinw1",
        "lnprobability",
    ]
)
# Get a random sample of the chains to compute the covariance matrix with
chains_sample = chains.sample(1000)
# Create covariance matrix
cov_df = chains_sample.cov()
# Get the mean value of each chain to set the mean of the distributions
means = chains.mean()
# Pull samples
samples = np.random.multivariate_normal(means, cov_df.values, n)


# Eccentricity
secosw = samples[:, 3]  # sqrt(e)*cos(w)
sesinw = samples[:, 4]  # sqrt(e)*sin(w)
e = secosw ** 2 + sesinw ** 2
e[e > 1] = 0.0001  # Remove the heretics
e[e == 0] = 0.0001
# Mass of planet and inclination
periods = samples[:, 0] * u.d
K = samples[:, 1]  # Semi-amplitude
Msini = rvu.Msini(K, periods, star_mass, e, Msini_units="earth")
inc = np.zeros(n)
Mp = np.zeros(n)
for i, Msini_val in enumerate(Msini):
    Icrit = np.arcsin(
        (Msini_val * u.M_earth).value / ((0.0800 * u.M_sun).to(u.M_earth)).value
    )
    Irange = [Icrit, np.pi - Icrit]
    C = 0.5 * (np.cos(Irange[0]) - np.cos(Irange[1]))
    inc[i] = np.arccos(np.cos(Irange[0]) - 2.0 * C * np.random.uniform())
    Mp[i] = Msini_val / np.sin(inc[i])
Mp = Mp * u.M_earth

# Semi-major axis
mu = const.G * (Mp + star_mass * u.M_sun)
a = (mu * (periods / (2 * np.pi)) ** 2) ** (1 / 3)

# W = np.zeros(n)

_, W, _ = PPop.gen_angles(n, None)
# Alternate version using the chain for w, but when tested it appears the same as a
# random uniform distribution
w_s = np.arctan2(sesinw, secosw) * u.rad
w = (w_s + np.pi*u.rad) % (2*np.pi *u.rad)
nu_p = (np.pi/2*u.rad - w_s) % (2*np.pi*u.rad)
E_p = 2*np.arctan2(np.sqrt((1-e))*np.tan(nu_p/2), np.sqrt((1+e)))
M_p = E_p - e*np.sin(E_p)*u.rad
T_c = samples[:, 2] * u.d
# T_c = np.array([2457628.75, 2457857.5]) * u.d
T_p = T_c - M_p/(2*np.pi*u.rad/periods)
T_p_2 = np.zeros(n)
for i in range(n):
    T_p_2[i] = rvo.timetrans_to_timeperi(T_c[i].value, periods[i].value, e[i], w[i].value)

t_missionStart = 2461041 #JD 61041 #MJD 01/01/2026
nD = (t_missionStart - T_p.value)*u.d #number of days since t_periastron
nT = np.floor(nD/periods) #number of periods since t_periastron
fT = nD/periods - nT #fractional period past periastron
M0 = 2.*np.pi*fT #Mean anomaly of the planet
E = eccanom(M0, e)

# Remove a few nan values from the list
inds_nan = ~np.isnan(E)
a = a[inds_nan]
e = e[inds_nan]
w = w[inds_nan]
W = W[inds_nan]
inc = inc[inds_nan]
M0 = M0[inds_nan]
E = E[inds_nan]
mu = mu[inds_nan]
Mp = Mp[inds_nan]

# Calculate the albedo and radius
# p = PPM.calc_albedo_from_sma(a)
Rp = RfromM(Mp.to(u.M_earth).value)*u.R_earth

# Remove the planets that are too large
indsTooBig = np.where(Mp  < (13*u.M_jup).to('M_earth'))[0]
a = a[indsTooBig]
e = e[indsTooBig]
w = w[indsTooBig]
W = W[indsTooBig]
inc = inc[indsTooBig]
M0 = M0[indsTooBig]
E = E[indsTooBig]
mu = mu[indsTooBig]
# p = p[indsTooBig]
Rp = Rp[indsTooBig]
Mp = Mp[indsTooBig]

print('Done generating orbital parameters')

# =============================================================================
# Generating Random Orbits for the planet
# =============================================================================
#Construct planet position vectors
O = W
I = inc
a1 = np.cos(O)*np.cos(w) - np.sin(O)*np.cos(I)*np.sin(w)
a2 = np.sin(O)*np.cos(w) + np.cos(O)*np.cos(I)*np.sin(w)
a3 = np.sin(I)*np.sin(w)
A = a*np.vstack((a1, a2, a3))
b1 = -np.sqrt(1 - e**2)*(np.cos(O)*np.sin(w) + np.sin(O)*np.cos(I)*np.cos(w))
b2 = np.sqrt(1 - e**2)*(-np.sin(O)*np.sin(w) + np.cos(O)*np.cos(I)*np.cos(w))
b3 = np.sqrt(1 - e**2)*np.sin(I)*np.cos(w)
B = a*np.vstack((b1, b2, b3))
r1 = np.cos(E) - e
r2 = np.sin(E)

v1 = np.sqrt(mu/a**3)/(1 - e*np.cos(E))
v2 = np.cos(E)

r = (A*r1 + B*r2).T.to('AU')  
d = np.linalg.norm(r, axis=1)
beta = np.arccos(r[:,2]/d)
print('Done calculating r and beta stuff')





###################

#Shove Above Properties Into EXOSIMS
SU = sim.SimulatedUniverse
TL = sim.TargetList
OS = sim.OpticalSystem
TL.dist = (np.zeros(len(indsTooBig)) + star_d)*u.pc
TL.MsTrue = (np.zeros(len(indsTooBig)) + star_mass)*u.M_sun
SU.r = (A*r1 + B*r2).T.to('AU')                           # position
SU.v = (v1*(-A*r2 + B*v2)).T.to('AU/day')                 # velocity
SU.s = np.linalg.norm(sim.SimulatedUniverse.r[:,0:2], axis=1)              # apparent separation
SU.d = np.linalg.norm(sim.SimulatedUniverse.r, axis=1)                     # planet-star distance

SU.a = a           # semi-major axis
SU.e = e           # eccentricity
SU.I = I           # inclinations
SU.O = O           # right ascension of the ascending node
SU.w = w           # argument of perigee
SU.M0 = M0*u.rad            # initial mean anomany
SU.E = eccanom(M0, e)                      # eccentric anomaly
SU.Mp = Mp                            # planet masses
print('Done Assigning to sim Properties')

# =============================================================================
# Allowable Roll Angles
# =============================================================================
# some functions to convert
pi = np.pi*u.rad
def angleConvert(a):
    a = a.to('deg').value
    a = (a % 360)*u.deg
    return a.to('rad')

def angleCompare(a,ub,lb):
    a = a.to('deg').value
    ub = ub.to('deg').value
    lb = lb.to('deg').value
    
    return (a>=lb)&(a<=ub) if lb < ub else np.logical_or( a>=lb, a<=ub)


nu,gam,dnu,dgam = obs.EulerAngles(fTL,0,missionStart,dtRange)

# the projected frame, P-frame, in R-frame components
#    p3 axis points towards the target star
#    p1-p2 plane is where we look at projected separation
#    p2 axis is parallel to ecliptic

# each component has dimensions 3xt where t is the number of days in the simulation
p1_R = np.array([np.cos(gam)*np.cos(nu),\
               np.cos(gam)*np.sin(nu),\
               -np.sin(gam)])

p2_R = np.array([-np.sin(nu),\
               np.cos(nu),\
               np.zeros(len(nu))])

p3_R = np.array([np.sin(gam)*np.cos(nu),\
               np.sin(gam)*np.sin(nu),\
               np.cos(gam)])

# absolute times including missionStart for simulation
absTimes = missionStart + dtRange                   #mission times  in jd
# the telescope's position (T) relative to the origin (0) projected onto R frame (_R)
rT0_R = obs.haloPosition(absTimes) + np.array([1,0,0])*obs.L2_dist.to('au')

# the sun's position (1) relative to the origin (0) in the R Frame
r10_R = obs.convertPos_to_dim(  np.array([-obs.mu,0,0]) )
# the sun's position (1) relative to the telescope (T) in the R Frame
r1T_R = r10_R - rT0_R

# sun line in the projected frame
r1T_P1 = np.array([ np.dot(a,b) for a,b in zip(p1_R.T , r1T_R.value ) ])
r1T_P2 = np.array([ np.dot(a,b) for a,b in zip(p2_R.T , r1T_R.value ) ])
u1T_P, d1T = obs.unitVector( np.vstack([r1T_P1 , r1T_P2]) )


azSunLine = np.array([angleConvert(ang).value for ang in np.arctan2(u1T_P[1,:],u1T_P[0,:])*u.rad])*u.rad

if IWantPlots:
    # inKO = [not bool(b) for b in koGood]
    # u1T_P[:,inKO] = 0
    
    plt.figure(figsize=(10,8))
    plt.plot(dtRange,azSunLine)

# =============================================================================
# Bowtie Filter with random Roll Angle
# =============================================================================


# calculating azimuth for all the planets
az = np.array([angleConvert(ang).value for ang in np.arctan2(SU.r[:,1],SU.r[:,0])])*u.rad
print('Done calculating az')

# generating a random roll angle
rollAngle_pos = angleConvert( rand.uniform(low=0,high=2*np.pi,size=1) *u.rad)
rollAngle_neg = angleConvert(rollAngle_pos - pi)

# applying bowtie -> assuming it has a width of dTheta
dTheta = np.pi/6 * u.rad
rollAngle_pos_upper = angleConvert(rollAngle_pos + dTheta)
rollAngle_pos_lower = angleConvert(rollAngle_pos - dTheta)

rollAngle_neg_upper = angleConvert(rollAngle_neg + dTheta)
rollAngle_neg_lower = angleConvert(rollAngle_neg - dTheta)

# which planets are within the bowtie? (JUST CHECKING AZIMUTH, NOT WA)
pInBowTie_pos = angleCompare(az,rollAngle_pos_upper,rollAngle_pos_lower)
pInBowTie_neg = angleCompare(az,rollAngle_neg_upper,rollAngle_neg_lower)
pInBowtie = np.logical_or(pInBowTie_pos,pInBowTie_neg)

# which planets are within the bowtie +/- roll angle?
dRoll = 13*np.pi/180 *u.rad
rollAngleRoll_pos_upper = angleConvert(rollAngle_pos + dTheta + dRoll)
rollAngleRoll_pos_lower = angleConvert(rollAngle_pos - dTheta - dRoll)
rollAngleRoll_neg_upper = angleConvert(rollAngle_neg + dTheta + dRoll)
rollAngleRoll_neg_lower = angleConvert(rollAngle_neg - dTheta - dRoll)
pInBowTieRoll_pos = angleCompare(az,rollAngleRoll_pos_upper,rollAngleRoll_pos_lower)
pInBowTieRoll_neg = angleCompare(az,rollAngleRoll_neg_upper,rollAngleRoll_neg_lower)
pInBowtieRoll = np.logical_or(pInBowTieRoll_pos,pInBowTieRoll_neg)

print('Done checking in bowtie')

### Plotting planets inside bowtie
if IWantPlots:
    sMax = np.max( SU.s).value
    origin = [0,0]
    #center lines
    posCenterLine = sMax *  np.array([np.cos(rollAngle_pos) , np.sin(rollAngle_pos)])
    negCenterLine = sMax *  np.array([np.cos(rollAngle_neg) , np.sin(rollAngle_neg)])
    
    #plus lines
    posPlusLine   = sMax *  np.array([np.cos(rollAngle_pos_upper) , np.sin(rollAngle_pos_upper)])
    negPlusLine   = sMax *  np.array([np.cos(rollAngle_neg_upper) , np.sin(rollAngle_neg_upper)])
    
    #minus lines
    posMinusLine   = sMax *  np.array([np.cos(rollAngle_pos_lower) , np.sin(rollAngle_pos_lower)])
    negMinusLine   = sMax *  np.array([np.cos(rollAngle_neg_lower) , np.sin(rollAngle_neg_lower)])
    
    plt.figure(figsize=(10,8))
    plt.plot(SU.r[:,0] , SU.r[:,1],'k.',label='Injected Planets - Apparent Separation')
    plt.plot(SU.r[pInBowtie,0] , SU.r[pInBowtie,1],'y.',label='Observable Planets due to BowTie')
    plt.plot( [origin[0] , posCenterLine[0]] , [origin[1] , posCenterLine[1]] ,'k-.',linewidth=3,label='CenterLine of BowTie')
    plt.plot( [origin[0] , negCenterLine[0]] , [origin[1] , negCenterLine[1]] ,'k-.',linewidth=3)
    
    plt.plot( [origin[0] , posPlusLine[0]] , [origin[1] , posPlusLine[1]] ,'k-',linewidth=4)
    plt.plot( [origin[0] , posPlusLine[0]] , [origin[1] , posPlusLine[1]] ,'r-',linewidth=2,label='Upper BowTie')
    plt.plot( [origin[0] , negPlusLine[0]] , [origin[1] , negPlusLine[1]] ,'k-',linewidth=4)
    plt.plot( [origin[0] , negPlusLine[0]] , [origin[1] , negPlusLine[1]] ,'r-',linewidth=2)
    
    plt.plot( [origin[0] , posMinusLine[0]] , [origin[1] , posMinusLine[1]] ,'k-',linewidth=4)
    plt.plot( [origin[0] , posMinusLine[0]] , [origin[1] , posMinusLine[1]] ,'c-',linewidth=2,label='Lower BowTie')
    plt.plot( [origin[0] , negMinusLine[0]] , [origin[1] , negMinusLine[1]] ,'k-',linewidth=4)
    plt.plot( [origin[0] , negMinusLine[0]] , [origin[1] , negMinusLine[1]] ,'c-',linewidth=2)
    
    plt.xlabel("X (Projected)",labelpad=16, fontsize=14,fontweight='bold')
    plt.ylabel("Y (Projected) ",labelpad=16, fontsize=14,fontweight='bold')
    
    buffer = 1.2
    plt.xlim(-sMax*buffer,sMax*buffer)
    plt.ylim(-sMax*buffer,sMax*buffer)
    plt.title("Roll Angle = %0.2f" % rollAngle_pos.to('deg').value)
    plt.legend(loc='best')



# =============================================================================
# IWA and OWA Filter
# =============================================================================
# Read the csv
photometry_filename = '47UMac_phot.csv'
photometry_file = os.path.join(os.path.normpath(os.path.expandvars(photometry_filename)))
photometry_table = np.genfromtxt(photometry_file, delimiter=',', skip_header=1)

# Get the beta values to act as the x, and the rest to function as the y values
beta_vals = photometry_table[:,1]
photometry_vals = photometry_table[:,2:]

# Create interpolant, using axis=0 to go down the columns only
photometry_interp = interpolate.interp1d(beta_vals, photometry_vals, axis=0)

# Get the values for all of the different cloud values
raw_p_phi_vals = photometry_interp(beta.to(u.deg).value)

# Sample f_sed according the table here: https://plandb.sioslab.com/docs/html/index.html
f_sed = np.random.choice([0, .01, .03, .1, .3, 1, 3, 6], len(beta),
                         p = [.099, .001, 0.005, .01, .025, .28, .3, .28])

# Dictionary used to index them according to the csv file
f_sed_ind = {0.00: 0,
             0.01: 1,
             0.03: 2,
             0.10: 3,
             0.30: 4,
             1.00: 5,
             3.00: 6,
             6.00: 7}

# Go through the raw p_phi and get only the one that corresponds to the correct
# cloud cover generated because the interpolant calculates the value for all
# cloud covers for each beta input
p_phi = np.zeros(len(f_sed))
for i, f_sed_i in enumerate(f_sed):
    p_phi[i] = raw_p_phi_vals[i][f_sed_ind[f_sed_i]]

dmags = -2.5*np.log10(p_phi*((Rp/d).decompose())**2).value

# Phi = PPM.calc_Phi(beta)
# dmags = deltaMag(p,Rp,d,Phi)
print('Done calculating p*Phi, dmag')

#Calculate Planet WA's
WA = (SU.s/TL.dist).decompose()*u.rad
print('Done calculating planet WA')

#OWA IWA che k
pInIWAOWA = (WA > OS.IWA.to('rad'))*(WA < OS.OWA.to('rad'))# Outside of IWA and inside of OWA
print('Done checking in IWA OWA')

### Creating interpolant to work with our l/D value
# Getting the csv file with the contrast curve loaded into numpy arrays
contrast_curve_filename = 'WFIRST_47UMac_Contrast.csv'
contrast_curve_file = os.path.join(os.path.normpath(os.path.expandvars(contrast_curve_filename)))
contrast_curve_table = np.genfromtxt(contrast_curve_file, delimiter=',', skip_header=1)

# Creating an interpolant to generate a contrast value for the given lambda/D
lam = contrast_curve_table[0,2]
l_over_D_vals = contrast_curve_table[:,0] * (lam*(u.nm).to(u.m)/2.363)*(u.rad)
contrast_vals = contrast_curve_table[:,1]
contrast_interp = interpolate.interp1d(l_over_D_vals, contrast_vals, kind='cubic',
                                       bounds_error=False)

# Find the dMag limits value
contrasts = contrast_interp(WA) 
contrasts[contrasts < 1e-11] = np.nan # This happens for working angles not in the IWA/OWA 
dmagLims = -2.5*np.log10(contrasts)



# Get the telescope's dmag0
print('Done calculating dMag')


# =============================================================================
# Contrast/Brightness Filter
# =============================================================================

ZL = sim.ZodiacalLight
TL.starMag = lambda sInds, lam: 5.03 #Apparent StarMag from wikipedia
sInds = np.zeros(len(pInIWAOWA))+0
mode = mode = list(filter(lambda mode: mode['detectionMode'] == True, OS.observingModes))[0]
pBrightEnough = dmags < dmagLims
print('Done checking bright enough')

#ANSWER Visible in BowTie at Instant
numObservablePlanetsInBowtie = pInBowtie*pInIWAOWA*pBrightEnough
fracObservablePlanetsInBowtie = np.count_nonzero(numObservablePlanetsInBowtie)/len(indsTooBig)
print(fracObservablePlanetsInBowtie)

#ANSWER Visible in BowTie+Roll at Instant
numObservablePlanetsInBowtieRoll = pInBowtieRoll*pInIWAOWA*pBrightEnough
fracObservablePlanetsInBowtieRoll = np.count_nonzero(numObservablePlanetsInBowtieRoll)/len(indsTooBig)
print(fracObservablePlanetsInBowtieRoll)

#ANSWER Visible in IWA and OWA and dMag #KNOWN AZ case
numObservablePlanetsKnownAZ = pInIWAOWA*pBrightEnough
fracObservablePlanetsKnownAZ = np.count_nonzero(numObservablePlanetsKnownAZ)/len(indsTooBig)
print(fracObservablePlanetsKnownAZ)


###############################################################################
# Time evolution of geometric visibility
###############################################################################
def r_calc(t, t0, SU, TL, OS):
    mu = const.G*(TL.MsTrue+SU.Mp)
    # Mean anomaly
    M = (SU.M0 + np.sqrt(mu/SU.a**3)*(t-t0)*u.rad)%(2*np.pi*u.rad)
    #Eccentric anomaly
    E = eccanom(M, SU.e)
    
    a1 = np.cos(SU.O)*np.cos(SU.w) - np.sin(SU.O)*np.cos(SU.I)*np.sin(SU.w)
    a2 = np.sin(SU.O)*np.cos(SU.w) + np.cos(SU.O)*np.cos(SU.I)*np.sin(SU.w)
    a3 = np.sin(SU.I)*np.sin(SU.w)
    A = SU.a*np.vstack((a1, a2, a3))
    b1 = -np.sqrt(1 - SU.e**2)*(np.cos(SU.O)*np.sin(SU.w) + np.sin(SU.O)*np.cos(SU.I)*np.cos(SU.w))
    b2 = np.sqrt(1 - SU.e**2)*(-np.sin(SU.O)*np.sin(SU.w) + np.cos(SU.O)*np.cos(SU.I)*np.cos(SU.w))
    b3 = np.sqrt(1 - SU.e**2)*np.sin(SU.I)*np.cos(SU.w)
    B = SU.a*np.vstack((b1, b2, b3))
    r1 = np.cos(E) - SU.e
    r2 = np.sin(E)
    
    r_1 = (A*r1 + B*r2).T.to('AU')  
    r_2, v = fun.orbElem2vec(E, mu, (SU.a, SU.e, SU.O, SU.I, SU.w))

    print(r_1-r_2.T.to(u.AU))

    return r_2.T

def WAs_visibility(r, WA, OS):
    '''
    Parameters
    ----------
    r : numpy array of astropy quantities
        The x, y, z components of the r vector

    Returns
    -------
    pInIWAOWA : numpy array
        Boolean values indicating whether a planet is inside the instruments
        inner and outer working angles

    '''
    # Calculate if it's within the IWA/OWA
    pInIWAOWA = (WA > OS.IWA.to('rad'))*(WA < OS.OWA.to('rad'))# Outside of IWA and inside of OWA
    
    return pInIWAOWA

def bowtie_visibility(r, rollAngle_pos):
    '''
    Parameters
    ----------
    r : numpy array of astropy quantities
        The x, y, z components of the r vector
    roll_angle : astropy quantity
        The center roll angle 

    Returns
    -------
    pInBowtie : numpy array
        Boolean values indicating whether a planet is inside the bowtie
    pInBowtieRoll : TYPE
        Boolean values indicating whether a planet is inside the bowtie+roll
    '''
    
    # Calculate the azimuth angle for the planets
    az = np.array([angleConvert(ang).value for ang in np.arctan2(r[:,1],r[:,0])])*u.rad
    
    rollAngle_neg = angleConvert(rollAngle_pos - pi)
    
    # applying bowtie -> assuming it has a width of dTheta
    dTheta = ((65*u.deg)/2).to(u.rad) # The proposed bowtie has a 65 degree opening
    rollAngle_pos_upper = angleConvert(rollAngle_pos + dTheta)
    rollAngle_pos_lower = angleConvert(rollAngle_pos - dTheta)
    
    rollAngle_neg_upper = angleConvert(rollAngle_neg + dTheta)
    rollAngle_neg_lower = angleConvert(rollAngle_neg - dTheta)
    
    # which planets are within the bowtie? (JUST CHECKING AZIMUTH, NOT WA)
    pInBowTie_pos = angleCompare(az,rollAngle_pos_upper,rollAngle_pos_lower)
    pInBowTie_neg = angleCompare(az,rollAngle_neg_upper,rollAngle_neg_lower)
    pInBowtie = np.logical_or(pInBowTie_pos,pInBowTie_neg)
    
    # which planets are within the bowtie +/- roll angle?
    dRoll = 13*np.pi/180 *u.rad
    rollAngleRoll_pos_upper = angleConvert(rollAngle_pos + dTheta + dRoll)
    rollAngleRoll_pos_lower = angleConvert(rollAngle_pos - dTheta - dRoll)
    rollAngleRoll_neg_upper = angleConvert(rollAngle_neg + dTheta + dRoll)
    rollAngleRoll_neg_lower = angleConvert(rollAngle_neg - dTheta - dRoll)
    pInBowTieRoll_pos = angleCompare(az,rollAngleRoll_pos_upper,rollAngleRoll_pos_lower)
    pInBowTieRoll_neg = angleCompare(az,rollAngleRoll_neg_upper,rollAngleRoll_neg_lower)
    pInBowtieRoll = np.logical_or(pInBowTieRoll_pos,pInBowTieRoll_neg)
    
    return pInBowtie, pInBowtieRoll

def dmag_visibility(p_phi, WA, contrast_interp, Rp, d, r, a):
    d = np.linalg.norm(r, axis=1)
    beta = np.arccos((-r[:,2]/d).decompose())
    phi = PPM.calc_Phi(beta)
    p = PPM.calc_albedo_from_sma(a)
    dmags_lambertian = deltaMag(p, Rp, d, phi)
    dmags = -2.5*np.log10(p_phi*(Rp/d).decompose()**2).value
    contrasts = contrast_interp(WA) 
    contrasts[contrasts < 1e-11] = np.nan
    dmagLims = -2.5*np.log10(contrasts)

    pBrightEnough = dmags < dmagLims
    return pBrightEnough, dmags, dmags_lambertian, contrasts

def p_phi_calc(beta, f_sed, f_sed_ind, photometry_interp):
    raw_p_phi_vals = photometry_interp(beta.to(u.deg).value)
    p_phi = np.zeros(len(f_sed))
    for i, f_sed_i in enumerate(f_sed):
        p_phi[i] = raw_p_phi_vals[i][f_sed_ind[f_sed_i]]
    return p_phi

def WA_calc(r, TL):
    s = np.linalg.norm(r[:,0:2], axis=1) 
    WA = (s/TL.dist).decompose()*u.rad 
    return WA

#############################
# Actually running the time evolution
##############################
probs_WA = np.zeros(len(mission_times))
probs_bowtie = np.zeros(len(mission_times))
probs_bowtie_roll = np.zeros(len(mission_times))
probs_dmag = np.zeros(len(mission_times))
probs_total_no_roll = np.zeros(len(mission_times))
probs_total_roll = np.zeros(len(mission_times))

kwargs = dict(histtype='stepfilled', alpha=0.8, bins=40, ec="k")#density=True
for i, t in enumerate(mission_times):
    plt.figure()
    center_roll_angle = azSunLine[i]
    r = r_calc(t, missionStart, SU, TL, OS)
    WA = WA_calc(r, TL)
    beta = np.arccos(-r[:,2]/(np.linalg.norm(r,axis=1).decompose()))
    p_phi = p_phi_calc(beta, f_sed, f_sed_ind, photometry_interp)
    pInWAs = WAs_visibility(r, WA, OS)
    pInBowtie, pInBowtieRoll = bowtie_visibility(r, center_roll_angle)
    pBrightEnough, dmag_current, dmag_current_lambertian, contrast_current = dmag_visibility(p_phi, WA, contrast_interp, Rp, d, r, a)
    current_keepout = koGood[i] # Here a value of 1 represents that the target is observable
    planets_visible_no_roll = pInBowtie*pInIWAOWA*pBrightEnough*current_keepout
    planets_visible_roll = pInBowtieRoll*pInIWAOWA*pBrightEnough*current_keepout
    # Append to arrays
    probs_WA[i] = np.sum(pInWAs*current_keepout)/len(pInWAs)
    probs_bowtie[i] = np.sum(pInBowtie*current_keepout)/len(pInBowtie)
    probs_bowtie_roll[i] = np.sum(pInBowtieRoll*current_keepout)/len(pInBowtieRoll)
    probs_dmag[i] = np.sum(pBrightEnough*current_keepout)/len(pBrightEnough)
    probs_total_no_roll[i] = np.sum(planets_visible_no_roll)/len(planets_visible_no_roll)
    probs_total_roll[i] = np.sum(planets_visible_roll)/len(planets_visible_roll)


    # Separation flux plots
    WAs = np.linspace(OS.IWA, OS.OWA, 1000)
    WA_contrasts = contrast_interp(WAs.to(u.rad))
    plt.ylim([10**-12, 10**-5])
    plt.xlim([0, 0.45])
    separation_for_plot = WA.to(u.arcsec)
    plt.scatter(separation_for_plot, 10**(dmag_current/-2.5), s = 0.1, alpha=0.1,  label='True values')
    # plt.scatter(separation_for_plot, 10**(dmag_current_lambertian/2.5), s = 0.1,  label='Lambertian value')
    plt.scatter(WAs, WA_contrasts, s = 0.1,  label='Limiting values')
    plt.xlabel('Separation (")')
    plt.ylabel('Flux Ratio')
    plt.yscale('log')
    plt.title(f'{(round(2026 + (t-mission_times[0]).to(u.yr).value,3)):.2f}')
    # plt.legend()
    plt.tight_layout()
    fname = f'plots/sep_flux{i:04}.png'
    plt.savefig(fname, dpi=250)
    print(f'\r {i}/{len(mission_times)}')

    # Plot for the p_phi evolution
    plt.figure()
    plt.hist(p_phi, **kwargs)
    plt.xlabel(r'$p \Phi(\beta)$')
    plt.xlim([0, 1])
    plt.ylim([0,35000])
    plt.title(f'{(round(2026 + (t-mission_times[0]).to(u.yr).value,3)):.2f}')
    plt.tight_layout()
    fname = f'plots/p_phi{i:04}.png'
    plt.savefig(fname, dpi=250)

    # Plot for the beta evolution
    plt.figure()
    plt.hist(beta.to(u.deg).value, **kwargs)
    plt.xlabel(r'$\beta$ (deg)')
    plt.xlim([0,180])
    plt.ylim([0,35000])
    plt.title(f'{(round(2026 + (t-mission_times[0]).to(u.yr).value,3)):.2f}')
    plt.tight_layout()
    fname = f'plots/beta{i:04}.png'
    plt.savefig(fname, dpi=250)
    plt.close('all')





# Convert times to years from mission start
mission_times_yr = ((mission_times-missionStart)*u.d.to(u.yr)).value

# Plot of the geometric constraints
plt.figure()
plt.plot(mission_times_yr, probs_bowtie, label='bowtie (65 deg)')
plt.plot(mission_times_yr, probs_WA, label='Working angles')
plt.plot(mission_times_yr, probs_bowtie_roll, label='bowtie (65 deg) + roll (26 deg)')
plt.title('Geometric constraints')
plt.xlabel('t, (years since mission launch)')
plt.ylabel('Probability the planet meets constraints')
plt.legend()

# Plot of all constraints combined
plt.figure()
plt.plot(mission_times_yr, probs_dmag)
plt.title('Photometric constraint')
plt.xlabel('t, (years since mission launch)')
plt.ylabel('Probability of being bright enough')
# Plot of all constraints combined
plt.figure()
plt.plot(mission_times_yr, probs_total_no_roll, label='Bowtie')
plt.plot(mission_times_yr, probs_total_roll, label='Bowtie + roll')
plt.title('Combined constraints')
plt.xlabel('t, (years since mission launch)')
plt.ylabel('Probability of meeting geometric and photometric constraints')
plt.legend()



#### Need to create table grid

#inside bowtie and bright enough
#inside bowtie and too dim
#outside bowtie and bright enough
#outside bowtie and too dim

# Create the plot of separation vs flux ratio
contrast_for_plot = 10**(dmags/-2.5)
separation_for_plot = WA.to(u.arcsec)
fig, ax = plt.subplots()
plt.ylim([10**-12, 10**-5])
plt.scatter(separation_for_plot, contrast_for_plot, s = 0.1,  label='True value')
plt.scatter(separation_for_plot, contrast_current, s = 0.1,  label='Limiting values')
plt.xlabel('Separation (")')
plt.ylabel('Flux Ratio')
plt.yscale('log')
plt.legend()
# plt.show()

#### Plot Probability of Detection In Bowtie vs Mission Year
if IWantPlots:
    data = np.asarray([[0,0.227],[1,0.226],[3,0.229],[3.5,0.226],[4,0.224],[5,0.227],[6,0.226]])
    plt.close(4999)
    plt.figure(4999)
    plt.scatter(data[:,0],data[:,1],color='black')
    plt.ylim([0,1])
    plt.ylabel(r'P(In Bowtie & $\delta$Mag)',weight='bold')
    plt.xlabel('Year Past 1/1/2026',weight='bold')
    plt.show(block=False)

#### Make Density Plots ##########################################################
if IWantPlots:
    #### dMag Density Plot
    plt.close(5000)
    fig = plt.figure(5000)
    plt.style.use('seaborn-white')
    kwargs = dict(histtype='stepfilled', alpha=0.8, bins=40, ec="k")#density=True

    dmag_histInBowtie_inds = np.where(dmags*numObservablePlanetsInBowtie)[0]
    dmag_histInRoll_inds = np.where(dmags*numObservablePlanetsInBowtieRoll)[0]
    dmag_histInAZ_inds = np.where(dmags*numObservablePlanetsKnownAZ)[0]

    plt.hist(dmags, label='All', **kwargs)
    plt.hist(dmags[dmag_histInAZ_inds], label='Known Az.', **kwargs)
    plt.hist(dmags[dmag_histInRoll_inds], label='Bowtie+Roll', **kwargs)
    plt.hist(dmags[dmag_histInBowtie_inds], label='Bowtie', **kwargs)

    plt.legend()
    plt.xlabel(r'$\Delta$mag',weight='bold')
    plt.ylabel('Count',weight='bold')
    plt.show(block=False)
    #### WA Density Plot
    plt.close(5001)
    fig = plt.figure(5001)
    plt.style.use('seaborn-white')
    kwargs = dict(histtype='stepfilled', alpha=0.8, bins=40, ec="k")#density=True

    WA_histInBowtie_inds = np.where(WA.value*numObservablePlanetsInBowtie)[0]
    WA_histInRoll_inds = np.where(WA.value*numObservablePlanetsInBowtieRoll)[0]
    WA_histInAZ_inds = np.where(WA.value*numObservablePlanetsKnownAZ)[0]

    plt.hist(WA.value, label='All', **kwargs)
    plt.hist(WA[WA_histInAZ_inds].value, label='Known Az.', **kwargs)
    plt.hist(WA[WA_histInRoll_inds].value, label='Bowtie+Roll', **kwargs)
    plt.hist(WA[WA_histInBowtie_inds].value, label='Bowtie', **kwargs)

    plt.legend()
    plt.xlabel('WA in (rad)',weight='bold')
    plt.ylabel('Count',weight='bold')
    plt.show(block=False)
    #### AZ Density Plot
    plt.close(5002)
    fig = plt.figure(5002)
    plt.style.use('seaborn-white')
    kwargs = dict(histtype='stepfilled', alpha=0.8, bins=40, ec="k")#density=True

    az_histInBowtie_inds = np.where(az.value*numObservablePlanetsInBowtie)[0]
    az_histInRoll_inds = np.where(az.value*numObservablePlanetsInBowtieRoll)[0]
    az_histInAZ_inds = np.where(az.value*numObservablePlanetsKnownAZ)[0]

    plt.hist(az.value, label='All', **kwargs)
    plt.hist(az[az_histInAZ_inds].value, label='Known Az.', **kwargs)
    plt.hist(az[az_histInRoll_inds].value, label='Bowtie+Roll', **kwargs)
    plt.hist(az[az_histInBowtie_inds].value, label='Bowtie', **kwargs)

    plt.legend()
    plt.xlabel('Azimuthal Angle in (rad)',weight='bold')
    plt.ylabel('Count',weight='bold')
    
    ##################################
    # Kepler orbital elements plots
    ##################################
    # fig = plt.figure()
    # plt.hist(a.to(u.AU).value, **kwargs)
    # plt.xlabel('Semi-major axis, a (AU)', weight='bold')
    # plt.ylabel('Count',weight='bold')
    
    # fig = plt.figure()
    # plt.hist(e, **kwargs)
    # plt.xlabel('Eccentricity, e', weight='bold')
    # plt.ylabel('Count',weight='bold')
    
    # fig = plt.figure()
    # plt.hist(inc, **kwargs)
    # plt.xlabel('Inclination, i (rad)', weight='bold')
    # plt.ylabel('Count',weight='bold')
    
    # fig = plt.figure()
    # plt.hist(w, **kwargs)
    # plt.xlabel('Argument of periapsis, w (rad)', weight='bold')
    # plt.ylabel('Count',weight='bold')

    # fig = plt.figure()
    # plt.hist(W.value, **kwargs)
    # plt.xlabel('Longitude of the ascending node, W (rad)', weight='bold')
    # plt.ylabel('Count',weight='bold')
    
    # fig = plt.figure()
    # plt.hist(M0.value, **kwargs)
    # plt.xlabel('Mean anomaly at mission start, M0 (rad)', weight='bold')
    # plt.ylabel('Count', weight='bold')

    # fig = plt.figure()
    # plt.hist(Rp.value, **kwargs)
    # plt.xlabel('Planet radius, Rp (Earth radii)', weight='bold')
    # plt.ylabel('Count', weight='bold')
    
    # fig = plt.figure()
    # plt.hist(p_phi, **kwargs)
    # plt.xlabel('Planet albedo multiplied by phase function',weight='bold')
    # plt.ylabel('Count', weight='bold')
    
    # fig = plt.figure()
    # plt.hist(Mp.to(u.M_jupiter).value, **kwargs)
    # plt.xlabel('Planet mass, Mp (Jupiter masses)', weight='bold')
    # plt.ylabel('Count', weight='bold')
    # plt.show(block=False)
