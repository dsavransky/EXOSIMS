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



import EXOSIMS,os.path
scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','WFIRST_47UMac.json')
import EXOSIMS.MissionSim
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)

PPop = sim.PlanetPopulation
PPM = sim.PlanetPhysicalModel

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


# =============================================================================
# keepout calculations
# =============================================================================
print('Creating a Fake Target List for HIP 53721')
from EXOSIMS.Prototypes.TargetList import TargetList
obs = sim.Observatory
missionStart = sim.TimeKeeping.missionStart  #Time Object

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

# one full year of run time
dtRange = np.arange(0,360,1)*u.d
oneFullYear = missionStart + dtRange
# star of interest
sInds = np.array([0])
# initializing arrays
koGood = np.zeros( oneFullYear.size)
culprit = np.zeros( [1,1,oneFullYear.size,12])
# calculating keepouts throguhout the year
for t,date in enumerate(oneFullYear):
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
n = 10**5
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
Msini = rvu.Msini(
    K, periods, star_mass, e, Msini_units="earth"
)  # Use RadVel.util.Msini to calculate M_planet
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

_, W, w = PPop.gen_angles(n, None)
# Time of periastron
T_c = samples[:, 2]
T_p = np.zeros(n)
eps = 0
for i in range(len(T_c)):
    T_p[i] = rvo.timetrans_to_timeperi(T_c[i], periods[i].value, e[i], w[i].value+W[i].value)

t_missionStart = 2461041 #JD 61041 #MJD 01/01/2026
nD = (t_missionStart - T_p)*u.d #number of days since t_periastron
nT = np.floor(nD/periods) #number of periods since t_periastron
fT = nD/periods - nT #fractional period past periastron
M0 = 2.*np.pi/periods*fT #Mean anomaly of the planet
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
p = PPM.calc_albedo_from_sma(a)
Rp = PPM.calc_radius_from_mass(Mp)

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
p = p[indsTooBig]
Rp = Rp[indsTooBig]

print('Done generating orbital parameters')

# =============================================================================
# Generating Random Orbits for the planet
# =============================================================================
#### Randomly Generate 47 UMa c planet parameters
# n = 10**5

# inc, W, w = PPop.gen_angles(n,None)
# inc = inc.to('rad').value
# inc[np.where(inc>np.pi/2)[0]] = np.pi - inc[np.where(inc>np.pi/2)[0]]
# W = W.to('rad').value
# a, e, p, Rp = PPop.gen_plan_params(n)
# a = a.to('AU').value
# if randomM0:
#     M0 = rand.uniform(low=0.,high=2*np.pi,size=n)#rand.random(360, size=n)
# elif periastronM0:
#     T = np.random.normal(loc=2391,scale=100) #orbital period 2391 +100 -87 days
#     t_periastron = np.random.normal(loc=2452441, scale=825,size=n) + nYears*365.25#2452441 +628 -825 in MJD
#     t_missionStart = 2461041 #JD 61041 #MJD 01/01/2026
#     nD = t_missionStart - t_periastron #number of days since t_periastron
#     nT = np.floor(nD/T) #number of days since t_periastron
#     fT = nT - nD/T #fractional period past periastron
#     M0 = 2.*np.pi/T*fT #Mean anomaly of the planet
# E = eccanom(M0, e)                      # eccentric anomaly

# #DELETEa = rand.uniform(low=3.5,high=3.7,size=n)*u.AU# (3.7-3.5)*rand.random(n)+3.5 #uniform randoma
# a = np.random.normal(loc=3.6,scale=0.1,size=n)*u.AU
# #DELETEe = rand.uniform(low=0.002,high=0.145,size=n)#(0.145-0.002)*rand.random(n)+0.02 #uniform random
# e = np.random.rayleigh(scale=0.098+0.047,size=n)
# e[e>1] = 0
# #DELETEMsini = rand.uniform(low=0.467,high=0.606,size=n)#(0.606-0.467)*rand.random(n)+0.467
# Msini = np.random.normal(loc=0.54,scale=0.073,size=n)#+.066 -.073)
# #DELETEw = (rand.uniform(low=295-160,high=295+114,size=n)*u.deg).to('rad')
# w = (np.random.normal(loc=295,scale=160,size=n)*u.deg).to('rad')
# Mp = (Msini/np.sin(inc)*u.M_jup).to('M_earth')
# indsTooBig = np.where(Mp < (13*u.M_jup).to('M_earth'))[0] #throws out planets with mass 12x larger than jupiter https://www.discovermagazine.com/the-sciences/how-big-is-the-biggest-possible-planet
# Mp = Mp[indsTooBig]
# Rp = PPM.calc_radius_from_mass(Mp)
# #TODO CHECK FOR INF/TOO LARGE
# print('Done Generating planets 1')

# #DELETEindsTooBig = np.where(Rp < 12*u.earthRad)[0] #throws out planets with radius 12x larger than Earth
# a = a[indsTooBig]
# e = e[indsTooBig]
# w = w[indsTooBig]
# W = W[indsTooBig]
# inc = inc[indsTooBig]
# M0 = M0[indsTooBig]
# E = E[indsTooBig]
# mu = mu[indsTooBig]
# p = p[indsTooBig]
# Rp = Rp[indsTooBig]
# p = PPM.calc_albedo_from_sma(a)
# print('Done Generating Planets 2')

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

SU.a = a*u.AU               # semi-major axis
SU.e = e                              # eccentricity
SU.I = I*u.rad              # inclinations
SU.O = O*u.rad              # right ascension of the ascending node
SU.w = w*u.rad              # argument of perigee
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

#DELETEPPM.calc_beta(Phi)
Phi = PPM.calc_Phi(beta)
dmags = deltaMag(p,Rp,d,Phi)
print('Done calculating Phi, dmag')

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
                                       fill_value='extrapolate', bounds_error=False)

# # Find the dMag value
# dmags = np.zeros(len(WA))
# for i, wa in enumerate(WA):
#     if not pInIWAOWA[i]:
#         # If it's outside of the working area then just assign a dMag of infinity
#         # so that it's removed in the bright enough check
#         dmags[i] = np.nan
#         continue
#     contrast = contrast_interp(wa)
#     dmags[i] = -2.5*np.log10(contrast)
contrasts = contrast_interp(WA) 
contrasts[contrasts < 1e-11] = 1e-2
dmagLims = -2.5*np.log10(contrasts)
print('Done calculating dMag')

# =============================================================================
# Contrast/Brightness Filter
# =============================================================================

ZL = sim.ZodiacalLight
TL.starMag = lambda sInds, lam: 5.03 #Apparent StarMag from wikipedia
sInds = np.zeros(len(pInIWAOWA))+0
mode = mode = list(filter(lambda mode: mode['detectionMode'] == True, OS.observingModes))[0]
#dmagLims = OS.calc_dMag_per_intTime( (np.zeros(len(sInds))+ 10**4)*u.d, TL, sInds, (np.zeros(len(sInds))+ZL.fZ0.value)*ZL.fZ0.unit, (np.zeros(len(sInds))+ZL.fEZ0.value)*ZL.fEZ0.unit, WA, mode, C_b=None, C_sp=None)
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



#### SPEC dmitry gave us
#l/D    contr_snr10  lambda t_int_hr  fpp   SNR
#Short caption: WFIRST CGI spectroscopy prediction: Modeled 10-sigma post-processed [RDI+fpp=2] contrast curve for Band 3 spectroscopy with the SPC bowtie, based on OS9. The model observation use MUFs=1 and no margins and component performance estimates for 21mo into mission. Integration time is  400hr. V=5 star. (source: B. Nemati, March 16, 2020 per. comm.)
#References: 20200316 B. Nemati spreadsheet
#3.0 4.5E-09 730 400 2   10
#3.5 3.8E-09 730 400 2   10
#4.0 2.8E-09 730 400 2   10
#6.0 2.8E-09 730 400 2   10
#7.0 2.8E-09 730 400 2   10
#8.0 3.4E-09 730 400 2   10
#9.0 5.6E-09 730 400 2   10

#### Need to create table grid

#inside bowtie and bright enough
#inside bowtie and too dim
#outside bowtie and bright enough
#outside bowtie and too dim

#### Create Single Keep-out map for 47 UMa


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
    plt.show(block=False)
    ####






