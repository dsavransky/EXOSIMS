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


import EXOSIMS,os.path
scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','WFIRST_47UMac.json')
import EXOSIMS.MissionSim
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)

PPop = sim.PlanetPopulation
PPM = sim.PlanetPhysicalModel

### !!!!!!!!!
### Make this value True if you want to output plots
IWantPlots = True

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
koGood = np.zeros( oneFullYear.size)
culprit = np.zeros( [1,1,oneFullYear.size,12])
# calculating keepouts throguhout the year
for t,date in enumerate(oneFullYear):
    koGood[t],r_body, r_targ, culprit[:,:,t,:], koangleArray = obs.keepout(fTL, sInds, date, koangles, returnExtra=True)

### Plotting!
if IWantPlots:
    keepoutString = """Sun KO = [%.0f,%.0f],
    Earth KO = [%.0f,%.0f],
    Moon KO = [%.0f,%.0f],
    Small Planet KO = [%.0f,%.0f],
    Solar Panel KO = [%.0f,%.0f]
    """ %(koangles[0,0,0],koangles[0,0,1], koangles[0,1,0],koangles[0,1,1], koangles[0,2,0],koangles[0,2,1], koangles[0,3,0],koangles[0,3,1],obs.koAngles_SolarPanel[0].value,obs.koAngles_SolarPanel[1].value)
    
    plt.figure(figsize=(10,8))
    plt.plot(dtRange , np.sum(culprit,axis=-1)[0,0],linewidth=3,label=keepoutString)
    plt.xlabel("Time (d)",labelpad=16, fontsize=14,fontweight='bold')
    plt.ylabel("Number of Culprits Causing Keepout",labelpad=16, fontsize=14,fontweight='bold')
    plt.title("Fraction of Time at which star is Observable = %0.2f" % (np.sum(koGood) / dtRange.size) )
    plt.legend(loc='best')


# =============================================================================
# Dean's Stuff
# =============================================================================
#### Randomly Generate 47 UMa c planet parameters
n = 10**5
inc, W, w = PPop.gen_angles(n,None)
inc = inc.to('rad').value
inc[np.where(inc>np.pi/2)[0]] = np.pi - inc[np.where(inc>np.pi/2)[0]]
W = W.to('rad').value
w = np.zeros(len(W)) + w#w.to('rad').value
a, e, p, Rp = PPop.gen_plan_params(n)
a = a.to('AU').value
M0 = rand.uniform(low=0.,high=2*np.pi,size=n)#rand.random(360, size=n)
E = eccanom(M0, e)                      # eccentric anomaly

a = rand.uniform(low=3.5,high=3.7,size=n)*u.AU# (3.7-3.5)*rand.random(n)+3.5 #uniform random
e = rand.uniform(low=0.002,high=0.145,size=n)#(0.145-0.002)*rand.random(n)+0.02 #uniform random
Msini = rand.uniform(low=0.467,high=0.606,size=n)#(0.606-0.467)*rand.random(n)+0.467
Mp = (Msini/np.sin(inc)*u.M_jup).to('M_earth')
#TODO CHECK FOR INF/TOO LARGE
print('Done Generating planets 1')

Rp = PPM.calc_radius_from_mass(Mp)
indsTooBig = np.where(Rp < 12*u.earthRad)[0] #throws out planets with radius 12x larger than Earth
Rp = Rp[indsTooBig]
a = a[indsTooBig]
e = e[indsTooBig]
w = w[indsTooBig]
W = W[indsTooBig]
inc = inc[indsTooBig]
M0 = M0[indsTooBig]
E = E[indsTooBig]
Mp = Mp[indsTooBig]
p = PPM.calc_albedo_from_sma(a)
print('Done Generating Planets 2')

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

mu = const.G*(Mp + star_mass*u.M_sun)#TL.MsTrue[self.plan2star])
v1 = np.sqrt(mu/a**3)/(1 - e*np.cos(E))
v2 = np.cos(E)

r = (A*r1 + B*r2).T.to('AU')  
d = np.linalg.norm(r, axis=1)
beta = np.arccos(r[:,2]/d)
print('Done calculating r and beta stuff')


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

# Azimuth
az = np.arctan2(SU.r[:,1],SU.r[:,0]) #azimuth angle in XY #ranges from -pi to pi
print('Done calculating az')

#DELETEPPM.calc_beta(Phi)
Phi = PPM.calc_Phi(beta)
dmags = deltaMag(p,Rp,d,Phi)
print('Done calculating Phi, dmag')

#bowtie limits??? # pulled from my limited knowledge of the bowtie
az_lim1 = (np.pi-np.pi/6)*u.rad
az_lim2 = (np.pi/6)*u.rad
az_lim3 = (-np.pi/6)*u.rad
az_lim4 = (-np.pi+np.pi/6)*u.rad

#Check if the planet is in the bowtie
pInBowtie1 = (az > az_lim1)*(az < az_lim4)
pInBowtie2 = (az < az_lim2)*(az > az_lim3)
pInBowtie = np.logical_or(pInBowtie1,pInBowtie2)
print('Done checking in bowtie')

#Calculate Planet WA's
WA = (SU.s/TL.dist).decompose()*u.rad
print('Done calculating planet WA')

#OWA IWA che k
pInIWAOWA = (WA > OS.IWA.to('rad'))*(WA < OS.OWA.to('rad'))# Outside of IWA and inside of OWA
print('Done checking in IWA OWA')

ZL = sim.ZodiacalLight
TL.starMag = lambda sInds, lam: 5.03 #Apparent StarMag from wikipedia
sInds = np.zeros(len(pInIWAOWA))+0
mode = mode = list(filter(lambda mode: mode['detectionMode'] == True, OS.observingModes))[0]
dmagLims = OS.calc_dMag_per_intTime( np.zeros(len(sInds))+ 10**4*u.d, TL, sInds, np.zeros(len(sInds))+ZL.fZ0, np.zeros(len(sInds))+ZL.fEZ0, WA, mode, C_b=None, C_sp=None)
pBrightEnough = dmags < dmagLims
print('Done checking bright enough')

#ANSWER
numObservablePlanetsInBowtie = pInBowtie*pInIWAOWA*pBrightEnough
fracObservablePlanetsInBowtie = np.count_nonzero(numObservablePlanetsInBowtie)/len(indsTooBig)
print(fracObservablePlanetsInBowtie)



### Creating interpolant to work with our l/D value
# Getting the csv file with the contrast curve loaded into numpy arrays
contrast_curve_filename = 'WFIRST_47UMac_Contrast.csv'
contrast_curve_file = os.path.join(os.path.normpath(os.path.expandvars(contrast_curve_filename)))
contrast_curve_table = np.genfromtxt(contrast_curve_file, delimiter=',', skip_header=1)

# Creating an interpolant to generate a contrast value for the given lambda/D
l_over_D_vals = contrast_curve_table[:,0]
contrast_vals = contrast_curve_table[:,1]
contrast_interp = interpolate.interp1d(l_over_D_vals, contrast_vals, kind='cubic', fill_value=0., bounds_error=False)


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