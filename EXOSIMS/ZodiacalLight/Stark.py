# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight
import numpy as np
import os
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata
import pickle
from astropy.time import Time
import sys


class Stark(ZodiacalLight):
    """Stark Zodiacal Light class

    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation using
    the model from Stark et al. 2014.

    """

    def __init__(self, magZ=23.0, magEZ=22.0, varEZ=0.0, **specs):
        """ """
        ZodiacalLight.__init__(self, magZ, magEZ, varEZ, **specs)
        (
            self.points,
            self.values,
        ) = (
            self.calcfbetaInput()
        )  # looking at certain lat/long rel to antisolar point, create interpolation
        # grid. in old version, do this for a certain value
        # Here we calculate the Zodiacal Light Model

        self.global_min = np.min(self.values)

    def fZ(self, Obs, TL, sInds, currentTimeAbs, mode):
        """Returns surface brightness of local zodiacal light

        Args:
            Obs (Observatory module):
                Observatory class object
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            currentTimeAbs (astropy Time array):
                Current absolute mission time in MJD
            mode (dict):
                Selected observing mode

        Returns:
            fZ (astropy Quantity array):
                Surface brightness of zodiacal light in units of 1/arcsec2

        """

        # observatory positions vector in heliocentric ecliptic frame
        r_obs = Obs.orbit(currentTimeAbs, eclip=True)
        # observatory distance from heliocentric ecliptic frame center
        # (projected in ecliptic plane)
        try:
            r_obs_norm = np.linalg.norm(r_obs[:, 0:2], axis=1)
            # observatory ecliptic longitudes
            r_obs_lon = (
                np.sign(r_obs[:, 1])
                * np.arccos(r_obs[:, 0] / r_obs_norm).to("deg").value
            )  # ensures the longitude is +/-180deg
        except:  # noqa: E722
            r_obs_norm = np.linalg.norm(r_obs[:, 0:2], axis=1) * r_obs.unit
            # observatory ecliptic longitudes
            r_obs_lon = (
                np.sign(r_obs[:, 1])
                * np.arccos(r_obs[:, 0] / r_obs_norm).to("deg").value
            )  # ensures the longitude is +/-180deg

        # longitude of the sun
        lon0 = (
            r_obs_lon + 180.0
        ) % 360.0  # turn into 0-360 deg heliocentric ecliptic longitude of spacecraft

        # target star positions vector in heliocentric true ecliptic frame
        r_targ = TL.starprop(sInds, currentTimeAbs, eclip=True)
        # target star positions vector wrt observatory in ecliptic frame
        r_targ_obs = (r_targ - r_obs).to("pc").value
        # tranform to astropy SkyCoordinates
        coord = SkyCoord(
            r_targ_obs[:, 0],
            r_targ_obs[:, 1],
            r_targ_obs[:, 2],
            representation_type="cartesian",
        ).represent_as("spherical")

        # longitude and latitude absolute values for Leinert tables
        lon = coord.lon.to("deg").value - lon0  # Get longitude relative to spacecraft
        lat = coord.lat.to("deg").value  # Get latitude relative to spacecraft
        lon = abs((lon + 180.0) % 360.0 - 180.0)  # converts to 0-180 deg
        lat = abs(lat)
        # technically, latitude is physically capable of being >90 deg

        # Interpolates 2D
        fbeta = griddata(self.points, self.values, list(zip(lon, lat)))

        lam = mode["lam"]  # extract wavelength
        BW = mode["BW"]  # extract bandwidth

        f = (
            10.0 ** (self.logf(np.log10(lam.to("um").value)))
            * u.W
            / u.m**2
            / u.sr
            / u.um
        )
        h = const.h  # Planck constant
        c = const.c  # speed of light in vacuum
        ephoton = h * c / lam / u.ph  # energy of a photon
        F0 = TL.F0(BW, lam)  # zero-magnitude star (sun) (in ph/s/m2/nm)
        f_corr = f / ephoton / F0  # color correction factor
        fZ = fbeta * f_corr.to("1/arcsec2")

        return fZ

    def calcfZmax(self, sInds, Obs, TL, TK, mode, hashname, koTimes=None):
        """Finds the maximum zodiacal light values for each star over an entire
        orbit of the sun not including keeoput angles

        Args:
            sInds (integer array):
                the star indicies we would like fZmax and fZmaxInds returned for
            Obs (module):
                Observatory module
            TL (TargetList object):
                Target List Module
            TK (TimeKeeping object):
                TimeKeeping object
            mode (dict):
                Selected observing mode
            hashname (string):
                hashname describing the files specific to the current json script
        Returns:
            valfZmax[sInds] (astropy Quantity array):
                the maximum fZ
            absTimefZmax[sInds] (astropy Time array):
                returns the absolute Time the maximum fZ occurs (for the prototype,
                these all have the same value)

        """
        # Generate cache Name
        cachefname = hashname + "fZmax"

        if koTimes is None:
            koTimes = self.fZTimes
            
        # Check if file exists
        if os.path.isfile(cachefname):  # check if file exists
            self.vprint("Loading cached fZmax from %s" % cachefname)
            with open(cachefname, "rb") as f:  # load from cache
                try:
                    tmpDat = pickle.load(f)
                except UnicodeDecodeError:
                    tmpDat = pickle.load(f, encoding="latin1")

                valfZmax = tmpDat[0, :]
                absTimefZmax = Time(tmpDat[1, :], format="mjd", scale="tai")
            return valfZmax[sInds] / u.arcsec**2, absTimefZmax[sInds]  # , fZmaxInds

        # IF the fZmax File Does Not Exist, Calculate It
        else:
            tmpfZ = np.asarray(self.fZMap[mode['syst']['name']])
            fZ_matrix = tmpfZ[sInds,:] #Apply previous filters to fZMap[sInds, 1000]
        
            #Find maximum fZ of each star
            valfZmax = np.zeros(sInds.shape[0])
            absTimefZmax = np.zeros(sInds.shape[0])
            for i in range(len(sInds)):
                valfZmax[i] = max(fZ_matrix[i,:])#fZ_matrix has dimensions sInds
                indfZmax = np.where(valfZmax[i])    #Gets indices where fZmin occurs
                absTimefZmax[i] = koTimes[indfZmax].value

            with open(cachefname, "wb") as fo:
                pickle.dump({"fZmaxes": valfZmax, "fZmaxTimes": absTimefZmax},fo)
                self.vprint("Saved cached fZmax to %s" % cachefname)
                
            absTimefZmax = Time(absTimefZmax, format="mjd", scale="tai")
            return valfZmax / u.arcsec**2, absTimefZmax  # , fZmaxInds

    def calcfZmin(self, sInds, Obs, TL, TK, mode, hashname, koMap=None, koTimes=None):
        """Finds the minimum zodiacal light values for each star over an entire orbit
        of the sun not including keeoput angles

        Args:
            sInds[sInds] (integer array):
                the star indicies we would like fZmin and fZminInds returned for
            Obs (module):
                Observatory module
            TL (module):
                Target List Module
            TK (TimeKeeping object):
                TimeKeeping object
            mode (dict):
                Selected observing mode
            hashname (string):
                hashname describing the files specific to the current json script
            koMap (boolean ndarray):
                True is a target unobstructed and observable, and False is a
                target unobservable due to obstructions in the keepout zone.
            koTimes (astropy Time ndarray):
                Absolute MJD mission times from start to end in steps of 1 d

        Returns:
            list:
                list of local zodiacal light minimum and times they occur at
                (should all have same value for prototype)

        """

        # Generate cache Name
        cachefname = hashname + "fZmin"
        
        #Check if file exists#######################################################################
        if os.path.isfile(cachefname):#check if file exists
            self.vprint("Loading cached fZmins from %s"%cachefname)
            with open(cachefname, 'rb') as f:#load from cache
                try:
                    tmp1 = pickle.load(f)  # of form tmpDat len sInds, tmpDat[0] len # of ko enter/exits and localmin occurences, tmpDat[0,0] form [type,fZvalue,absTime]
                except UnicodeDecodeError:
                    tmp1 = pickle.load(f,encoding='latin1')  # of form tmpDat len sInds, tmpDat[0] len # of ko enter/exits and localmin occurences, tmpDat[0,0] form [type,fZvalue,absTime]
                fZmins = tmp1['fZmins']
                fZtypes = tmp1['fZtypes']
            return fZmins, fZtypes
        else:
            assert np.any(self.fZMap[mode['syst']['name']]) == True, "fZMap does not exist for the mode of interest"

            tmpfZ = np.asarray(self.fZMap[mode['syst']['name']])
            fZ_matrix = tmpfZ[sInds,:] #Apply previous filters to fZMap[sInds, 1000]
#            pdb.set_trace()
            #When are stars in KO regions
            missionLife = TK.missionLife.to('yr')
            # if this is being calculated without a koMap, or if missionLife is less than a year
            if (koMap is None) or (missionLife.value < 1):
                koTimes = self.fZTimes
                # calculating keepout angles and keepout values for 1 system in mode
                koStr     = list(filter(lambda syst: syst.startswith('koAngles_') , mode['syst'].keys()))
                koangles  = np.asarray([mode['syst'][k] for k in koStr]).reshape(1,4,2)
                kogoodStart = Obs.keepout(TL, sInds, koTimes, koangles)[0].T
                nn = len(sInds)
                if koTimes is None:
                    mm = len(self.fZTimes)
                else:
                    mm = len(koTimes)
            else:
                # getting the correct koTimes to look up in koMap
                assert koTimes != None, "koTimes not included in input statement."
                kogoodStart = koMap.T
                [nn,mm] = np.shape(koMap)
                
            fZmins = np.ones([nn,mm])*sys.float_info.max
            fZtypes = np.ones([nn,mm])*sys.float_info.max
            # Find inds Entering, exiting ko
            #i = 0 # star ind
            for k in np.arange(len(sInds)):
                i = sInds[k]  # Star ind

                # double check this is entering
                indsEntering = list(
                    np.where(np.diff(kogoodStart[:, i].astype(int)) == -1.0)[0]
                )

                # without the +1, this gives kogoodStart[indsExiting,i] = 0 meaning
                # the stars are still in keepout
                indsExiting = (
                    np.where(np.diff(kogoodStart[:, i].astype(int)) == 1.0)[0] + 1
                )
                indsExiting = [
                    indsExiting[j] if indsExiting[j] < len(kogoodStart[:, i]) - 1 else 0
                    for j in np.arange(len(indsExiting))
                ]  # need to ensure +1 increment doesnt exceed kogoodStart size

                # Find inds of local minima in fZ
                fZlocalMinInds = np.where(
                    np.diff(np.sign(np.diff(fZ_matrix[i, :]))) > 0
                )[
                    0
                ]  # Find local minima of fZ
                # Filter where local minima occurs in keepout region
                fZlocalMinInds = [ind for ind in fZlocalMinInds if kogoodStart[ind, i]]

                # Remove any indsEntering/indsExiting from fZlocalMinInds
                tmp1 = set(list(indsEntering) + list(indsExiting))
                # remove anything in tmp1 from fZlocalMinInds
                fZlocalMinInds = list(set(list(fZlocalMinInds)) - tmp1)

                minInds = (np.append(np.append(indsEntering, indsExiting),fZlocalMinInds)).astype(int)
                
                if np.any(minInds):
                    fZmins[i,minInds] = fZ_matrix[i,minInds]
                    fZtypes[i,indsEntering] = 0
                    fZtypes[i,indsExiting] = 1
                    fZtypes[i,fZlocalMinInds] = 2

            with open(cachefname, "wb") as fo:
                pickle.dump({"fZmins": fZmins, "fZtypes": fZtypes},fo)
                self.vprint("Saved cached fZmins to %s"%cachefname)

            return fZmins, fZtypes


    def global_zodi_min(self, mode):
        """
        This is used to determine the minimum zodi value globally with a color
        correction

        Args:
            mode (dict):
                Selected observing mode

        Returns:
            fZminglobal (astropy Quantity):
                The global minimum zodiacal light value for the observing mode,
                in (1/arcsec**2)
        """

        lam = mode["lam"]

        f = (
            10.0 ** (self.logf(np.log10(lam.to("um").value)))
            * u.W
            / u.m**2
            / u.sr
            / u.um
        )
        h = const.h
        c = const.c

        # energy of a photon
        ephoton = h * c / lam / u.ph

        # zero-magnitude star (sun) (in ph/s/m2/nm)
        F0 = (
            1e4 * 10 ** (4.01 - (lam / u.nm - 550) / 770) * u.ph / u.s / u.m**2 / u.nm
        )

        # color correction factor
        f_corr = f / ephoton / F0

        fZminglobal = self.global_min * f_corr.to("1/arcsec2")

        return fZminglobal
