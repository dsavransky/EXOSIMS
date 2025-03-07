# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight
import numpy as np
import os
import astropy.units as u
from astropy.coordinates import SkyCoord
import pickle
from astropy.time import Time
from synphot import units
import sys


class Stark(ZodiacalLight):
    """Stark Zodiacal Light class

    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation using
    the model from Stark et al. 2014.

    """

    def __init__(self, magZ=23.0, magEZ=22.0, varEZ=0.0, **specs):
        ZodiacalLight.__init__(self, magZ, magEZ, varEZ, **specs)

        self.global_min = np.min(self.zodi_values)

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
            ~astropy.units.Quantity(~numpy.ndarray(float)):
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

        # First get intensities at reference values
        Izod = self.zodi_intensity_at_location(lon * u.deg, lat * u.deg)
        # Now correct for color
        Izod *= self.zodi_color_correction_factor(mode["lam"])

        # convert to photon units
        Izod_photons = (
            units.convert_flux(mode["lam"], Izod * u.sr, units.PHOTLAM) / u.sr
        )

        # finally, scale by mode's zero mag flux
        fZ = (Izod_photons / (mode["F0"] / mode["deltaLam"])).to("1/arcsec2")

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
            koTimes (~astropy.time.Time(~numpy.ndarray(float)), optional):
                Absolute MJD mission times from start to end in steps of 1 d

        Returns:
            tuple:
                valfZmax[sInds] (~astropy.units.Quantity(~numpy.ndarray(float))):
                    the maximum fZ with units 1/arcsec**2
                absTimefZmax[sInds] (astropy.time.Time):
                    returns the absolute Time the maximum fZ occurs
        """
        # Generate cache Name
        cachefname = hashname + "fZmax"

        if koTimes is None:
            koTimes = self.fZTimes

        # Check if file exists
        if os.path.isfile(cachefname):  # check if file exists
            self.vprint("Loading cached fZmax from %s" % cachefname)
            with open(cachefname, "rb") as f:  # load from cache
                tmpDat = pickle.load(f)

                valfZmax = tmpDat[0, :]
                absTimefZmax = Time(tmpDat[1, :], format="mjd", scale="tai")
            return valfZmax[sInds] / u.arcsec**2, absTimefZmax[sInds]  # , fZmaxInds

        # IF the fZmax File Does Not Exist, Calculate It
        else:
            tmpfZ = np.asarray(self.fZMap[mode["syst"]["name"]])
            fZ_matrix = tmpfZ[sInds, :]  # Apply previous filters to fZMap

            # Find maximum fZ of each star
            valfZmax = np.zeros(sInds.shape[0])
            absTimefZmax = np.zeros(sInds.shape[0])
            for i in range(len(sInds)):
                valfZmax[i] = max(fZ_matrix[i, :])  # fZ_matrix has dimensions sInds
                indfZmax = np.where(np.array(valfZmax[i], ndmin=1))
                # indices where fZmin occurs:
                absTimefZmax[i] = koTimes[indfZmax].value

            with open(cachefname, "wb") as fo:
                pickle.dump({"fZmaxes": valfZmax, "fZmaxTimes": absTimefZmax}, fo)
                self.vprint("Saved cached fZmax to %s" % cachefname)

            absTimefZmax = Time(absTimefZmax, format="mjd", scale="tai")
            return valfZmax / u.arcsec**2, absTimefZmax  # , fZmaxInds

    def calcfZmin(self, sInds, Obs, TL, TK, mode, hashname, koMap=None, koTimes=None):
        """Finds the minimum zodiacal light values for each star over an entire orbit
        of the sun not including keeoput angles

        Args:
            sInds (~numpy.ndarray(int)):
                the star indicies we would like fZmin and fZminInds returned for
            Obs (module):
                Observatory module
            TL (module):
                Target List Module
            TK (TimeKeeping object):
                TimeKeeping object
            mode (dict):
                Selected observing mode
            hashname (str):
                hashname describing the files specific to the current json script
            koMap (boolean ndarray, optional):
                True is a target unobstructed and observable, and False is a
                target unobservable due to obstructions in the keepout zone.
            koTimes (~astropy.time.Time(~numpy.ndarray(float)), optional):
                Absolute MJD mission times from start to end in steps of 1 d

        Returns:
            tuple:
                fZmins[n, TL.nStars] (~astropy.units.Quantity(~numpy.ndarray(float))):
                    fZMap, but only fZmin candidates remain. All other values are set to
                    the maximum floating number. Units are 1/arcsec2
                fZtypes [n, TL.nStars] (~numpy.ndarray(float)):
                    ndarray of flags for fZmin types that map to fZmins
                    0 - entering KO
                    1 - exiting KO
                    2 - local minimum
                    max float - not a fZmin candidate

        """

        # Generate cache Name
        cachefname = hashname + "fZmin"

        # Check if file exists
        if os.path.isfile(cachefname):  # check if file exists
            self.vprint("Loading cached fZmins from %s" % cachefname)
            with open(cachefname, "rb") as f:  # load from cache
                tmp1 = pickle.load(f)
                fZmins = tmp1["fZmins"]
                fZtypes = tmp1["fZtypes"]
            return fZmins, fZtypes
        else:
            tmpAssert = np.any(self.fZMap[mode["syst"]["name"]])
            assert tmpAssert, "fZMap does not exist for the mode of interest"

            tmpfZ = np.asarray(self.fZMap[mode["syst"]["name"]])
            fZ_matrix = tmpfZ[sInds, :]  # Apply previous filters to fZMap
            # When are stars in KO regions
            # if this is being calculated without a koMap
            if koMap is None:
                koTimes = self.fZTimes

                # calculating keepout angles and keepout values for 1 system in mode
                koStr = list(
                    filter(
                        lambda syst: syst.startswith("koAngles_"), mode["syst"].keys()
                    )
                )
                koangles = np.asarray([mode["syst"][k] for k in koStr]).reshape(1, 4, 2)
                kogoodStart = Obs.keepout(TL, sInds, koTimes[0], koangles)[0].T
                nn = len(sInds)
                mm = len(koTimes)
            else:
                # getting the correct koTimes to look up in koMap
                assert (
                    koTimes is not None
                ), "Corresponding koTimes not included with koMap."
                kogoodStart = koMap.T
                [nn, mm] = np.shape(koMap)

            fZmins = np.ones([nn, mm]) * sys.float_info.max
            fZtypes = np.ones([nn, mm]) * sys.float_info.max
            # Find inds Entering, exiting ko
            # i = 0 # star ind
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
                fZlocalMinInds = (
                    np.where(np.diff(np.sign(np.diff(fZ_matrix[i, :]))) > 0)[0] + 1
                )  # Find local minima of fZ, +1 to correct for indexing offset
                # Filter where local minima occurs in keepout region
                fZlocalMinInds = [ind for ind in fZlocalMinInds if kogoodStart[ind, i]]

                # Remove any indsEntering/indsExiting from fZlocalMinInds
                tmp1 = set(list(indsEntering) + list(indsExiting))
                # remove anything in tmp1 from fZlocalMinInds
                fZlocalMinInds = list(set(list(fZlocalMinInds)) - tmp1)

                minInds = (
                    np.append(np.append(indsEntering, indsExiting), fZlocalMinInds)
                ).astype(int)

                if np.any(minInds):
                    fZmins[i, minInds] = fZ_matrix[i, minInds]
                    fZtypes[i, indsEntering] = 0
                    fZtypes[i, indsExiting] = 1
                    fZtypes[i, fZlocalMinInds] = 2

            with open(cachefname, "wb") as fo:
                pickle.dump({"fZmins": fZmins, "fZtypes": fZtypes}, fo)
                self.vprint("Saved cached fZmins to %s" % cachefname)

            return fZmins, fZtypes

    def global_zodi_min(self, mode):
        """
        This is used to determine the minimum zodi value globally with a color
        correction

        Args:
            mode (dict):
                Selected observing mode

        Returns:
            ~astropy.units.Quantity:
                The global minimum zodiacal light value for the observing mode,
                in (1/arcsec**2)
        """

        fZminglobal = self.global_min * self.zodi_color_correction_factor(mode["lam"])

        # convert to photon units
        fZminglobal = (
            units.convert_flux(mode["lam"], fZminglobal * u.sr, units.PHOTLAM) / u.sr
        )

        # finally, scale by mode's zero mag flux
        fZminglobal = (fZminglobal / (mode["F0"] / mode["deltaLam"])).to("1/arcsec2")

        return fZminglobal
