import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from EXOSIMS.Prototypes.TargetList import TargetList
from EXOSIMS.util._numpy_compat import copy_if_needed


class EclipticTargetList(TargetList):
    """Target list in which star positions may be obtained in heliocentric equatorial
    or ecliptic coordinates.

    Args:
        **specs:
            user specified values

    """

    def __init__(self, **specs):

        TargetList.__init__(self, **specs)

    def nan_filter(self):
        """Populates Target List and filters out values which are nan"""

        # filter out nan values in numerical attributes
        for att in self.catalog_atts:
            if ("close" in att) or ("bright" in att):
                continue
            if getattr(self, att).shape[0] == 0:
                pass
            elif isinstance(getattr(self, att)[0], (str, bytes)):
                # FIXME: intent here unclear:
                #   note float('nan') is an IEEE NaN, getattr(.) is a str,
                #   and != on NaNs is special
                i = np.where(getattr(self, att) != float("nan"))[0]
                self.revise_lists(i)
            # exclude non-numerical types
            elif type(getattr(self, att)[0]) not in (
                np.unicode_,
                np.string_,
                np.bool_,
                bytes,
            ):
                if att == "coords":
                    i1 = np.where(~np.isnan(self.coords.lon.to("deg").value))[0]
                    i2 = np.where(~np.isnan(self.coords.lat.to("deg").value))[0]
                    i = np.intersect1d(i1, i2)
                else:
                    i = np.where(~np.isnan(getattr(self, att)))[0]
                self.revise_lists(i)

    def revise_lists(self, sInds):
        """Replaces Target List catalog attributes with filtered values,
        and updates the number of target stars.

        Args:
            sInds (integer ndarray):
                Integer indices of the stars of interest

        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)

        if len(sInds) == 0:
            raise IndexError("Target list filtered to empty.")

        for att in self.catalog_atts:
            if att == "coords":
                ra = self.coords.lon[sInds].to("deg")
                dec = self.coords.lat[sInds].to("deg")
                self.coords = SkyCoord(
                    ra, dec, self.dist.to("pc"), frame="barycentrictrueecliptic"
                )
            else:
                if getattr(self, att).size != 0:
                    setattr(self, att, getattr(self, att)[sInds])
        try:
            self.Completeness.revise_updates(sInds)
        except AttributeError:
            pass
        self.nStars = len(sInds)
        assert self.nStars, "Target list is empty: nStars = %r" % self.nStars

    def starprop(self, sInds, currentTime, eclip=False):
        """Finds target star positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).

        This method uses ICRS coordinates which is approximately the same as
        equatorial coordinates.

        Args:
            sInds (integer ndarray):
                Integer indices of the stars of interest
            currentTime (astropy Time):
                Current absolute mission time in MJD
            eclip (boolean):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to
                False, corresponding to heliocentric equatorial frame.

        Returns:
            r_targ (astropy Quantity array):
                Target star positions vector in heliocentric equatorial (default)
                or ecliptic frame in units of pc. Will return an m x n x 3 array
                where m is size of currentTime, n is size of sInds. If either m or
                n is 1, will return n x 3 or m x 3.

        Note: Use eclip=True to get ecliptic coordinates.

        """

        # if multiple time values, check they are different otherwise reduce to scalar
        if currentTime.size > 1:
            if np.all(currentTime == currentTime[0]):
                currentTime = currentTime[0]

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)

        # get all array sizes
        nStars = sInds.size
        nTimes = currentTime.size

        # if the starprop_static method was created (staticStars is True), then use it
        if self.starprop_static is not None:
            r_targ = self.starprop_static(sInds, currentTime, eclip)
            if nTimes == 1 or nStars == 1 or nTimes == nStars:
                return r_targ
            else:
                return np.tile(r_targ, (nTimes, 1, 1))

        # target star ICRS coordinates
        coord_old = self.coords[sInds]
        # right ascension and declination
        ra = coord_old.lon
        dec = coord_old.lat
        # directions
        p0 = np.array([-np.sin(ra), np.cos(ra), np.zeros(sInds.size)])
        q0 = np.array(
            [-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)]
        )
        r0 = coord_old.cartesian.xyz / coord_old.distance
        # proper motion vector
        mu0 = p0 * self.pmra[sInds] + q0 * self.pmdec[sInds]
        # space velocity vector
        v = mu0 / self.parx[sInds] * u.AU + r0 * self.rv[sInds]
        # set J2000 epoch
        j2000 = Time(2000.0, format="jyear")

        # if only 1 time in currentTime
        if nTimes == 1 or nStars == 1 or nTimes == nStars:
            # target star positions vector in heliocentric equatorial frame
            dr = v * (currentTime.mjd - j2000.mjd) * u.day
            r_targ = (coord_old.cartesian.xyz + dr).T.to("pc")

            if eclip:
                # transform to heliocentric true ecliptic frame
                coord_new = SkyCoord(
                    r_targ[:, 0], r_targ[:, 1], r_targ[:, 2], representation="cartesian"
                )
                r_targ = coord_new.heliocentrictrueecliptic.cartesian.xyz.T.to("pc")
            return r_targ

        # create multi-dimensional array for r_targ
        else:
            # target star positions vector in heliocentric equatorial frame
            r_targ = np.zeros([nTimes, nStars, 3]) * u.pc
            for i, m in enumerate(currentTime):
                dr = v * (m.mjd - j2000.mjd) * u.day
                r_targ[i, :, :] = (coord_old.cartesian.xyz + dr).T.to("pc")

            if eclip:
                # transform to heliocentric true ecliptic frame
                coord_new = SkyCoord(
                    r_targ[i, :, 0],
                    r_targ[i, :, 1],
                    r_targ[i, :, 2],
                    representation="cartesian",
                )
                r_targ[i, :, :] = coord_new.heliocentrictrueecliptic.cartesian.xyz.T.to(
                    "pc"
                )
            return r_targ
