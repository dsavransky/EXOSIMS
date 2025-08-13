import os
import astropy.units as u
import numpy as np
from scipy.interpolate import CubicSpline
from astropy.table import Table
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from astropy.coordinates import HCRS
from angutils.angutils import projplane, calcang, rotMat

from EXOSIMS.Prototypes.Observatory import Observatory
from EXOSIMS.util.get_dirs import get_downloads_dir
from EXOSIMS.util._numpy_compat import copy_if_needed


class ObservatoryTESS(Observatory):
    """Roman Space Telescope Observatory. Data is retrieved from JPL Horizons."""

    def __init__(self, **specs):

        # run prototype __init__
        Observatory.__init__(self, **specs)

        # load orbit data (download as needed)
        downloadsdir = get_downloads_dir()
        eclip_file = os.path.join(downloadsdir, "roman_orbit_ecliptic.ecsv")
        equat_file = os.path.join(downloadsdir, "roman_orbit_equatorial.ecsv")

        if not (os.path.exists(eclip_file)) or not (os.path.exists(equat_file)):
            print("Roman orbit data not found. Downloading.")
            obj = Horizons(
                id="-95",
                location="0",
                epochs={"start": "2022-01-01", "stop": "2023-01-01", "step": "1d"},
            )
            roman_ecliptic_vectors_table = obj.vectors()
            roman_ecliptic_vectors_table.write(eclip_file)

            roman_equatorial_vectors_table = obj.vectors(refplane="earth")
            roman_equatorial_vectors_table.write(equat_file)

        roman_ecliptic_vectors_table = Table.read(eclip_file)
        roman_equatorial_vectors_table = Table.read(equat_file)

        # set up interpolants
        self.roman_ecliptic_interpolant = self.genRomanOrbitInterpolant(
            roman_ecliptic_vectors_table
        )
        self.roman_equatorial_interpolant = self.genRomanOrbitInterpolant(
            roman_equatorial_vectors_table
        )

        # save original tables as attributes
        self.roman_ecliptic_vectors_table = roman_ecliptic_vectors_table
        self.roman_equatorial_vectors_table = roman_equatorial_vectors_table

        # save orbit start and end time
        self.t0 = Time(roman_ecliptic_vectors_table["datetime_jd"].data[0], format="jd")
        self.tf = Time(
            roman_ecliptic_vectors_table["datetime_jd"].data[-1], format="jd"
        )

        # define inertial basis vectors
        self.e1 = np.array([1, 0, 0])
        self.e2 = np.array([0, 1, 0])
        self.e3 = np.array([0, 0, 1])

    def genRomanOrbitInterpolant(self, vectors_table):
        """

        Args:
            vectors_table (astropy.table.Table):
                Table of orbit data queried from Horizons

        Returns:
            scipy.interpolate.CubicSpline:
                Cubic spline interpolant over orbit data

        """

        # get time array and convert to MJD
        ts = Time(vectors_table["datetime_jd"].data, format="jd")

        # get position data
        pos = np.vstack([vectors_table[l].data for l in ["x", "y", "z"]]).T

        # ensure that this is in AU
        assert (
            vectors_table["x"].info.unit
            == vectors_table["y"].info.unit
            == vectors_table["z"].info.unit
        ), "Orbit table columns have different units."
        if vectors_table["x"].info.unit is not u.AU:
            pos = (pos * vectors_table["x"].info.unit).to_value(u.AU)

        return CubicSpline(ts.mjd, pos)

    def orbit(self, currentTime, eclip=False):
        """Finds observatory orbit positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).

        This method returns the telescope geosynchronous circular orbit position vector.

        Args:
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD
            eclip (bool):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to
                False, corresponding to heliocentric equatorial frame.

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Observatory orbit positions vector in heliocentric equatorial (default)
                or ecliptic frame in units of AU. nx3

        Note:
            Use eclip=True to get ecliptic coordinates.

        """

        mjdtime = np.array(currentTime.mjd, ndmin=1, copy=copy_if_needed)
        assert (
            currentTime.min() >= self.t0
        ), f"No orbit data is available for dates prior to {self.t0.to_datetime()}"
        assert (
            currentTime.max() <= self.tf
        ), f"No orbit data is available for dates prior to {self.tf.to_datetime()}"

        if eclip:
            return self.roman_ecliptic_interpolant(mjdtime) << u.AU
        else:
            return self.roman_equatorial_interpolant(mjdtime) << u.AU

    def calc_pitch(self, target_coords, ts):
        """

        Args:
            target_coords (astropy.coordinates.SkyCoord):
                Target coordinates
            ts (astropy.time.Time):
                Observation time(s) - can be an array of times

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Observatory pitch angles

        """

        # sun position and unit vector wrt observatory
        r_obs_sun = self.orbit(ts).T
        r_sun_obs = -r_obs_sun
        rhat_sun_obs = (r_sun_obs / np.linalg.norm(r_sun_obs, axis=0)).value

        # update target position and compute position and unit vector wrt observatory
        r_target_sun = (
            target_coords.apply_space_motion(new_obstime=ts)
            .transform_to(HCRS)
            .cartesian.xyz
        )
        r_target_obs = r_target_sun - r_obs_sun
        rhat_target_obs = (r_target_obs / np.linalg.norm(r_target_obs, axis=0)).value

        # align b_3 with -r_obs/sun (equivalently r_sun/obs) in 2 steps:
        # 1. rotate about b_2 by the angle between b_3 and the projection
        #    of the sun/obs vector onto the e1/e3 plane

        # projection of sun/obs vector onto e1/e3 plane:
        r_sun_obs_proj1 = projplane(r_sun_obs, self.e2)
        rhat_sun_obs_proj1 = (
            r_sun_obs_proj1 / np.linalg.norm(r_sun_obs_proj1, axis=0)
        ).value
        ang1 = np.array([calcang(x, self.e3, self.e2) for x in rhat_sun_obs_proj1.T])

        # DCM between inertial and body frames
        B_C_I = np.dstack([rotMat(2, -a) for a in ang1])

        # 2. rotate about b_1 by the angle between the new b_3 and r_sun/obs
        b_3 = B_C_I[2, :, :].T
        b_1 = B_C_I[0, :, :].T
        ang2 = np.array(
            [calcang(x, b3, b1) for x, b3, b1 in zip(rhat_sun_obs.T, b_3, b_1)]
        )

        # update DCM
        B_C_I = np.dstack(
            [np.matmul(rotMat(1, -a), B_C_I[:, :, j]) for j, a in enumerate(ang2)]
        )

        # now we wish to align b_1 to r_star/obs with yaw, pitch (b_3, b_2)
        # projection of star/obs vector onto b1/e2 plane:
        r_target_obs_proj1 = np.hstack(
            [
                projplane(np.array(r_target_obs[:, j], ndmin=2).T, B_C_I[2, :, j].T)
                for j in range(len(ts))
            ]
        )
        rhat_target_obs_proj1 = r_target_obs_proj1 / np.linalg.norm(
            r_target_obs_proj1, axis=0
        )

        # yaw is angle between projection and b_1
        b_1 = B_C_I[0, :, :].T
        b_3 = B_C_I[2, :, :].T
        yaw = -np.array(
            [calcang(x, b1, b3) for x, b1, b3 in zip(rhat_target_obs_proj1.T, b_1, b_3)]
        )

        # update DCM
        B_C_I = np.dstack(
            [np.matmul(rotMat(3, a), B_C_I[:, :, j]) for j, a in enumerate(yaw)]
        )

        # next we pitch! rotate about b_2 by the angle between b_1 and final look vector
        b_1 = B_C_I[0, :, :].T
        b_2 = B_C_I[1, :, :].T
        pitch = -np.array(
            [calcang(x, b1, b2) for x, b1, b2 in zip(rhat_target_obs.T, b_1, b_2)]
        )

        return pitch * u.rad