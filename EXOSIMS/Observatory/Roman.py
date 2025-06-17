import os
import astropy.units as u
import numpy as np
from scipy.interpolate import CubicSpline
from astropy.table import Table
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from EXOSIMS.Prototypes.Observatory import Observatory
from EXOSIMS.util.get_dirs import get_downloads_dir
from EXOSIMS.util._numpy_compat import copy_if_needed


class Roman(Observatory):
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
                id="-211",
                location="@0",
                epochs={"start": "2026-10-31", "stop": "2032-01-28", "step": "1d"},
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
