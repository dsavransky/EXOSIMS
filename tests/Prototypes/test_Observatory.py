#!/usr/local/bin/python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>

r"""Observatory module unit tests

Michael Turmon, JPL, Feb. 2016
"""

import unittest
from EXOSIMS.Prototypes.Observatory import Observatory
from EXOSIMS.util.get_dirs import get_downloads_dir
import os
import urllib
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from jplephem.spk import SPK


class TestObservatoryMethods(unittest.TestCase):
    r"""Test Observatory class."""

    def setUp(self):
        # print '[setup] ',
        self.fixture = Observatory(forceStaticEphem=True)

    def tearDown(self):
        del self.fixture

    def test_keepout(self):
        r"""Test keepout method.

        Approach: Ensures the output is set, and is the correct size.
        The method implementation in the Prototype class is mostly a stub,
        so no real check is applicable.
        """

        print("keepout()")

        class MockStarCatalog(object):
            r"""Micro-catalog containing stars for testing.

            The class only supplies the attributes needed by the routine under test.
            """

            # Current contents:
            #   [0]: Arcturus
            #   [1]: Dummy

            Name = ["Arcturus", "Dummy_Star"]

            coords = SkyCoord(
                ra=np.array([213.91530029, 0.0]) * u.deg,
                dec=np.array([19.18240916, 0.0]) * u.deg,
                distance=np.array([11.26, 10]) * u.pc,
            )
            pmra = np.array([-1093.39, 0.0])  # mas/yr
            pmdec = np.array([-2000.06, 0.0])  # mas/yr
            rv = np.array([-5.19, 0.0])  # km/s
            parx = np.array([88.83, 1.0])  # mas
            nStars = 2
            staticStars = False

            def starprop(x, y, z):
                return (
                    np.array([[-8.82544227, -5.93387264, 3.69977354], [10, 0, 0]])
                    * u.pc
                )

        star_catalog = MockStarCatalog()
        t_ref = Time(2000.5, format="jyear")
        obs = self.fixture
        # generating koangle array with 1 subsystem
        nSystems = 1
        koAngles = np.zeros([nSystems, 4, 2])
        koAngles[0, :, 0] = 0
        koAngles[0, :, 1] = 180
        kogood = obs.keepout(
            star_catalog, np.arange(star_catalog.nStars), t_ref, koAngles
        )
        # return value should be True
        self.assertTrue(np.all(kogood))
        self.assertEqual(kogood.shape[0], nSystems)
        self.assertEqual(kogood.shape[1], star_catalog.nStars)

    def test_cent(self):
        r"""Test cent method.

        Approach: Probes for a range of inputs.
        """

        print("cent()")
        obs = self.fixture
        # origin at 12:00 on 2000.Jan.01
        t_ref_string = "2000-01-01T12:00:00.0"
        t_ref = Time(t_ref_string, format="isot", scale="utc")
        self.assertEqual(obs.cent(t_ref), 0.0)

        # even-julian-year probes
        t_probe = np.linspace(1950.0, 2050.0, 101)
        for t_ref in t_probe:
            # get the Time object, required by the cent() method
            t_probe_2 = Time(t_ref, format="jyear")
            # exosims century (offset from 2000)
            t_exo = obs.cent(t_probe_2)
            # reference century
            t_ref_cent = (t_ref - 2000.0) / 100.0
            # they are not exactly equal
            self.assertAlmostEqual(t_exo, t_ref_cent, places=10)

    def test_moon_earth(self):
        r"""Test moon_earth method.

        Approach: Reference to pre-computed result from Matlab.
        """

        print("moon_earth()")
        obs = self.fixture
        # TODO: add other times besides this one
        # century = 0
        t_ref_string = "2000-01-01T12:00:00.0"
        t_ref = Time(t_ref_string, format="isot", scale="utc")
        moon = obs.moon_earth(t_ref).flatten().to(u.km)
        # print moon
        r_earth = 6378.137  # earth radius [km], as used in Vallado's code
        moon_ref = [
            -45.74169421,
            -41.80825511,
            -11.88954996,
        ]  # pre-computed from Matlab
        for coord in range(3):
            self.assertAlmostEqual(
                moon[coord].value, moon_ref[coord] * r_earth, places=1
            )

    def test_keplerplanet(self):
        r"""Test keplerplanet method.

        Approach: Reference to result computed by external ephemeris.
        """

        print("keplerplanet()")
        obs = self.fixture
        # JPL ephemeris spice kernel data
        #   this is from: http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
        downloadsdir = get_downloads_dir()
        spkpath = os.path.join(downloadsdir, "de432s.bsp")
        if not os.path.exists(spkpath) and os.access(downloadsdir, os.W_OK | os.X_OK):
            spk_on_web = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp"
            try:
                urllib.urlretrieve(spk_on_web, spkpath)
            except:
                # Note: the SPK.open() below will fail in this case
                print("de432s.bsp missing in {}".format(spkpath))

        kernel = SPK.open(spkpath)  # smaller file, covers mission time range

        # t_ref and julian_day need to be consistent
        t_ref_string = "2000-01-01T12:00:00.0"
        t_ref = Time(t_ref_string, format="isot", scale="utc")
        julian_day = 2451545.0

        au_in_km = 149597870.7  # 1 AU in km

        # each planet to test: (name, JPL-ephem-number)
        bodies = [
            ("Mercury", 1),
            ("Venus", 2),
            ("Earth", 3),
            ("Mars", 4),
            ("Jupiter", 5),
            ("Saturn", 6),
            ("Uranus", 7),
            ("Neptune", 8),
            ("Pluto", 9),
        ]

        for body in bodies:
            (b_name, b_index) = body
            # 1: get EXOSIMS location
            #   get the planet ephemeris object
            # planet = obs.planets[b_name]
            pos_exo = obs.keplerplanet(t_ref, b_name).flatten()
            # 2: get JPL ephem location
            #   index of "i,j" refers to ephemeris body ID numbers in above list,
            #   0 for solar-system barycenter, 4 for Mars, etc.
            #   We do not account for:
            #     (1) Earth barycenter != Earth-moon-system barycenter
            #     (2) Sun barycenter != solar-system barycenter
            pos_ref = kernel[0, b_index].compute(julian_day)

            # convert to AU
            pos_ref_au = [_ / au_in_km for _ in pos_ref]
            pos_exo_au = pos_exo.to(u.au)

            # string versions
            pos_exo_str = ", ".join(["%.6f" % _.value for _ in pos_exo_au])
            pos_ref_str = ", ".join(["%.6f" % _ for _ in pos_ref_au])

            coord_names = ["x", "y", "z"]
            for coord in range(3):
                # common normalization in AU -- always at least 0.5
                norm = np.linalg.norm(pos_ref_au) + 0.5
                # delta in AU, relative to overall distance
                message = "%s, %s coord: %s vs. %s" % (
                    b_name,
                    coord_names[coord],
                    pos_exo_str,
                    pos_ref_str,
                )
                # FIXME: we should see more accuracy here
                self.assertAlmostEqual(
                    pos_exo_au[coord].value / norm,
                    pos_ref_au[coord] / norm,
                    msg=message,
                    delta=0.1,
                )
        kernel.close()

    def test_rot(self):
        r"""Test the rotation matrix generator.

        Method: (1) Ensure one axis is always kept fixed, as specified.
        (2) Randomized probes ensuring that R(axis,-theta) * R(axis,+theta) = I,
        but cumulated over many probes."""

        print("rot()")
        obs = self.fixture

        # rotation(0) = identity
        for axis in [1, 2, 3]:
            # theta = 0.0
            rotation = obs.rot(0.0, axis)
            # find || eye - rot1 ||
            diff = np.linalg.norm(np.eye(3) - rotation)
            self.assertAlmostEqual(diff, 0.0, delta=1e-12)
            # theta = 2*pi
            rotation = obs.rot(2.0 * np.pi, axis)
            # find || eye - rot1 ||
            diff = np.linalg.norm(np.eye(3) - rotation)
            self.assertAlmostEqual(diff, 0.0, delta=1e-12)

        # perform many randomized tests
        num_tests = 100
        num_products = 10
        for _test_counter in range(num_tests):
            thetas = []
            axes = []
            base = np.eye(3)
            # we will multiply a series of rotations into "base"
            rot_all = base
            for _rot_counter in range(num_products):
                theta = np.random.uniform(2 * np.pi)  # in [0,2 pi]
                axis = np.random.randint(3) + 1  # in {1,2,3}
                axes.append(axis)
                thetas.append(theta)
                rotation = obs.rot(theta, axis)
                # multiply rot1 into the cumulative rotation
                rot_all = np.dot(rot_all, rotation)
            # now, back all the rotations out
            for _rot_counter in range(num_products):
                theta = thetas.pop()
                axis = axes.pop()
                # apply the inverse rotation
                rotation = obs.rot(-theta, axis)
                rot_all = np.dot(rot_all, rotation)
            # find || base - rot1 * rot2 ||
            diff = np.linalg.norm(base - rot_all)
            self.assertAlmostEqual(diff, 0.0, delta=1e-10 * num_products)


if __name__ == "__main__":
    unittest.main()
