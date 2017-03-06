#!/usr/local/bin/python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>

r"""Observatory module unit tests

Michael Turmon, JPL, Feb. 2016
"""

import sys
import unittest
import StringIO
from collections import namedtuple
from EXOSIMS.Prototypes.Observatory import Observatory
from tests.TestSupport.Info import resource_path
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from jplephem.spk import SPK


class TestObservatoryMethods(unittest.TestCase):
    r"""Test Observatory class."""

    def setUp(self):
        # print '[setup] ',
        self.fixture = Observatory()

    def tearDown(self):
        del self.fixture

    def test_init(self):
        r"""Test of initialization and __init__.
        """
        obs = self.fixture
        self.assertEqual(obs._modtype, 'Observatory')
        self.assertEqual(type(obs._outspec), type({}))
        # check for presence of one class attribute
        self.assertGreater(obs.thrust.value, 0.0)

    def test_str(self):
        r"""Test __str__ method, for full coverage."""
        obs = self.fixture
        # replace stdout and keep a reference
        original_stdout = sys.stdout
        sys.stdout = StringIO.StringIO()
        # call __str__ method
        result = obs.__str__()
        # examine what was printed
        contents = sys.stdout.getvalue()
        self.assertEqual(type(contents), type(''))
        self.assertIn('thrust', contents)
        sys.stdout.close()
        # it also returns a string, which is not necessary
        self.assertEqual(type(result), type(''))
        # put stdout back
        sys.stdout = original_stdout

    def test_orbit(self):
        r"""Test orbit method.

        Approach: Ensures the output is set.  According to the documentation,
        "orbits are determined by specific instances of Observatory classes"
        so no quantitative check is applicable.
        """

        print 'orbit()'
        t_ref = Time(2000.0, format='jyear')
        obs = self.fixture
        r_sc = obs.orbit(t_ref)
        # the r_sc attribute is set and is a 3-tuple of astropy Quantity's
        self.assertEqual(type(r_sc), type(1.0 * u.km))
        self.assertEqual(r_sc.unit, u.km)
        self.assertEqual(r_sc.shape, (1,3))

    def test_keepout(self):
        r"""Test keepout method.

        Approach: Ensures the output is set, and is the correct size.
        The method implementation in the Prototype class is mostly a stub,
        so no real check is applicable.
        """

        print 'keepout()'
        class MockStarCatalog(object):
            r"""Micro-catalog containing stars for testing.

            The class only supplies the attributes needed by the routine under test.
            """

            # Current contents:
            #   [0]: Arcturus
            #   [1]: Dummy
            Name = ['Arcturus', 'Dummy_Star']
            coords = SkyCoord(ra=np.array([213.91530029, 0.0]),
                              dec=np.array([19.18240916, 0.0]),
                              unit='deg')
            pmra = np.array([-1093.39, 0.0]) # mas/yr
            pmdec = np.array([-2000.06, 0.0]) # mas/yr
            rv = np.array([-5.19, 0.0]) # km/s
            parx = np.array([88.83, 1.0]) # mas

        star_catalog = MockStarCatalog()
        t_ref = Time(2000.0, format='jyear')
        keepout_degrees = 4.0 # must be supplied, but currently unused
        obs = self.fixture
        # the routine under test
        success = obs.keepout(t_ref, star_catalog, keepout_degrees)
        # return value should be True
        self.assertTrue(success)
        # the r_sc attribute is set and is a 3-tuple of astropy Quantity's
        self.assertEqual(type(obs.kogood), type(np.array([True])))
        self.assertEqual(len(obs.kogood), len(star_catalog.Name))


    def test_cent(self):
        r"""Test cent method.

        Approach: Probes for a range of inputs.
        """

        print 'cent()'
        obs = self.fixture
        # origin at 12:00 on 2000.Jan.01
        t_ref_string = '2000-01-01T12:00:00.0'
        t_ref = Time(t_ref_string, format='isot', scale='utc')
        self.assertEqual(obs.cent(t_ref), 0.0)

        # even-julian-year probes
        t_probe = np.linspace(1950.0, 2050.0, 101)
        for t_ref in t_probe:
            # get the Time object, required by the cent() method
            t_probe_2 = Time(t_ref, format='jyear')
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

        print 'moon_earth()'
        obs = self.fixture
        # TODO: add other times besides this one
        # century = 0
        t_ref_string = '2000-01-01T12:00:00.0'
        t_ref = Time(t_ref_string, format='isot', scale='utc')
        moon = obs.moon_earth(t_ref)
        # print moon
        r_earth = 6378.137 # earth radius [km], as used in Vallado's code
        moon_ref = [-45.74169421, -41.80825511, -11.88954996] # pre-computed from Matlab
        for coord in range(3):
            self.assertAlmostEqual(moon[coord].value, moon_ref[coord] * r_earth, places=1)


    def test_keplerplanet(self):
        r"""Test keplerplanet method.

        Approach: Reference to result computed by external ephemeris.
        """

        print 'keplerplanet()'
        # to test this method, we need to have static ephemeris turned on
        if True:
            obs = Observatory(forceStaticEphem=True)
        else:
            obs = self.fixture # do not do it this way any longer
        # JPL ephemeris spice kernel data
        #   this is from: http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
        # kernel = SPK.open(resource_path() + '/de430.bsp') # large file, covers huge time range
        kernel = SPK.open(resource_path() + '/de432s.bsp') # smaller file, covers mission time range

        # t_ref and julian_day need to be consistent
        t_ref_string = '2000-01-01T12:00:00.0'
        t_ref = Time(t_ref_string, format='isot', scale='utc')
        julian_day = 2451545.0

        au_in_km = 149597870.7 # 1 AU in km

        # each planet to test: (name, JPL-ephem-number)
        bodies = [
            ('Mercury', 1),
            ('Venus', 2),
            ('Earth', 3),
            ('Mars', 4),
            ('Jupiter', 5),
            ('Saturn', 6),
            ('Uranus', 7),
            ('Neptune', 8),
            ('Pluto', 9),
        ]

        for body in bodies:
            (b_name, b_index) = body
            # 1: get EXOSIMS location
            #   get the planet ephemeris object
            # planet = obs.planets[b_name]
            pos_exo = obs.keplerplanet(t_ref, b_name)
            # 2: get JPL ephem location
            #   index of "i,j" refers to ephemeris body ID numbers in above list,
            #   0 for solar-system barycenter, 4 for Mars, etc.
            #   We do not account for:
            #     (1) Earth barycenter != Earth-moon-system barycenter
            #     (2) Sun barycenter != solar-system barycenter
            pos_ref = kernel[0, b_index].compute(julian_day)

            # convert to AU
            pos_ref_au = [_/au_in_km for _ in pos_ref]
            pos_exo_au = pos_exo.to(u.au)

            # string versions
            pos_exo_str = ', '.join(['%.6f' % _.value for _ in pos_exo_au])
            pos_ref_str = ', '.join(['%.6f' % _ for _ in pos_ref_au])

            coord_names = ['x', 'y', 'z']
            for coord in range(3):
                # common normalization in AU -- always at least 0.5
                norm = np.linalg.norm(pos_ref_au) + 0.5
                # delta in AU, relative to overall distance
                message = (
                    '%s, %s coord: %s vs. %s' %
                    (b_name, coord_names[coord], pos_exo_str, pos_ref_str))
                # FIXME: we should see more accuracy here
                self.assertAlmostEqual(pos_exo_au[coord].value/norm,
                                       pos_ref_au[coord]/norm,
                                       msg=message,
                                       delta=0.1)

    def test_rot(self):
        r"""Test the rotation matrix generator.

        Method: (1) Ensure one axis is always kept fixed, as specified.
        (2) Randomized probes ensuring that R(axis,-theta) * R(axis,+theta) = I,
        but cumulated over many probes."""

        print 'rot()'
        obs = self.fixture

        # rotation(0) = identity
        for axis in [1, 2, 3]:
            # theta = 0.0
            rotation = obs.rot(0.0, axis)
            # find || eye - rot1 ||
            diff = np.linalg.norm(np.eye(3) - rotation)
            self.assertAlmostEqual(diff, 0.0, delta=1e-12)
            # theta = 2*pi
            rotation = obs.rot(2.0*np.pi, axis)
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
                theta = np.random.uniform(2*np.pi) # in [0,2 pi]
                axis = np.random.randint(3) + 1 # in {1,2,3}
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
            self.assertAlmostEqual(diff, 0.0, delta=1e-10*num_products)

    def test_starprop(self):
        r"""Test stellar-location propagation."""

        # (Not sure if this is mocking a TargetList or a StarCatalog, actually)
        class MockStarCatalog(object):
            r"""Micro-catalog containing stars for testing.

            The class only supplies the attributes needed by the routine under test.
            """

            # Current contents:
            #   [0]: Arcturus, with values loaded from the EXOSIMS SIMBAD300 catalog,
            #   checked with the SIMBAD website
            #   (http://simbad.u-strasbg.fr/simbad/sim-id?Ident=NAME+ARCTURUS).
            #   It has a large proper motion.
            Name = ['Arcturus']
            coords = SkyCoord(ra=np.array([213.91530029]),
                              dec=np.array([19.18240916]),
                              unit='deg')
            # 10/2016:
            #  these were formerly dimensionless, and now there is a units issue
            pmra = np.array([-1093.39])*u.mas/u.yr # mas/yr
            pmdec = np.array([-2000.06])*u.mas/u.yr # mas/yr
            rv = np.array([-5.19])*u.km/u.s # km/s
            parx = np.array([88.83])*u.mas # mas

        print 'starprop()'
        obs = self.fixture

        # instantiate a catalog
        star_catalog = MockStarCatalog()
        sInd = np.array(range(len(star_catalog.rv)))

        # now, and now + 50k years
        t_now    = Time(2000.0, format='jyear')
        t_future = Time(2000.0+50000.0, format='jyear')

        # locations
        #loc_now = obs.starprop(t_now, star_catalog, 0)
        #loc_future = obs.starprop(t_future, star_catalog, 0)
        loc_now    = obs.starprop(star_catalog, sInd, t_now)
        loc_future = obs.starprop(star_catalog, sInd, t_future)

        # locations following calculations in http://www.astronexus.com/a-a/motions-long-term
        # current location
        self.assertAlmostEqual(np.linalg.norm(loc_now.to(u.pc).value - [-8.823, -5.932, 3.699]),
                               0.0, delta=0.01)
        # future location
        self.assertAlmostEqual(np.linalg.norm(loc_future.to(u.pc).value - [-11.77, -4.32, -1.55]),
                               0.0, delta=0.01)


    def test_distForces(self):
        r"""Test force-on-occulter calculation

        Unimplemented."""
        # FIXME: Have not taken the time to completely unpack the details of
        # this calculation.  Overview:
        # The distForces method uses:
        #   -- the s/c position (from obs.orbit()),
        #   -- the Earth position (from obs.keplerplanet())
        #   -- the target position (from obs.starprop() and a target list).
        # The target position is used to give the position of the occulter.
        # It seems to regard the s/c as two pieces (telescope and occulter), computes
        # the force on both (from Sun and Earth+Moon), and then decomposes the resulting
        # force into two components, lateral and axial, which are returned by this
        # method.
        # The lateral force component must be counteracted with other s/c thrust
        # (see mass_dec() method).
        # To test, one could either instantiate an Observatory, get the position, etc.,
        # from the methods above, and re-do the computation.  Or, one could fully over-ride
        # the methods above so that the positions were chosen in this test routine. Or, again,
        # one could simply check a necessary condition on the returned forces.  This last choice
        # seems too weak.
        obs = self.fixture
        class MockObservatory(object):
            def __init__(self, obs, pos_earth, pos_sc, pos_targets):
                r"""New mock observatory; first arg 'obs' is the object this mock replaces."""
                self.pos_earth = np.array(pos_earth) * u.au
                self.pos_sc = np.array(pos_sc) * u.au
                self.pos_targets = np.array(pos_targets) * u.au
                # obs.r_sc is the current Observatory position, which we need to
                # be able to access from within here
                self.obs = obs
                return
            def keplerplanet(self, time, dummy):
                r"""Earth position."""
                return self.pos_earth
            def orbit(self, time):
                r"""Spacecraft position."""
                # return value is set this way
                self.obs.r_sc = self.pos_sc
                return True
            def starprop(self, time, dummy, index):
                r"""Target position."""
                return self.pos_targets[index]

        # mock time keeper object -- it is not used, but it is dereferenced
        MockTimekeeper = namedtuple('MockTimekeeper', 'currentTimeAbs')
        # formerly used this:
        #   timekeeper = MockTimekeeper(currentTimeAbs=Time(2000.0, format='jyear').mjd)
        # but API changed, and now pass a Time object like so:
        timekeeper = MockTimekeeper(currentTimeAbs=Time(2000.0, format='jyear'))

        # create mock observatory with stubs for some methods
        # First test:
        #  puts Sun, Earth-Moon, Observatory, and target in a line so all force is axial
        obs_stubs = MockObservatory(obs,
                                    [1.00, 0.0, 0.0],
                                    [1.1,  0.0, 0.0],
                                    [
                                        [1e3, 0.0, 0.0]
                                    ])
        # insert stubs into observatory
        obs.orbit = obs_stubs.orbit
        obs.keplerplanet = obs_stubs.keplerplanet
        obs.starprop = obs_stubs.starprop
        # invoke method
        #    arg 1, time, is not actually used; arg 2, star catalog, is not used
        (dF_lateral, dF_axial) = obs.distForces(timekeeper, None, 0)
        #print 'lateral = %g axial = %g' % (dF_lateral.to(u.N).value, dF_axial.to(u.N).value) 
        self.assertAlmostEqual(dF_lateral.to(u.N).value, 0.0, delta=1e-6)
        self.assertGreater(dF_axial.to(u.N).value, 0.0)

        ## Second test: move observatory farther out
        obs_stubs = MockObservatory(obs,
                                    [1.00, 0.0, 0.0],
                                    [1.2,  0.0, 0.0],
                                    [
                                        [1e3, 0.0, 0.0]
                                    ])
        # insert stubs into observatory
        obs.orbit = obs_stubs.orbit
        obs.keplerplanet = obs_stubs.keplerplanet
        obs.starprop = obs_stubs.starprop
        # invoke method
        #    arg 1, time, is not actually used; arg 2, star catalog, is not used
        (dF_lateral_far, dF_axial_far) = obs.distForces(timekeeper, None, 0)
        self.assertAlmostEqual(dF_lateral_far.to(u.N).value, 0.0, delta=1e-6)
        # ensure axial force decreases
        self.assertGreater(dF_axial.to(u.N).value, dF_axial_far.to(u.N).value)
        return


    def untested_mass_dec(self):
        r"""Test mass for station-keeping calculation.

        Unimplemented."""
        # FIXME: Unable to determine the principle used for this calculation.
        # The specific stumbling block is the angles [cos(45 deg.) and cos(5 deg.)]
        # used to relate m-dot (mass change per time) to force needed (dF_lateral)
        # and specific impulse available (sk Isp)
        pass


if __name__ == '__main__':
    unittest.main()
