#!/usr/bin/env python
#
# This class is designed to be imported and used as such a mixin.
# To exercise the usage example contained here, the following works:
#   % python <this_file.py>
# You may need to put EXOSIMS in your $PYTHONPATH, e.g.,
#   % PYTHONPATH=/path/to/exomissionsim <this_file.py>

r"""PlanetPhysicalModel module unit tests

This contains a PlanetPhysicalModel mixin class that generically tests these methods:
  __init__
  __str__
  calc_albedo_from_sma
  calc_radius_from_mass
  calc_mass_from_radius

Usage:
  See the end of the file for an example.  Loosely, you define your test class
based on unittest.TestCase, with the class here as a mixin.  If you want to add 
more tests to a class derived this way, add them as additional test_* methods 
within your derived test class.

Michael Turmon, JPL, Apr. 2016
"""

import sys
import math
import unittest
import StringIO
import numpy as np
import astropy.units as u
import astropy.constants as const

# use the prototype for the usage example
from EXOSIMS.Prototypes.PlanetPhysicalModel import PlanetPhysicalModel


class TestPlanetPhysicalModelMixin(object):
    r"""Generic methods to test the PlanetPhysicalModel class, usable as a mixin."""

    #
    # define some class attributes that classes may over-ride
    #

    # the main one: the class under test
    # *You must over-ride this attribute in your derived class*
    planet_model = None

    # arbitrary list of planet counts
    planet_numbers = [0, 1, 10, 1001, 42]
    # semi-major-axis range endpoints (log10 scale, multiple of AU)
    semi_major_axis_endpoints = (-1.0, 2.0)
    # planetary mass range endpoints (log10 scale, multiple of M_earth)
    planetary_mass_endpoints = (-1.0, 2.0)
    # planetary radius range endpoints (log10 scale, multiple of R_earth)
    planetary_radius_endpoints = (-1.0, 2.0)
    #
    roundtrip_radius_generator = None
    # allowed relative error in roundtrip radius->mass->radius computation
    delta_roundtrip = 1.0e-7
    
    def setUp(self):
        # print '[setup] ',
        assert self.planet_model, "You must over-ride self.planet_model in derived class"
        self.fixture = self.planet_model

    def tearDown(self):
        # del self.fixture
        pass

    def test_init(self):
        r"""Test of initialization and __init__.
        """
        ppmodel = self.fixture()
        self.assertEqual(ppmodel._modtype, 'PlanetPhysicalModel')
        self.assertEqual(type(ppmodel._outspec), type({}))

    def test_str(self):
        r"""Test __str__ method, for full coverage."""
        ppmodel = self.fixture()
        # replace stdout and keep a reference
        original_stdout = sys.stdout
        sys.stdout = StringIO.StringIO()
        # call __str__ method
        result = ppmodel.__str__()
        # examine what was printed
        contents = sys.stdout.getvalue()
        self.assertEqual(type(contents), type(''))
        sys.stdout.close()
        # it also returns a string, which is not necessary
        self.assertEqual(type(result), type(''))
        # put stdout back
        sys.stdout = original_stdout


    def test_calc_Phi(self):
        r"""Test calc_Phi method for computation of exoplanet phase function.
        
        Approach: Check the computation of Phi at sample points based on
        the Lambert scattering function from, e.g., Traub et al., JATIS 2(1), 
        2016, sec. 2.3."""
        print 'calc_Phi()'

        ppmodel = self.fixture()
        r = np.linspace(0.0, 2*math.pi, 100) * u.rad

        result = ppmodel.calc_Phi(r)

        # phase function (radians)
        phi = lambda(x): (math.sin(x) + (math.pi - x) * math.cos(x))/math.pi
        expected = [phi(x.value) for x in r]

        np.testing.assert_allclose(result, expected, rtol=1e-07, atol=0)
        

    def test_calc_albedo_from_sma(self):
        r"""Test calc_albedo_from_sma method.

        Approach: Ensure albedo is present, in correct size, with correct range.
        The prototype function is rather unconstrained.
        """

        print 'albedo()'
        for n_planet in self.planet_numbers:
            # specific values are presently unused, but make them reasonable
            semi_major_axis = np.logspace(self.semi_major_axis_endpoints[0],
                                          self.semi_major_axis_endpoints[1],
                                          num=n_planet) * u.AU

            ppmodel = self.fixture()
            albedo = ppmodel.calc_albedo_from_sma(semi_major_axis)

            # check that the albedo is legal type
            self.assertIsInstance(albedo, np.ndarray)
            self.assertEqual(albedo.size, n_planet)
            # and in legal range
            self.assertTrue(np.all(albedo >= 0.0))
            self.assertTrue(np.all(albedo <= 1.0))
            self.assertTrue(np.all(np.isfinite(albedo)))
            # (albedo is dimensionless)
            # get rid of the temporary object
            del ppmodel

    def test_calc_radius_from_mass(self):
        r"""Test calc_radius_from_mass method.

        Approach: Ensure radius is present, in correct size, with correct range.
        The prototype function is rather unconstrained.
        """

        print 'radius_from_mass()'
        for n_planet in self.planet_numbers:
            # a selection of reasonable planetary masses
            masses = np.logspace(self.planetary_mass_endpoints[0],
                                 self.planetary_mass_endpoints[1],
                                 num=n_planet) * const.M_earth

            ppmodel = self.fixture()
            radii = ppmodel.calc_radius_from_mass(masses)

            # check that the radii are legal type
            self.assertIsInstance(radii, np.ndarray)
            self.assertEqual(radii.size, n_planet)
            self.assertIsInstance(masses, u.quantity.Quantity)
            # and in legal range
            self.assertTrue(np.all(radii > 0.0))
            self.assertTrue(np.all(np.isfinite(radii)))
            # with correct units: ensure it has units of length
            self.assertTrue(np.all(radii + 1*u.km > 0.0))
            # get rid of the temporary object
            del ppmodel


    def test_calc_mass_from_radius(self):
        r"""Test calc_mass_from_radius method.

        Approach: Ensure mass is present, in correct size, with correct range.
        The prototype function is rather unconstrained.
        """

        print 'mass_from_radius()'
        for n_planet in self.planet_numbers:
            # a selection of reasonable planetary radii
            radii = np.logspace(self.planetary_radius_endpoints[0],
                                self.planetary_radius_endpoints[1],
                                num=n_planet) * const.R_earth

            ppmodel = self.fixture()
            masses = ppmodel.calc_mass_from_radius(radii)

            # check that the masses are legal
            self.assertIsInstance(masses, np.ndarray)
            self.assertEqual(masses.size, n_planet)
            self.assertIsInstance(masses, u.quantity.Quantity)
            # and in legal range
            # print 'mass-from-radius', masses
            self.assertTrue(np.all(masses > 0.0 * u.kg))
            self.assertTrue(np.all(np.isfinite(masses)))
            # with correct units: ensure it has units of mass
            self.assertTrue(np.all((masses + 1*u.kg) > 0.0))
            # get rid of the temporary object
            del ppmodel


    def test_radius_mass_radius(self):
        r"""Test roundtrip of calc_mass_from_radius -> calc_radius_from mass.

        Approach: Pick a radius, find a mass, recover radius from the mass,
        ensure they are close to equal.
        """

        print 'mass/radius roundtrip'
        max_difference = 0.0
        for n_planet in self.planet_numbers:

            if self.roundtrip_radius_generator is None:
                # use the standard selection of reasonable planetary radii
                radii_1 = np.logspace(self.planetary_radius_endpoints[0],
                                    self.planetary_radius_endpoints[1],
                                    num=n_planet) * const.R_earth
            else:
                radii_1 = self.roundtrip_radius_generator(n_planet) * const.R_earth

            # empty list is tested elsewhere, and messes up numerical test below
            if len(radii_1) == 0: continue

            ppmodel = self.fixture()
            masses = ppmodel.calc_mass_from_radius(radii_1)
            radii_2 = ppmodel.calc_radius_from_mass(masses)

            # basic legality: the other tests do this properly
            self.assertIsInstance(masses, np.ndarray)
            self.assertIsInstance(radii_2, np.ndarray)
            self.assertEqual(masses.size, radii_2.size)
            # ensure no NaNs
            self.assertTrue(np.all(np.isfinite(masses)))
            self.assertTrue(np.all(np.isfinite(radii_2)))

            # radius disparity (relative)
            #   The error will vary across the radius, so need to divide elementwise
            difference = np.max(np.abs((radii_1 - radii_2) / radii_1).value)
            # keep track of the max, for fun
            if difference > max_difference: max_difference = difference
            self.assertAlmostEqual(difference, 0.0, delta=self.delta_roundtrip)

            if False and n_planet < 20:
                print 'R1', radii_1/const.R_earth
                print 'R2', radii_2/const.R_earth
                print 'M', masses/const.M_earth
                print 'D', ((radii_1 - radii_2) / radii_1).value
                crap = ((radii_1 - radii_2) / radii_1).value
                print 'Dtop', crap[np.abs(crap) > 1]
                print 'Dmax', difference

            # get rid of the temporary object
            del ppmodel

        print ('[Info] PlanetPhysicalModel radius roundtrip: max_diff = %.3g; allowed = %g' %
               (max_difference, self.delta_roundtrip))
        

# This is a usage example to keep the present file
# self-contained.  See nearby for other examples.
class TestPlanetPhysicalModelExample(TestPlanetPhysicalModelMixin,
                                     unittest.TestCase):
    r"""Usage example: Prototype PlanetPhysicalModel class."""
    # MUST over-ride to set up the specific test fixture (class constructor) we will use
    planet_model = PlanetPhysicalModel
    # Can over-ride class attributes if desired
    planet_numbers = [0, 1, 10]


def main():
    unittest.main()

if __name__ == '__main__':
    main()
