#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>
# You may need to put EXOSIMS in your $PYTHONPATH, e.g.,
#   % PYTHONPATH=/path/to/exomissionsim <this_file.py>

r"""FortneyMarleyCahoyMix1 specialization of PlanetPhysicalModel unit tests

Michael Turmon, JPL, Apr. 2016
"""

import unittest
import numpy as np
import astropy.units as u
import astropy.constants as const
import scipy.interpolate as interpolate


# Generic test suite for PlanetPhysicalModel
from TestPlanetPhysicalModel import TestPlanetPhysicalModelMixin

# the specific PlanetPhysicalModel class we test here
from EXOSIMS.PlanetPhysicalModel.FortneyMarleyCahoyMix1 import FortneyMarleyCahoyMix1


####################################################################################
# Taken from Table 4 of Cahoy et al., 2010:
#   Cahoy, K. L., Marley, M.S., & Fortney, J.J. 2010, ApJ, 724.1, 189
# Caption:
# Geometric albedos (Ag) of the exoplanet model Jupiter and Neptunes
# in this work, for each of the B, V, R, and I filters shown in Fig. 14
# and the albedo spectra model described in Sec. 3.2.5 and Appendix A.
# Columns are:
#   distance[AU] metallicity Ag(B) Ag(V) Ag(R)
# We only use the Ag(V) albedo figure. Ag is geometric albedo.

albedo_tabulation = [
    ( 0.8,  1.0, 0.556, 0.322, 0.156),
    ( 0.8,  3.0, 0.482, 0.241, 0.102),
    ( 0.8, 10.0, 0.455, 0.209, 0.074),
    ( 0.8, 30.0, 0.359, 0.142, 0.045),
    ( 2.0,  1.0, 0.733, 0.742, 0.727),
    ( 2.0,  3.0, 0.758, 0.766, 0.726),
    ( 2.0, 10.0, 0.744, 0.728, 0.616),
    ( 2.0, 30.0, 0.735, 0.674, 0.498),
    ( 5.0,  1.0, 0.609, 0.567, 0.531),
    ( 5.0,  3.0, 0.567, 0.506, 0.458),
    ( 5.0, 10.0, 0.434, 0.326, 0.238),
    ( 5.0, 30.0, 0.430, 0.303, 0.191),
    (10.0,  1.0, 0.450, 0.386, 0.358),
    (10.0,  3.0, 0.316, 0.260, 0.246),
    (10.0, 10.0, 0.388, 0.295, 0.228),
    (10.0, 30.0, 0.388, 0.279, 0.189),
    ]

####################################################################################
# Mass-Radius relationships pasted from the erratum:
# The Astrophysical Journal, 668:1267, 2007, October 20 2007. 
# J. J. Fortney, M. S. Marley, and J. W. Barnes
# Erratum: "Planetary radii across five orders of magnitude in
# mass and stellar insolation: application to transits"
# (Original article in ApJ, 659, 1661 [2007])

def mass2radius_ice_rock(imf, logM):
    r"""Equation (7) of Fortney et al., as corrected."""
    assert imf >= 0.0 and imf <= 1.0, "Fraction in [0,1]"
    radius = (
        (0.0912 * imf + 0.1603) * logM*logM +
        (0.3330 * imf + 0.7387) * logM +
        (0.4639 * imf + 1.1193)
        )
    return radius

def mass2radius_rock_iron(rmf, logM):
    r"""Equation (8) of Fortney et al., as corrected."""
    assert rmf >= 0.0 and rmf <= 1.0, "Fraction in [0,1]"
    radius = (
        (0.0592 * rmf + 0.0975) * logM*logM +
        (0.2337 * rmf + 0.4938) * logM +
        (0.3102 * rmf + 0.7932)
        )
    return radius

####################################################################################
# Giant Planet Radii at 4.5 Gyr, processed from Table 4 of
# J. J. Fortney, M. S. Marley, and J. W. Barnes,
# The Astrophysical Journal, 659, pg. 1670, 2007.
# Tabular contents pasted into text-file from Acrobat Pro, then reformatted
# by short awk script.
# Each quad-tuple is one entry of the table, in units as stated:
#   (distance[AU], CoreMass[M_earth], PlanetMass[M_earth], Radius[R_jup])
# NaNs exist where CoreMass > PlanetMass, which is impossible.  The first
# NaN is given as "..." in the printed table, probably because unrealistic.
giant_planet_radii = [
    ( 0.02,   0,  17,np.NaN), ( 0.02,   0,  28, 1.355), ( 0.02,   0,  46, 1.252),
    ( 0.02,   0,  77, 1.183), ( 0.02,   0, 129, 1.190), ( 0.02,   0, 215, 1.189),
    ( 0.02,   0, 318, 1.179), ( 0.02,   0, 464, 1.174), ( 0.02,   0, 774, 1.170),
    ( 0.02,   0,1292, 1.178), ( 0.02,   0,2154, 1.164), ( 0.02,   0,3594, 1.118),
    ( 0.02,  10,  17, 0.726), ( 0.02,  10,  28, 0.934), ( 0.02,  10,  46, 1.019),
    ( 0.02,  10,  77, 1.072), ( 0.02,  10, 129, 1.123), ( 0.02,  10, 215, 1.148),
    ( 0.02,  10, 318, 1.153), ( 0.02,  10, 464, 1.157), ( 0.02,  10, 774, 1.160),
    ( 0.02,  10,1292, 1.172), ( 0.02,  10,2154, 1.160), ( 0.02,  10,3594, 1.116),
    ( 0.02,  25,  17,np.NaN), ( 0.02,  25,  28, 0.430), ( 0.02,  25,  46, 0.756),
    ( 0.02,  25,  77, 0.928), ( 0.02,  25, 129, 1.032), ( 0.02,  25, 215, 1.091),
    ( 0.02,  25, 318, 1.116), ( 0.02,  25, 464, 1.131), ( 0.02,  25, 774, 1.145),
    ( 0.02,  25,1292, 1.163), ( 0.02,  25,2154, 1.155), ( 0.02,  25,3594, 1.112),
    ( 0.02,  50,  17,np.NaN), ( 0.02,  50,  28,np.NaN), ( 0.02,  50,  46,np.NaN),
    ( 0.02,  50,  77, 0.695), ( 0.02,  50, 129, 0.891), ( 0.02,  50, 215, 1.004),
    ( 0.02,  50, 318, 1.056), ( 0.02,  50, 464, 1.091), ( 0.02,  50, 774, 1.121),
    ( 0.02,  50,1292, 1.148), ( 0.02,  50,2154, 1.144), ( 0.02,  50,3594, 1.106),
    ( 0.02, 100,  17,np.NaN), ( 0.02, 100,  28,np.NaN), ( 0.02, 100,  46,np.NaN),
    ( 0.02, 100,  77,np.NaN), ( 0.02, 100, 129, 0.613), ( 0.02, 100, 215, 0.841),
    ( 0.02, 100, 318, 0.944), ( 0.02, 100, 464, 1.011), ( 0.02, 100, 774, 1.076),
    ( 0.02, 100,1292, 1.118), ( 0.02, 100,2154, 1.125), ( 0.02, 100,3594, 1.095),
    (0.045,   0,  17, 1.103), (0.045,   0,  28, 1.065), (0.045,   0,  46, 1.038),
    (0.045,   0,  77, 1.049), (0.045,   0, 129, 1.086), (0.045,   0, 215, 1.105),
    (0.045,   0, 318, 1.107), (0.045,   0, 464, 1.108), (0.045,   0, 774, 1.113),
    (0.045,   0,1292, 1.118), (0.045,   0,2154, 1.099), (0.045,   0,3594, 1.053),
    (0.045,  10,  17, 0.599), (0.045,  10,  28, 0.775), (0.045,  10,  46, 0.878),
    (0.045,  10,  77, 0.964), (0.045,  10, 129, 1.029), (0.045,  10, 215, 1.069),
    (0.045,  10, 318, 1.083), (0.045,  10, 464, 1.092), (0.045,  10, 774, 1.104),
    (0.045,  10,1292, 1.112), (0.045,  10,2154, 1.095), (0.045,  10,3594, 1.050),
    (0.045,  25,  17,np.NaN), (0.045,  25,  28, 0.403), (0.045,  25,  46, 0.686),
    (0.045,  25,  77, 0.846), (0.045,  25, 129, 0.952), (0.045,  25, 215, 1.019),
    (0.045,  25, 318, 1.050), (0.045,  25, 464, 1.069), (0.045,  25, 774, 1.090),
    (0.045,  25,1292, 1.104), (0.045,  25,2154, 1.090), (0.045,  25,3594, 1.047),
    (0.045,  50,  17,np.NaN), (0.045,  50,  28,np.NaN), (0.045,  50,  46,np.NaN),
    (0.045,  50,  77, 0.648), (0.045,  50, 129, 0.831), (0.045,  50, 215, 0.942),
    (0.045,  50, 318, 0.996), (0.045,  50, 464, 1.033), (0.045,  50, 774, 1.068),
    (0.045,  50,1292, 1.090), (0.045,  50,2154, 1.081), (0.045,  50,3594, 1.042),
    (0.045, 100,  17,np.NaN), (0.045, 100,  28,np.NaN), (0.045, 100,  46,np.NaN),
    (0.045, 100,  77,np.NaN), (0.045, 100, 129, 0.587), (0.045, 100, 215, 0.798),
    (0.045, 100, 318, 0.896), (0.045, 100, 464, 0.961), (0.045, 100, 774, 1.026),
    (0.045, 100,1292, 1.062), (0.045, 100,2154, 1.063), (0.045, 100,3594, 1.032),
    (  0.1,   0,  17, 1.068), (  0.1,   0,  28, 1.027), (  0.1,   0,  46, 1.005),
    (  0.1,   0,  77, 1.024), (  0.1,   0, 129, 1.062), (  0.1,   0, 215, 1.085),
    (  0.1,   0, 318, 1.090), (  0.1,   0, 464, 1.092), (  0.1,   0, 774, 1.099),
    (  0.1,   0,1292, 1.104), (  0.1,   0,2154, 1.084), (  0.1,   0,3594, 1.038),
    (  0.1,  10,  17, 0.592), (  0.1,  10,  28, 0.755), (  0.1,  10,  46, 0.858),
    (  0.1,  10,  77, 0.942), (  0.1,  10, 129, 1.008), (  0.1,  10, 215, 1.051),
    (  0.1,  10, 318, 1.067), (  0.1,  10, 464, 1.077), (  0.1,  10, 774, 1.090),
    (  0.1,  10,1292, 1.098), (  0.1,  10,2154, 1.080), (  0.1,  10,3594, 1.036),
    (  0.1,  25,  17,np.NaN), (  0.1,  25,  28, 0.404), (  0.1,  25,  46, 0.675),
    (  0.1,  25,  77, 0.829), (  0.1,  25, 129, 0.934), (  0.1,  25, 215, 1.002),
    (  0.1,  25, 318, 1.034), (  0.1,  25, 464, 1.054), (  0.1,  25, 774, 1.077),
    (  0.1,  25,1292, 1.090), (  0.1,  25,2154, 1.075), (  0.1,  25,3594, 1.033),
    (  0.1,  50,  17,np.NaN), (  0.1,  50,  28,np.NaN), (  0.1,  50,  46,np.NaN),
    (  0.1,  50,  77, 0.639), (  0.1,  50, 129, 0.817), (  0.1,  50, 215, 0.928),
    (  0.1,  50, 318, 0.982), (  0.1,  50, 464, 1.019), (  0.1,  50, 774, 1.055),
    (  0.1,  50,1292, 1.076), (  0.1,  50,2154, 1.066), (  0.1,  50,3594, 1.027),
    (  0.1, 100,  17,np.NaN), (  0.1, 100,  28,np.NaN), (  0.1, 100,  46,np.NaN),
    (  0.1, 100,  77,np.NaN), (  0.1, 100, 129, 0.582), (  0.1, 100, 215, 0.788),
    (  0.1, 100, 318, 0.884), (  0.1, 100, 464, 0.949), (  0.1, 100, 774, 1.014),
    (  0.1, 100,1292, 1.049), (  0.1, 100,2154, 1.049), (  0.1, 100,3594, 1.018),
    (  1.0,   0,  17, 1.014), (  1.0,   0,  28, 0.993), (  1.0,   0,  46, 0.983),
    (  1.0,   0,  77, 1.011), (  1.0,   0, 129, 1.050), (  1.0,   0, 215, 1.074),
    (  1.0,   0, 318, 1.081), (  1.0,   0, 464, 1.084), (  1.0,   0, 774, 1.091),
    (  1.0,   0,1292, 1.096), (  1.0,   0,2154, 1.075), (  1.0,   0,3594, 1.030),
    (  1.0,  10,  17, 0.576), (  1.0,  10,  28, 0.738), (  1.0,  10,  46, 0.845),
    (  1.0,  10,  77, 0.931), (  1.0,  10, 129, 0.997), (  1.0,  10, 215, 1.041),
    (  1.0,  10, 318, 1.058), (  1.0,  10, 464, 1.068), (  1.0,  10, 774, 1.082),
    (  1.0,  10,1292, 1.090), (  1.0,  10,2154, 1.072), (  1.0,  10,3594, 1.028),
    (  1.0,  25,  17,np.NaN), (  1.0,  25,  28, 0.400), (  1.0,  25,  46, 0.666),
    (  1.0,  25,  77, 0.820), (  1.0,  25, 129, 0.924), (  1.0,  25, 215, 0.993),
    (  1.0,  25, 318, 1.026), (  1.0,  25, 464, 1.046), (  1.0,  25, 774, 1.069),
    (  1.0,  25,1292, 1.082), (  1.0,  25,2154, 1.067), (  1.0,  25,3594, 1.025),
    (  1.0,  50,  17,np.NaN), (  1.0,  50,  28,np.NaN), (  1.0,  50,  46,np.NaN),
    (  1.0,  50,  77, 0.633), (  1.0,  50, 129, 0.810), (  1.0,  50, 215, 0.920),
    (  1.0,  50, 318, 0.974), (  1.0,  50, 464, 1.011), (  1.0,  50, 774, 1.048),
    (  1.0,  50,1292, 1.068), (  1.0,  50,2154, 1.058), (  1.0,  50,3594, 1.020),
    (  1.0, 100,  17,np.NaN), (  1.0, 100,  28,np.NaN), (  1.0, 100,  46,np.NaN),
    (  1.0, 100,  77,np.NaN), (  1.0, 100, 129, 0.578), (  1.0, 100, 215, 0.782),
    (  1.0, 100, 318, 0.878), (  1.0, 100, 464, 0.942), (  1.0, 100, 774, 1.007),
    (  1.0, 100,1292, 1.041), (  1.0, 100,2154, 1.041), (  1.0, 100,3594, 1.010),
    (  9.5,   0,  17, 0.798), (  9.5,   0,  28, 0.827), (  9.5,   0,  46, 0.866),
    (  9.5,   0,  77, 0.913), (  9.5,   0, 129, 0.957), (  9.5,   0, 215, 0.994),
    (  9.5,   0, 318, 1.019), (  9.5,   0, 464, 1.037), (  9.5,   0, 774, 1.056),
    (  9.5,   0,1292, 1.062), (  9.5,   0,2154, 1.055), (  9.5,   0,3594, 1.023),
    (  9.5,  10,  17, 0.508), (  9.5,  10,  28, 0.653), (  9.5,  10,  46, 0.759),
    (  9.5,  10,  77, 0.844), (  9.5,  10, 129, 0.911), (  9.5,  10, 215, 0.966),
    (  9.5,  10, 318, 0.999), (  9.5,  10, 464, 1.024), (  9.5,  10, 774, 1.048),
    (  9.5,  10,1292, 1.057), (  9.5,  10,2154, 1.052), (  9.5,  10,3594, 1.021),
    (  9.5,  25,  17,np.NaN), (  9.5,  25,  28, 0.378), (  9.5,  25,  46, 0.611),
    (  9.5,  25,  77, 0.750), (  9.5,  25, 129, 0.849), (  9.5,  25, 215, 0.926),
    (  9.5,  25, 318, 0.972), (  9.5,  25, 464, 1.037), (  9.5,  25, 774, 1.035),
    (  9.5,  25,1292, 1.050), (  9.5,  25,2154, 1.047), (  9.5,  25,3594, 1.018),
    (  9.5,  50,  17,np.NaN), (  9.5,  50,  28,np.NaN), (  9.5,  50,  46,np.NaN),
    (  9.5,  50,  77, 0.594), (  9.5,  50, 129, 0.754), (  9.5,  50, 215, 0.865),
    (  9.5,  50, 318, 0.926), (  9.5,  50, 464, 0.973), (  9.5,  50, 774, 1.015),
    (  9.5,  50,1292, 1.037), (  9.5,  50,2154, 1.039), (  9.5,  50,3594, 1.013),
    (  9.5, 100,  17,np.NaN), (  9.5, 100,  28,np.NaN), (  9.5, 100,  46,np.NaN),
    (  9.5, 100,  77,np.NaN), (  9.5, 100, 129, 0.558), (  9.5, 100, 215, 0.746),
    (  9.5, 100, 318, 0.842), (  9.5, 100, 464, 0.911), (  9.5, 100, 774, 0.976),
    (  9.5, 100,1292, 1.012), (  9.5, 100,2154, 1.023), (  9.5, 100,3594, 1.004),
    ]

class TestPlanetPhysicalModelPrototype(TestPlanetPhysicalModelMixin,
                                       unittest.TestCase):
    r"""Test the FortneyMarleyCahoyMix1 specialization of PlanetPhysicalModel."""

    # over-ride to set up the specific test fixture (class constructor) we will use
    planet_model = FortneyMarleyCahoyMix1

    # planetary radius range endpoints (log10 scale, multiple of R_earth)
    planetary_radius_endpoints = (-0.3, 1.5)
    # allowed relative error in roundtrip radius->mass->radius computation
    # NB: We have to be very, very generous for now, because the individual transformations
    # use side parameters that are filled in, sometimes inconsistently, across the
    # mass -> radius vs. radius -> mass conversions.
    delta_roundtrip = 4.0

    # over-ride the generation of planetary radii, in the roundtrip check only.
    # there is a discontinuity around 2.0 * R_earth, so this uses the same range above,
    # but filters out radii near 2 * R_earth.  It's not totally satisfactory.
    def roundtrip_radius_generator(self, n_radii):
        # just the standard evenly-spaced radii
        r_initial = np.logspace(self.planetary_radius_endpoints[0],
                                self.planetary_radius_endpoints[1],
                                num=n_radii)
        # filter out the ones near 2.0
        r_final = r_initial[np.fabs(np.log(r_initial/2.0)) < 0.3]
        # return the filtered set
        return r_final

    ############################################################
    ## Test methods are below this point.

    def test_albedo_pointwise(self):
        r"""Test that the known values of the Cahoy et al. albedo are correct.

        Method: Loop over each known value.  The reference (Cahoy) albedo uses
        two parameters, semi-major-axis + metallicity, to determine albedo.
        The method under test only uses semi-major-axis, and generates metallicity
        at random.  This test patches the numpy random-number routine to return a fixed
        metallicity value in order to test against known values.
        Note that, for non-tabulated values, the returned albedo can differ significantly
        from nearby values because bi-cubic interpolation is used."""

        print 'albedo-pointwise'
        ppmodel = self.fixture()
        # check each known albedo
        for table_line in albedo_tabulation:
            # unpack the line
            (semi_major_axis1, metallicity, _, true_albedo, _) = table_line
            # make a numpy array
            semi_major_axes = np.array([semi_major_axis1]) * u.AU
            # patch the numpy random number generator with one that returns
            # the metallicity figure when called
            with patch_uniform(metallicity):
                test_albedo = ppmodel.calc_albedo_from_sma(semi_major_axes)

            self.assertEqual(test_albedo, true_albedo)

        #np.random.uniform = save_uniform

    def test_radius_mass_radius(self):
        # This is not a very effective test, because the procedures used within
        # calc_mass_from_radius are not invertible.  They use a planet-type switch
        # (rock/ice, iron/rock, gas-giant) that changes the formulas, but
        # is not accessible from outside the method.  
        # These routines would have to be changed to supply this selector externally
        # in order to improve this test.
        # However, the underlying formulae are checked in another test in this class.

        print 'superclass radius-mass-radius'
        # patch np.random.uniform for these tests, to at least return non-random values.
        with patch_uniform(None):
            # apply the standard roundtrip test
            super(TestPlanetPhysicalModelPrototype, self).test_radius_mass_radius()

    def test_radius_mass_rocky_helpers(self):
        r"""Test rock/ice and iron/rock radius/mass relations against Fortney, Marley, & Barnes (2007).

        Method: Loop over a range of (mass, ice/rock or rock/iron) values.
        The reference formulae (re-implemented above) are checked against the
        corresponding formulae as implemented in the EXOSIMS code.
        This is testing methods that live within the class that are not in the
        main API, so if their implementation or names change, this test will
        have to be revised."""

        print 'radius-to-mass helper functions'
        ppmodel = self.fixture()

        # give a decent message if required helpers are not present
        self.assertIn('R_ir', ppmodel.__dict__)
        self.assertIn('R_ri', ppmodel.__dict__)
        self.assertIn('M_ir', ppmodel.__dict__)
        self.assertIn('M_ri', ppmodel.__dict__)
        # mass -> radius formulas
        for frac in np.linspace(0.0, 1.0, 21):
            for mass in np.logspace(-1.0, 2.0, 21):
                # 1:  ice/rock case
                # 1A: mass -> radius
                radius_1 = ppmodel.R_ir(frac, mass)
                radius_2 = mass2radius_ice_rock(frac, np.log10(mass))
                self.assertAlmostEqual(radius_1, radius_2, delta=1e-7)
                # 1B: roundtrip: back to mass
                mass_back = ppmodel.M_ir(frac, radius_1)
                self.assertAlmostEqual(mass, mass_back, delta=1e-7)
                # 2:  rock/iron case
                # 2A: mass -> radius
                radius_1 = ppmodel.R_ri(frac, mass)
                radius_2 = mass2radius_rock_iron(frac, np.log10(mass))
                self.assertAlmostEqual(radius_1, radius_2, delta=1e-7)
                # 2B: roundtrip: back to mass
                mass_back = ppmodel.M_ri(frac, radius_1)
                self.assertAlmostEqual(mass, mass_back, delta=1e-7)
                
    def test_radius_mass_giant_helper(self):
        r"""Test gas-giant radius/mass relations against Fortney, Marley, & Barnes (2007).

        Method: Loop over a grid of tabulated (planet_mass, radius) values, which
        also vary (core_mass, orbital_distance).  The reference table is checked
        against the corresponding table as implemented in the EXOSIMS code.
        This is testing some methods that live within the class, but are not in the
        main API, so if their implementation or names change, this test will
        have to be revised."""

        print 'radius-to-mass helper functions: gas giants'
        ppmodel = self.fixture()

        # give a decent message if required helpers are not present
        self.assertIn('giant_pts2', ppmodel.__dict__)
        self.assertIn('giant_vals2', ppmodel.__dict__)
        # loop over tabulated tuples 
        for tuple in giant_planet_radii:
            # the tuple in this code contains:
            #   (dist[AU], core_mass[M_earth], p_mass[M_earth], radius[R_jup])
            (dist, core_mass, p_mass, radius) = tuple
            # (distance, core_mass, planet_mass) -> radius
            r_lookup_array = interpolate.griddata(ppmodel.giant_pts2,
                                                  ppmodel.giant_vals2,
                                                  np.array([core_mass, dist, p_mass]))
            r_lookup = r_lookup_array[0] * (const.R_earth/const.R_jup)
            
            # it is OK if *both* are NaN
            if np.isnan(r_lookup) and np.isnan(radius):
                continue

            # at this point, it's an error if either is NaN
            # How this is to be handled is debatable.
            if np.isnan(r_lookup) or np.isnan(radius):
                print 'NaN mismatch at (dist = %.2f, core = %.2f, mass = %.2f)' % (dist, core_mass, p_mass)
                continue

            self.assertAlmostEqual(radius, r_lookup, delta=1e-7)
            # 1B: roundtrip: back to mass
            #mass_back = ppmodel.M_ir(frac, radius_1)
            #self.assertAlmostEqual(mass, mass_back, delta=1e-7)



class patch_uniform(object):
    r"""Used in a `with' statement to patch the numpy random number generator (rng).
    """
    # reference to the stored numpy method
    np_method = None
    # do we want to specify a value for the patched rng to return?
    np_value = None
    def __init__(self, value):
        """Supply a fixed numerical value to return, or None to return (low+high)/2.
        """
        self.np_value = value
    def __enter__(self):
        self.np_method = np.random.uniform # save it
        np.random.uniform = random_uniform_replacement(self.np_value)
    def __exit__(self, type, value, traceback):
        np.random.uniform = self.np_method # restore


def random_uniform_replacement(value=None):
    r"""Replacement for numpy.random.uniform, used for monkey-patching during testing.

    Two behaviors are supported:
    (1) If value is not given, return a false random number generator (a function) that
    deterministically returns a value halfway between the low and high value of the
    distribution.
    (2) If a value is given, return a false random number generator that deterministically
    returns the supplied value."""
    def my_uniform_rng(size=1, low=0.0, high=1.0, saved_value=value):
        if saved_value is not None:
            # the fixed, externally-supplied value
            rv = np.zeros(size) + saved_value
        else:
            # non-random, halfway between low and high
            rv = np.zeros(size) + (low+high)*0.5
        #print '** Returning', rv
        return rv
    return my_uniform_rng

if __name__ == '__main__':
    unittest.main()
