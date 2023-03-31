#!/usr/local/bin/python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>

r"""OpticalSystem module unit tests

Michael Turmon, JPL, May 2016
"""

import os
import sys
import re
import numbers
import unittest
import inspect
from copy import deepcopy
from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
from tests.TestSupport.Info import resource_path
import numpy as np
import astropy.units as u

#
# A few specs dictionaries that can be used to instantiate OpticalSystem objects
#

# the most basic one allowed
specs_default = {
    "scienceInstruments": [
        {"name": "imager"},
        {"name": "spectrograph"},
    ],
    "starlightSuppressionSystems": [
        {"name": "umbrella"},
    ],
}


# a less basic example
specs_simple = {
    "pupilDiam": 2.37,
    "obscurFac": 0.2,
    "intCutoff": 100,
    "scienceInstruments": [
        {
            "name": "imaging-EMCCD",
            "QE": 0.88,
            "CIC": 0.0013,
            "sread": 16,
            "ENF": 1.414,
        }
    ],
    "starlightSuppressionSystems": [
        {
            "name": "internal-imaging-HLC",
            "IWA": 0.1,
            "OWA": 0,
            "lam": 565,
            "BW": 0.10,
            "core_thruput": 1,
            "core_contrast": 1,
            "PSF": 1,
        }
    ],
}

# multiple instruments + shades
specs_multi = {
    "pupilDiam": 2.37,
    "obscurFac": 0.2,
    "intCutoff": 100,
    "scienceInstruments": [
        {
            "name": "imaging-EMCCD",
            "QE": 0.88,
            "CIC": 0.0013,
            "sread": 16,
            "ENF": 1.414,
        },
        {
            "name": "spectrograph",
            "QE": 0.88,
            "sread": 16,
            "ENF": 1.414,
        },
    ],
    "starlightSuppressionSystems": [
        {
            "name": "internal-imaging-HLC",
            "IWA": 0.1,
            "OWA": 0,
            "core_thruput": 1,
            "core_contrast": 1,
            "PSF": 1,
            "lam": 565,
            "BW": 0.10,
        },
        {
            "name": "umbrella",
            "occulter": True,
            "IWA": 0.2,
            "OWA": 60,
            "core_contrast": 0.5,
            "lam": 565,
            "BW": 0.10,
            "PSF": 1,
        },
    ],
}

#
# Test Metatata
#

# a few of the attributes we require to be present in OpticalSystem
attr_expect = [
    "IWA",
    "OWA",
    "haveOcculter",
    "intCutoff",
    "obscurFac",
    "observingModes",
    "pupilArea",
    "pupilDiam",
    "scienceInstruments",
    "shapeFac",
    "starlightSuppressionSystems",
]


# This dictionary-of-dictionaries lays out a program of unit tests for
# every input parameter used by the Prototype OpticalSystem.
#
# The outer dictionary maps parameter names (like shapeFac) to an inner
# dictionary d.  The inner dictionary contains fields:
#   default = function default value (unused by this code)
#   trial = list of trial values to use to instantiate the OpticalSystem object with
#      - for instance, if trial=(0.1,0.5), two tests are run, first with 0.1, then
#        with 0.5, to be sure the parameter is set.
#   unit = float or astropy unit
#      - to check the units of the parameter
#   target = 0, 1, or 2
#      - 0 if the parameter goes in the OpticalSystem root level (OpticalSystem.shapeFac),
#        1 if it goes in the scienceInstruments, 2 if it goes in starlightSuppressionSystems.
#   optionally:
#   raises = a list of exceptions raised, or None if no exception is raised
#      - this list is in 1:1 correspondence with the trial values in "trial",
#        e.g., if the third "trial" value will cause an assertion to fail, then
#        the third "raises" value will be AssertionError.
#
# Summary of possible tests
# -------------------------
# Typically, the trial values are scalars.  EXOSIMS does not validate the type or
# range of scalars, so no value is prima facie illegal.  Thus, we validate by poking
# in a few test values, and being sure they are set by EXOSIMS.
#
# EXOSIMS does validate the scienceInstruments and starlightSuppressionSystems.  We
# have inserted tests to be sure that EXOSIMS raises assertions when these
# attributes are not set correctly.  We test that EXOSIMS plugs in correct values
# for these attributes.
#
# The hard case is interpolants (like "QE", etc.) -- the trial values are strings
# that correspond to filenames in the directory pointed to by resource_path, and
# both errors and nontrivial processing can occur.
# In this case, the string named in "trials" will go to EXOSIMS, and the interpolant
# will be loaded by EXOSIMS from the named FITS file.
# There are two kinds of interpolants: [1] functions (QE, throughput, contrast), and
# [2] matrices (PSF).
# [1]: The FITS files for the first kind are given as a specific (quadratic) form
# ("quad100.fits", etc.).  We send the file to EXOSIMS and the test code checks that
# the correct values are being computed by the interpolant, by performing random probes.
# The form used is 0.25 + 3*x*(1-x) for x between 0 and 1.  This ranges from 0.25 to 1 and
# back down to 0.25, and is smooth enough so the cubic interpolation done by EXOSIMS agrees
# very closely with the actual value.
# [2]: The FITS file for PSF is always given as an MxN matrix with ascending numbers,
# 0...M*N-1.  This allows us to check that EXOSIMS reads the FITS file correctly without
# transpositions.
# There is, additionally, a third, simpler, kind of interpolant -- fixed values.
# This is the case where (say) contrast is given as a scalar, like 0.1.  In this case,
# EXOSIMS again returns an interpolant -- a function of wavelength and working angle.
# But the function returns the same thing for any input.  We also test this interpolant
# with random probes, making sure it equals the given value.

opsys_params = dict(
    # group GP: general optical system parameters
    obscurFac=dict(default=0.2, trial=(0.1, 0.5), unit=float, target=0),
    shapeFac=dict(default=np.pi / 4, trial=(0.1, 0.5), unit=float, target=0),
    pupilDiam=dict(default=4.0, trial=(1.0, 10.0), unit=u.m, target=0),
    intCutoff=dict(default=50, trial=(10, 100), unit=u.d, target=0),
    # scienceInstruments: a special case.  we ensure the error checking is OK for
    # illegal arguments -- it must be a list of dicts, each containing a 'type' key
    scienceInstruments=dict(
        default=None,
        trial=(
            None,
            {},
            [],
            [{}],
            [{"name": "imager"}],
            [{"name": "imager"}, {"name": "spectrograph"}],
        ),
        raises=(AssertionError,) * 4 + (None,) * 2,
        unit=None,
        target=0,
    ),
    # group SIP: science instrument parameters
    lam=dict(default=500, trial=(400, 400.0, 600, 600.0), unit=u.nm, target=2),
    BW=dict(default=0.2, trial=(0.1, 0.9), unit=float, target=2),
    pixelSize=dict(default=13e-6, trial=(1e-6, 1e-2), unit=u.m, target=1),
    focal=dict(default=240, trial=(200, 200.0), unit=u.m, target=1),
    idark=dict(default=9e-5, trial=(1e-6, 1e-2), unit=1 / u.s, target=1),
    texp=dict(default=1e3, trial=(1e2, 100), unit=u.s, target=1),
    sread=dict(default=3, trial=(2.0, 2, 5), unit=float, target=1),
    CIC=dict(default=0.0013, trial=(0.1, 0.01), unit=float, target=1),
    ENF=dict(default=1, trial=(1.5, 2), unit=float, target=1),
    PCEff=dict(default=1, trial=(0.5, 1.0), unit=float, target=1),
    # [the following test fails because Rs is now [10/2016] assigned conditionally]
    # Rs = dict(default=70, trial=(50,50.0), unit=float, target=1),
    # QE: a couple of constant values, the quadratic interpolants, and
    # various error values.
    # This pattern is followed for throughput and contrast.
    QE=dict(
        default=0.9,
        trial=(
            0.1,
            0.2,
            "i_quad100.fits",
            "i_quad100t.fits",
            # errors follow
            "i_err_nofile.fits",
            "i_err_3d.fits",
            "i_err_2d_bad.fits",
            "i_err_2d_bad_t.fits",
            "i_err_negative.fits",
        ),
        raises=(None,) * 4 + (AssertionError,) * 5,
        unit=1 / u.photon,
        target=1,
    ),
    # starlightSuppressionSystems: a special case.  we ensure the error checking is OK for
    # illegal arguments -- it must be a list of dicts, each containing a 'type' key
    # the last trials are minimal acceptable systems
    starlightSuppressionSystems=dict(
        default=None,
        trial=(
            None,
            {},
            [],
            [{}],
            [{"name": "parasol"}],
            [{"name": "parasol"}, {"name": "umbrella"}],
        ),
        raises=(AssertionError,) * 4 + (None,) * 2,
        unit=None,
        target=0,
    ),
    # group SSP: starlight suppression parameters
    core_thruput=dict(
        default=1e-2,
        trial=(
            0.02,
            0.04,
            "i_quad100.fits",
            "i_quad100t.fits",
            # errors follow
            "err_nofile.fits",
            "err_3d.fits",
            "i_err_2d_bad.fits",
            "i_err_2d_bad_t.fits",
            "i_err_negative.fits",
        ),
        raises=(None,) * 4 + (AssertionError,) * 5,
        unit=float,
        target=2,
    ),
    core_contrast=dict(
        default=1e-9,
        trial=(
            1e-8,
            1e-6,
            "i_quad100.fits",
            "i_quad100t.fits",
            # errors follow
            "err_nofile.fits",
            "err_3d.fits",
            "i_err_2d_bad.fits",
            "i_err_2d_bad_t.fits",
            "i_err_negative.fits",
        ),
        raises=(None,) * 4 + (AssertionError,) * 5,
        unit=float,
        target=2,
    ),
    # PSF is slightly different.  The fixed trial values are numpy
    # matrices, and two FITS files as described above.  There are also
    # four types of error-causing values.
    PSF=dict(
        default=np.ones((3, 3)),
        trial=(
            np.random.rand(3, 3),
            np.random.rand(5, 5),
            "psf_5x5.fits",
            "psf_11x11.fits",
            # errors follow
            np.zeros((0, 0)),
            np.zeros((3, 3)),
            "err_3d.fits",
            "err_psf_zero.fits",
        ),
        raises=(None,) * 4 + (AssertionError,) * 4,
        unit=float,
        target=2,
    ),
    core_platescale=dict(default=10, trial=(), unit=u.arcsec, target=2),
    ohTime=dict(default=1, trial=(), unit=u.d, target=2),
    timeMultiplier=dict(default=1, trial=(), unit=float, target=2),
    # group FP: fundamental IWA, OWA
    #   -- related to SSP parameters
    IWA=dict(default=None, trial=(1, 10), unit=u.arcsec, target=0),
    OWA=dict(default=None, trial=(10, 100), unit=u.arcsec, target=0),
)


class TestOpticalSystemMethods(unittest.TestCase):
    r"""Test OpticalSystem class."""

    # display of assert-specific as well as generic message
    longMessage = True

    def setUp(self):
        self.fixture = OpticalSystem

    def tearDown(self):
        pass

    def validate_basic(self, optsys, spec={}):
        r"""Basic validation of an OpticalSystem object."""
        # check for presence of some class attributes
        for att in attr_expect:
            self.assertIn(att, optsys.__dict__)
            self.assertIsNotNone(optsys.__dict__[att])
        # optionally, check against a supplied reference dictionary
        for (att, val) in spec.items():
            self.assertIn(att, optsys.__dict__)
            val_e = optsys.__dict__[att]
            if isinstance(val_e, u.quantity.Quantity):
                # val has no units
                self.assertEqual(val_e.value, val)
            elif isinstance(val_e, dict):
                # weak check
                self.assertEqual(type(val_e), type(val))
            elif isinstance(val_e, list):
                # weak check
                self.assertEqual(type(val_e), type(val))
                self.assertEqual(len(val_e), len(val))
            else:
                self.assertEqual(val_e, val)

    def test_init(self):
        r"""Test of initialization and __init__ -- simple.

        Method: Instantiate OpticalSystem objects from a few known-correct
        specs lists, and verify that the object is valid and expected fields
        are in place.
        """
        for specs in [specs_default, specs_simple, specs_multi]:
            # the input dict is modified in-place -- so copy it
            optsys = self.fixture(**deepcopy(specs))
            self.validate_basic(optsys, specs)

    def test_init_occulter(self):
        r"""Test of initialization and __init__ -- occulter.

        Method: If any starlight suppression system has an occulter , the
        attribute OpticalSystem.haveOcculter is set.
        We instantiate OpticalSystem objects and verify that this is done.
        """
        our_specs = deepcopy(specs_default)
        optsys = self.fixture(**deepcopy(our_specs))
        self.assertFalse(optsys.haveOcculter, "Expect to NOT haveOcculter")

        our_specs["starlightSuppressionSystems"][0]["occulter"] = True
        optsys = self.fixture(**deepcopy(our_specs))
        self.assertTrue(optsys.haveOcculter, "Expect to haveOcculter")

        optsys = self.fixture(**deepcopy(specs_multi))
        self.assertTrue(optsys.haveOcculter, "Expect to haveOcculter")

    def test_init_owa_inf(self):
        r"""Test of initialization and __init__ -- OWA.

        Method: An affordance to allow you to set OWA = +Infinity from a JSON
        specs-file is offered by OpticalSystem: if OWA is supplied as 0, it is
        set to +Infinity.  We instantiate OpticalSystem objects and verify that
        this is done.
        """
        for specs in [specs_default, specs_simple, specs_multi]:
            # the input dict is modified in-place -- so copy it
            our_specs = deepcopy(specs)
            our_specs["OWA"] = 0
            for syst in our_specs["starlightSuppressionSystems"]:
                syst["OWA"] = 0
            optsys = self.fixture(**deepcopy(our_specs))
            for syst in optsys.starlightSuppressionSystems:
                self.assertTrue(np.isposinf(syst["OWA"].value))
        # repeat, but allow the special value to propagate up
        for specs in [specs_default, specs_simple, specs_multi]:
            # the input dict is modified in-place -- so copy it
            our_specs = deepcopy(specs)
            for syst in our_specs["starlightSuppressionSystems"]:
                syst["OWA"] = 0
            optsys = self.fixture(**deepcopy(our_specs))
            for syst in optsys.starlightSuppressionSystems:
                self.assertTrue(np.isposinf(syst["OWA"].value))

    def test_init_iwa_owa(self):
        r"""Test of initialization and __init__ -- IWA, OWA.

        Method: We instantiate OpticalSystem objects and verify
        various IWA/OWA relationships.
        """
        for specs in [specs_default, specs_simple, specs_multi]:
            # the input dict is modified in-place -- so copy it
            our_specs = deepcopy(specs)
            # set root object IWA and OWA in conflict
            our_specs["IWA"] = 10
            our_specs["OWA"] = 1
            with self.assertRaises(AssertionError):
                optsys = self.fixture(**deepcopy(our_specs))
        for specs in [specs_default, specs_simple, specs_multi]:
            # various settings of sub-object IWA and OWA
            for IWA, OWA in zip([0.1, 0.1, 1, 1, 2, 2], [0.5, 1.5, 0.5, 1.5, 0.5, 1.5]):
                # the input dict is modified in-place -- so copy it
                our_specs = deepcopy(specs)
                # set sub-object IWA and OWA
                for syst in our_specs["starlightSuppressionSystems"]:
                    syst["IWA"] = IWA
                    syst["OWA"] = OWA
                if IWA < OWA:
                    # will succeed in this case
                    optsys = self.fixture(**deepcopy(our_specs))
                    # they must propagate up to main object
                    self.assertTrue(optsys.OWA.value == OWA)
                    self.assertTrue(optsys.IWA.value == IWA)
                else:
                    # they propagate up, and cause a failure
                    with self.assertRaises(AssertionError):
                        optsys = self.fixture(**deepcopy(our_specs))

    def test_init_iwa_owa_contrast(self):
        r"""Test of initialization and __init__ -- IWA, OWA vs. contrast domain constraint.

        Method: We instantiate OpticalSystem objects and verify
        that IWA and OWA vary as expected with the domain of WA of the contrast
        lookup table (from 0 to 1).
        """
        filename = os.path.join(resource_path(), "OpticalSystem", "i_quad100.fits")
        Cmin = 0.25  # the minimum of the above lookup table is 0.25
        expected_dmaglim = -2.5 * np.log10(Cmin)
        # the input dict is modified in-place -- so copy it
        our_specs = deepcopy(specs_default)
        for (IWA, OWA) in zip([0.0, 0.2, 0.5, 1.1], [1.0, 1.4, 1.6, 2.0]):
            for syst in our_specs["starlightSuppressionSystems"]:
                syst["core_contrast"] = filename
                syst["IWA"] = IWA
                syst["OWA"] = OWA
            if IWA <= 1.0:
                optsys = self.fixture(**deepcopy(our_specs))
                # Check that the range constraint of the contrast lookup table
                # (it covers WA in 0 to 1) does constrain the system OWA and IWA.
                self.assertTrue(
                    optsys.OWA.value == min(1.0, OWA), msg="contrast lookup table OWA"
                )
                self.assertTrue(
                    optsys.IWA.value == max(0.0, IWA), msg="contrast lookup table IWA"
                )
            else:
                # IWA > 1 but lookup table covers [0,1] -- conflict
                with self.assertRaises(AssertionError):
                    optsys = self.fixture(**deepcopy(our_specs))

    def test_init_iwa_owa_throughput(self):
        r"""Test of initialization and __init__ -- IWA, OWA vs. throughput domain constraint.

        Method: We instantiate OpticalSystem objects and verify
        that IWA and OWA vary as expected with the domain of WA of the throughput
        lookup table (from 0 to 1).
        """
        filename = os.path.join(resource_path(), "OpticalSystem", "i_quad100.fits")

        our_specs = deepcopy(specs_default)
        for (IWA, OWA) in zip([0.0, 0.2, 0.5, 1.1], [1.0, 1.4, 1.6, 2.0]):
            for syst in our_specs["starlightSuppressionSystems"]:
                syst["core_thruput"] = filename
                syst["IWA"] = IWA
                syst["OWA"] = OWA
            if IWA <= 1.0:
                optsys = self.fixture(**deepcopy(our_specs))
                # Check that the range constraint of the throughput lookup table
                # (it covers WA in 0 to 1) does constrain the system OWA and IWA.
                self.assertTrue(optsys.OWA.value == min(1.0, OWA))
                self.assertTrue(optsys.IWA.value == max(0.0, IWA))
            else:
                # IWA > 1 but lookup table covers [0,1] -- conflict
                with self.assertRaises(AssertionError):
                    optsys = self.fixture(**deepcopy(our_specs))

    @unittest.skip("PSF sampling handling needs to be updated.")
    def test_init_psf(self):
        r"""Test of initialization and __init__ -- PSF

        Method: We instantiate OpticalSystem objects and verify
        that IWA and OWA vary as expected with the domain of WA of the throughput
        lookup table (from 0 to 1).
        """
        filename = os.path.join(resource_path(), "OpticalSystem", "psf_5x5.fits")
        sampling = 1.234e-5 * u.arcsec  # sampling rate keyword in above file
        for specs in [specs_default]:
            # the input dict is modified in-place -- so copy it
            our_specs = deepcopy(specs_default)
            for syst in our_specs["starlightSuppressionSystems"]:
                syst["PSF"] = filename
            optsys = self.fixture(**deepcopy(our_specs))
            # Check that the sampling rate is correct
            self.assertEqual(optsys.starlightSuppressionSystems[0]["samp"], sampling)
            # Check that the PSF is present and has right size
            # Values are checked elsewhere
            psf = optsys.starlightSuppressionSystems[0]["PSF"](1.0, 1.0)
            self.assertIsInstance(psf, np.ndarray)
            self.assertEqual(psf.shape, (5, 5))

    def outspec_compare(self, outspec1, outspec2):
        r"""Compare two _outspec dictionaries.

        This is in service of the roundtrip comparison, test_roundtrip."""
        self.assertEqual(sorted(list(outspec1)), sorted(list(outspec2)))
        for k in outspec1:
            # check for scienceInstrument, starlightSuppression, and observingModes,
            # which are all lists of dicts
            if isinstance(outspec1[k], list) and isinstance(outspec1[k][0], dict):
                # this happens for scienceInstrument and starlightSuppression,
                # which are lists of dictionaries
                for (d1, d2) in zip(outspec1[k], outspec2[k]):
                    for kk in d1:
                        if kk.split("_")[0] == "koAngles":
                            self.assertEqual(d1[kk][0], d2[kk][0])
                            self.assertEqual(d1[kk][1], d2[kk][1])
                        else:
                            self.assertEqual(d1[kk], d2[kk])
            elif isinstance(outspec1[k], np.ndarray):
                self.assertTrue(
                    np.all(outspec1[k] == outspec2[k]),
                    f"outspecs don't match for attribute {k}",
                )

            else:
                # these should all be things we can directly compare
                self.assertEqual(
                    outspec1[k], outspec2[k], f"outspecs don't match for attribute {k}"
                )

    def test_roundtrip(self):
        r"""Test of initialization and __init__ -- round-trip parameter check.

        Method: Instantiate an OpticalSystem, use its resulting _outspec to
        instantiate another OpticalSystem.  Assert that all quantities in each
        dictionary are the same.  This checks that OpticalSystem objects are
        in fact reproducible from the _outspec alone.
        """
        optsys = self.fixture(**deepcopy(specs_simple))
        self.validate_basic(optsys, specs_simple)
        # save the _outspec
        outspec1 = optsys._outspec
        # recycle the _outspec into a new OpticalSystem
        optsys_next = self.fixture(**deepcopy(outspec1))
        # this is the new _outspec
        outspec2 = optsys_next._outspec
        # ensure the two _outspec's are the same
        self.outspec_compare(outspec1, outspec2)

    def make_interpolant(self, param, value, unit):
        """Make an interpolating function."""

        # the generic quadratic function we have used to test
        # always positive, reaches a maximum of 1.0 at x=0.5
        quadratic = lambda x: 0.25 + 3.0 * (1 - x) * x

        if isinstance(value, (numbers.Number, np.ndarray)):
            if param == "QE":
                return lambda lam: value * unit
            elif param in ("core_thruput", "core_contrast", "PSF"):
                # for PSF, will be an ndarray
                return lambda lam, WA: value * unit
        elif isinstance(value, str):
            # for most ty
            if param == "QE":
                return lambda lam: quadratic(lam) * unit
            elif param in ("core_thruput", "core_contrast"):
                return lambda lam, WA: quadratic(WA) * unit
            elif param == "PSF":
                # this rather messy construct uses a value like
                # "psf_5x5.fits" to recover a PSF matrix to use, else,
                # it uses a fixed value.  The pattern matches
                # psf_NNNxMMM.fits where NNN and MMM are digit sequences.
                m = re.search("psf_(\d+)x(\d+)\.fits", value)
                if m is None:
                    # use a fixed value, we won't need it anyway
                    a_value = np.array([1])
                else:
                    # this is the size, like [5,5]
                    s = [int(n) for n in m.groups()]
                    # this is the value, which is always a progression
                    # from 0...prod(s)-1, reshaped to be the size asked for
                    a_value = np.arange(np.prod(s)).reshape(s)
                return lambda lam, WA: a_value * unit
        else:
            assert False, "unknown interpolant needed"

    def compare_interpolants(self, f1, f2, param, msg=""):
        r"""Compare two interpolants f1 and f2 by probing them randomly."""
        # Find out the number of input arguments expected
        f1_info = inspect.getargspec(f1)
        f2_info = inspect.getargspec(f2)
        # this is the number of formal arguments MINUS the number of defaults
        # (EXOSIMS uses defaults to provide functional closure, though it's unneeded)
        nargin1 = len(f1_info.args) - (
            0 if not f1_info.defaults else len(f1_info.defaults)
        )
        nargin2 = len(f2_info.args) - (
            0 if not f2_info.defaults else len(f2_info.defaults)
        )
        if nargin1 != nargin2:
            raise self.failureException(
                msg + "-- functions have different arity (arg lengths)"
            )
        # make a few random probes of the interpolant on the interval (0,1)
        for count in range(10):
            # obtain a vector of length nargin1
            arg_in = np.random.random(nargin1)
            # the result can be a float (for contrast),
            # a numpy array (for PSF), or a Quantity (for QE)
            if param in ("core_thruput", "core_contrast"):
                out_1 = f1(arg_in[0] * u.nm, arg_in[1] * u.arcsec)
            else:
                out_1 = f1(*arg_in)
            out_2 = f2(*arg_in)
            diff = out_1 - out_2
            # if it's a quantity, unbox the difference
            if isinstance(diff, u.quantity.Quantity):
                diff = diff.value
            if np.any(np.abs(diff) > 1e-5):
                errmsg = msg + "-- function mismatch: %r != %r" % (out_1, out_2)
                raise self.failureException(errmsg)

    def compare_lists(self, list1, list2, msg=""):
        r"""Compare two lists-of-dicts list1 and list2 to ensure list2 attributes are in f1."""
        if len(list1) != len(list2):
            raise self.failureException(
                msg + " -- list length mismatch: %d vs %d" % (len(list1), len(list2))
            )
        for d1, d2 in zip(list1, list2):
            if type(d1) != type(d2):
                raise self.failureException(
                    msg + " -- type mismatch: %d vs %d" % (type(d1), type(d2))
                )
            assert isinstance(d2, dict), (
                msg + " -- compare_lists expects lists-of-dicts"
            )
            # note: we need d2 to be a subset of d1
            for k in d2:
                self.assertEqual(d1[k], d2[k], msg + " -- key %s mismatch" % k)

    @unittest.skip("All of these need to be tested separately")
    def test_init_sweep_inputs(self):
        r"""Test __init__ method, sweeping over all parameters.

        Method: Driven by the table at the top of this file, we sweep
        through all input parameters to OpticalSystem, and in each case,
        we instantiate an OpticalSystem with the given parameter, and
        check that the parameter is properly set.  Additionally, for
        interpolants, we check that the proper function is obtained
        by EXOSIMS.  Additionally, also for file-based interpolants, we
        ensure that appropriate exceptions are raised for inaccessible
        files and for out-of-range values."""

        # iterate over all tests
        for (param, recipe) in opsys_params.iteritems():
            print(param)
            # one inner loop tests that we can set 'param'
            #   example simple param dictionary:
            #   pupilDiam = dict(default=4.0, trial=(1.0, 10.0), unit=u.m)
            # set up "raises" key, if not given
            if "raises" not in recipe:
                recipe["raises"] = (None,) * len(recipe["trial"])
            assert len(recipe["raises"]) == len(recipe["trial"]), (
                "recipe length mismatch for " + param
            )
            # target dictionary: 0 for top-level, 1 for inst, 2 for starlight
            target = recipe["target"]
            # the unit is multiplied by the numerical value when comparing
            unit = 1.0 if recipe["unit"] is float else recipe["unit"]
            # loop over all trials requested for "param"
            for (param_val, err_val) in zip(recipe["trial"], recipe["raises"]):
                # need a copy because we are about to alter the specs
                specs = deepcopy(specs_simple)
                # print param, '<--', param_val, '[', err_val, ']'
                # make strings into full filenames
                if isinstance(param_val, str):
                    param_val = os.path.join(
                        resource_path(), "OpticalSystem", param_val
                    )
                # insert param_val into approriate slot within specs - a bit messy
                if target == 0:
                    specs[param] = param_val
                elif target == 1:
                    specs["scienceInstruments"][0][param] = param_val
                elif target == 2:
                    specs["starlightSuppressionSystems"][0][param] = param_val
                else:
                    # this is a failure of the "recipe" dictionary entry
                    assert False, "Target must be 0, 1, 2 -- " + param
                # prepare to get an OpticalSystem
                if err_val is None:
                    # no error expected in this branch
                    # note: deepcopy specs because EXOSIMS alters them,
                    # which would in turn alter our reference param_val
                    # if it's mutable!
                    optsys = self.fixture(**deepcopy(specs))
                    # find the right place to seek the param we tried
                    if target == 0:
                        d = optsys.__dict__
                    elif target == 1:
                        d = optsys.__dict__["scienceInstruments"][0]
                    elif target == 2:
                        d = optsys.__dict__["starlightSuppressionSystems"][0]
                    # check the value: scalar or interpolant
                    if hasattr(d[param], "__call__"):
                        # it's an interpolant -- make an interpolant ourselves
                        param_int = self.make_interpolant(param, param_val, unit)
                        # compare our interpolant with the EXOSIMS interpolant
                        self.compare_interpolants(d[param], param_int, param)
                    elif isinstance(d[param], list):
                        # a small list-of-dict's in the case of the
                        # 'scienceInstruments' and 'starlightSuppressionSystems' checks.
                        # this checks that param_val is a subset of d[param]
                        self.compare_lists(d[param], param_val, param)
                    else:
                        # a number or Quantity - simple assertion
                        if isinstance(param_val, list):
                            print("***", param)
                            print(d[param])
                        self.assertEqual(
                            d[param],
                            param_val * unit,
                            msg='failed to set "%s" parameter' % param,
                        )
                else:
                    # else, ensure the right error is raised
                    with self.assertRaises(err_val):
                        self.fixture(**deepcopy(specs))


if __name__ == "__main__":
    unittest.main()
