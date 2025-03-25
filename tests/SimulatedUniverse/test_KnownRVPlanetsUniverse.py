#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>
# You may need to put EXOSIMS in your $PYTHONPATH, e.g.,
#   % PYTHONPATH=/path/to/exomissionsim <this_file.py>

r"""KnownRVPlanetsUniverse module unit tests

Michael Turmon, JPL, May 2016
"""

import sys
import os
import unittest
import warnings
import json
from collections import namedtuple
from EXOSIMS.SimulatedUniverse.KnownRVPlanetsUniverse import KnownRVPlanetsUniverse
import numpy as np
import astropy.units as u
from tests.TestSupport.Utilities import RedirectStreams

# A JSON string containing KnownRVPlanets - from simplest-old.json
# The part we require is the "modules" dictionary.
# EXOSIMS also insists on instruments and starlightsuppression
# being present, so they are included.  The particular values
# within do not seem to be used.
# This string could also have been a JSON file, or a literal Python
# dictionary "specs".  We left it as a literal here because it
# makes the test code more self-contained.
ScriptLiteral = """{
  "scienceInstruments": [
    {
      "name": "imaging-EMCCD",
      "type": "imaging-EMCCD",
      "lam": 565,
      "BW": 0.10,
      "QE": 0.88,
      "CIC": 0.0013,
      "sread": 16,
      "ENF": 1.414,
      "Gem": 500
    }
  ],
  "starlightSuppressionSystems": [
    {
      "name": "internal-imaging-HLC",
      "type": "internal-imaging-HLC",
      "IWA": 0.1,
      "OWA": 0,
      "throughput": 1,
      "contrast": 1,
      "PSF": 1
    }
  ],
  "modules": {
    "PlanetPopulation": "KnownRVPlanets",
    "StarCatalog": " ",
    "OpticalSystem": " ",
    "ZodiacalLight": " ",
    "BackgroundSources": " ",
    "PlanetPhysicalModel": "FortneyMarleyCahoyMix1",
    "Observatory": "WFIRSTObservatory",
    "TimeKeeping": " ",
    "PostProcessing": " ",
    "Completeness": "Completeness",
    "TargetList": "KnownRVPlanetsTargetList",
    "SimulatedUniverse": "KnownRVPlanetsUniverse",
    "SurveySimulation": " ",
    "SurveyEnsemble": " "
  }
}"""

# vo-table fields relevant to stars with exoplanets
# See the "atts_mapping" dictionary within the module under test for
# a reference to which fields are included here (pl_letter is
# the only exception).
# Note: the units here must match the units within the TargetList
# attribute.  st_dist is in parsec and is tagged as such (by scaling
# by u.pc) -- because that's how TargetList does it.  Similarly,
# the proper motions (pmra/pmdec) are not tagged with units,
# either here or in the TargetList attribute
exostar_unit_map = dict(
    pl_hostname=str,
    pl_letter=str,
    ra=lambda x: float(x) * u.deg,
    dec=lambda x: float(x) * u.deg,
    st_spstr=str,
    st_plx=float,
    st_uj=float,
    st_bj=float,
    st_vj=float,
    st_rc=float,
    st_ic=float,
    st_j=float,
    st_h=float,
    st_k=float,
    st_dist=lambda x: float(x) * u.pc,
    st_bmvj=float,
    st_lum=lambda x: 10 ** float(x),
    st_pmra=float,
    st_pmdec=float,
    st_radv=float,
)

# Encapsulate constraints on attributes of the Simulated Universe.
# Constraint structure consists of:
#   name -- printable name
#   att -- attribute of the SimulatedUniverse containing it
#   unit -- the astropy unit (None if it does not apply)
#   range -- upper and lower limit, None if it does not apply
attribute_constraint_list = [
    dict(name="sma", att="a", unit=u.AU, range=(0.0, None)),
    dict(name="eccentricity", att="e", unit=None, range=(0.0, 1.0)),
    dict(name="periapsis", att="w", unit=u.deg, range=(None, None)),
    dict(name="long.A.N.", att="O", unit=u.deg, range=(None, None)),
    dict(name="inclination", att="I", unit=u.deg, range=(None, None)),
    dict(name="mass", att="Mp", unit=u.kg, range=(0.0, None)),
    dict(name="radius", att="Rp", unit=u.km, range=(0.0, None)),
    dict(name="albedo", att="p", unit=None, range=(0.0, 1.0)),
    dict(name="position", att="r", unit=u.km, range=(None, None)),
    dict(name="velocity", att="v", unit=u.km / u.s, range=(None, None)),
    dict(name="distance", att="d", unit=u.km, range=(0.0, None)),
    dict(name="separation", att="s", unit=u.km, range=(0.0, None)),
    dict(name="exozodi", att="nEZ", unit=None, range=(0.0, None)),
]

# convenient holder for the above constraints
PlanetInfo = namedtuple("PlanetInfo", ["name", "att", "unit", "range"])
AttributeConstraints = [PlanetInfo(**d) for d in attribute_constraint_list]


class TestKnownRVPlanetsUniverseMethods(unittest.TestCase):
    r"""Test SimulatedUniverse.KnownRVPlanetsUniverse class."""

    def setUp(self):
        self.dev_null = open(os.devnull, "w")
        # print '[setup] ',
        specs = json.loads(ScriptLiteral)
        with RedirectStreams(stdout=self.dev_null):
            with warnings.catch_warnings():
                # filter out warnings about the votable RVplanets file
                warnings.filterwarnings("ignore", ".*votable")
                self.fixture = KnownRVPlanetsUniverse(**specs)

    def tearDown(self):
        self.dev_null.close()
        del self.fixture

    def basic_validation(self, universe):
        r"""Perform basic validation of SimulatedUniverse.

        Factored out into a separate routine to avoid duplication.
        """
        self.assertEqual(universe._modtype, "SimulatedUniverse")
        self.assertEqual(type(universe._outspec), type({}))
        # check for presence of a couple of class attributes
        # self.assertIn('eta', universe.__dict__)
        self.assertIn("nPlans", universe.__dict__)
        self.assertIn("TargetList", universe.__dict__)
        self.assertIn("PlanetPopulation", universe.__dict__)

    # @unittest.skip("Skipping init.")
    def test_init(self):
        r"""Test of initialization and __init__."""
        universe = self.fixture
        self.basic_validation(universe)

    # @unittest.skip("Skipping init.")
    def test_init_attributes(self):
        r"""Test of initialization and __init__ -- object attributes.

        Method: Ensure that the attributes special to KnownRVPlanetsUniverse are
        all present, of right length, have correct units, and are within acceptable
        bounds.
        """
        universe = self.fixture
        self.basic_validation(universe)
        # unittest class variable:
        #   the provided message in self.assert* is in addition to default message
        self.longMessage = True

        # ensure star attributes are present and have right length, units, and range
        #   these attributes are set in populate_target_list
        for constraint in AttributeConstraints:
            # constraint has attributes: 'name', 'att', 'unit', 'range'
            self.assertIn(constraint.att, universe.__dict__)
            # it's there: extract it
            att = universe.__dict__[constraint.att]
            # length
            self.assertEqual(len(att), universe.nPlans)
            # no element is NaN or Infinity -- double negative below is the
            # cleanest way, because some attributes are triples
            self.assertEqual(
                0,
                np.count_nonzero(np.logical_not(np.isfinite(att))),
                "nan/inf in %s" % constraint.name,
            )
            # abbreviations
            c_unit = constraint.unit
            c_scale = c_unit if c_unit is not None else 1.0
            c_range = constraint.range
            # units
            if c_unit is not None:
                self.assertEqual(
                    (att / c_unit).decompose().unit,
                    u.dimensionless_unscaled,
                    "unit conflict in %s" % constraint.name,
                )
            # lower range
            if c_range[0] is not None:
                self.assertEqual(
                    0,
                    np.count_nonzero(att < c_range[0] * c_scale),
                    "lower range violation in %s" % constraint.name,
                )
            # upper range
            if c_range[1] is not None:
                self.assertEqual(
                    0,
                    np.count_nonzero(att > c_range[1] * c_scale),
                    "upper range violation in %s" % constraint.name,
                )

    # @unittest.skip("Skipping init - indexes.")
    def test_init_indexes(self):
        r"""Test of initialization and __init__ -- indexes.

        Method: Insure the plan2star and sInds indexes are present.
        Performs sanity check on the range of index values.
        TODO: More could be done to ensure the index values are correct.
        """
        universe = self.fixture
        self.basic_validation(universe)
        # indexes present
        self.assertIn("plan2star", universe.__dict__)
        self.assertIn("sInds", universe.__dict__)
        # range: 0 <= sInds < nStars
        self.assertEqual(0, np.count_nonzero(universe.sInds < 0))
        self.assertEqual(
            0, np.count_nonzero(universe.sInds >= universe.TargetList.nStars)
        )
        # domain: plan2star covers 0...nPlans-1
        self.assertEqual(len(universe.plan2star), universe.nPlans)
        # range: 0 <= plan2star < nStars
        self.assertEqual(0, np.count_nonzero(universe.plan2star < 0))
        self.assertEqual(
            0, np.count_nonzero(universe.plan2star >= universe.TargetList.nStars)
        )


if __name__ == "__main__":
    unittest.main()
