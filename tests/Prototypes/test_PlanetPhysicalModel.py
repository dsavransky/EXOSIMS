#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>
# You may need to put EXOSIMS in your $PYTHONPATH, e.g.,
#   % PYTHONPATH=/path/to/exomissionsim <this_file.py>

r"""PlanetPhysicalModel module unit tests

Michael Turmon, JPL, Apr. 2016
"""

import unittest

# Generic test suite for PlanetPhysicalModel
from tests.PlanetPhysicalModel.TestPlanetPhysicalModel import TestPlanetPhysicalModelMixin

# the specific PlanetPhysicalModel class we test here
from EXOSIMS.Prototypes.PlanetPhysicalModel import PlanetPhysicalModel


class TestPlanetPhysicalModelPrototype(TestPlanetPhysicalModelMixin,
                                       unittest.TestCase):
    r"""Test the PlanetPhysicalModel Prototype class."""

    # over-ride to set up the specific test fixture (class constructor) we will use
    planet_model = PlanetPhysicalModel
    # (most default test parameters are OK)
    # tighten this tolerance
    delta_roundtrip = 1e-10


if __name__ == '__main__':
    unittest.main()
