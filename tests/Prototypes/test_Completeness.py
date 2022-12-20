import unittest
from EXOSIMS.Prototypes import Completeness


class Test_Completeness_Prototype(unittest.TestCase):
    """
    All prototype functionality is covered by the generic test_Complteness.
    The only thing to check for here is the completeness specs behavior.
    """

    def setUp(self):
        self.spec = {"modules": {"PlanetPhysicalModel": " ", "PlanetPopulation": " "}}
        self.fixture = Completeness.Completeness(**self.spec)

        self.nStars = 100

    def tearDown(self):
        pass

    def test_completeness_specs(self):
        spec = {
            "completeness_specs": {},
            "modules": {"PlanetPopulation": " ", "PlanetPhysicalModel": " "},
        }
        Comp = Completeness.Completeness(**spec)

        self.assertTrue(
            Comp.PlanetPopulation.__class__.__name__ == "PlanetPopulation",
            "empty completeness_specs did not load prototype PlanetPopulation",
        )
        self.assertTrue(
            Comp.PlanetPhysicalModel.__class__.__name__ == "PlanetPhysicalModel",
            "empty completeness_specs did not load prototype PlanetPhysicalModel",
        )
