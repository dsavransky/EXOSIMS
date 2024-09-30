import unittest
from EXOSIMS.Observatory.KulikStarshade import KulikStarshade as KS
import EXOSIMS.TargetList.KnownRVPlanetsTargetList as KnownRVPlanetsTargetList
import numpy as np 
import EXOSIMS.Prototypes.TargetList as TargetList


class TestKulikStarshade(unittest.TestCase):
    """
    Maxwell Zweig, August 2024, Cornell

    This class tests helper functinos of KulikStarshade as well as instantiation of Kulik Starshade.
    This class also tests some basic sanity inputs that should always be correct. 
    """

    def test_initialization(self):
        """makes sure that star shade initialization is behaving as expected."""
        # Verify that ValueError is raised when dividing by zero
        with self.assertRaises(Exception) as context:
            KS(mode="energyOptimals", dynamics=0, exponent=8, precompfname="EXOSIMS/Observatory/haloImpulsive", starShadeRadius=10)
        self.assertEqual(str(context.exception), 'Mode must be one of "energyOptimal" or "impuslive"')

        try: 
            KS(mode="energyOptimal", dynamics=0, exponent=8, precompfname="EXOSIMS/Observatory/haloEnergy", starShadeRadius=10)
        except Exception as e:
            self.fail(f"Instantiation raised an exception: {e}")

        
    def test_inert_to_rel(self):
        pass


    def test_time_to_can(self):
        pass 



    def test_dV(self):

        starshade = KS(mode="energyOptimal", dynamics=0, exponent=8, precompfname="EXOSIMS/Observatory/haloEnergy", starShadeRadius = 10)
        targets = TargetList.TargetList(modules= {"StarCatalog": "EXOCAT1", "OpticalSystem": " ", "ZodiacalLight": "  ", "PostProcessing": " ", "Completeness": " ", "PlanetPopulation" : "KeplerLike1", "BackgroundSources" : " ", "PlanetPhysicalModel" : " "})
        dV = starshade.calculate_dV(targets, 5, np.array([13, 14, 17, 20]), np.ones((4, 50)) + np.random.rand(4,50), starshade.equinox)
        self.assertTrue(np.all(dV >= 0))
        self.assertTrue(dV.shape == np.ones((4, 50)).shape)

        starshade = KS(mode="energyOptimal", dynamics=0, exponent=8, precompfname="EXOSIMS/Observatory/haloEnergy", starShadeRadius = 10)
        targets = TargetList.TargetList(modules= {"StarCatalog": "EXOCAT1", "OpticalSystem": " ", "ZodiacalLight": "  ", "PostProcessing": " ", "Completeness": " ", "PlanetPopulation" : "KeplerLike1", "BackgroundSources" : " ", "PlanetPhysicalModel" : " "})
        dV = starshade.calculate_dV(targets, 5, np.array([13]), np.ones((1, 50)) + np.random.rand(1,50), starshade.equinox)
        self.assertTrue(np.all(dV >= 0))
        self.assertTrue(dV.shape == np.ones((1, 50)).shape)

        starshade = KS(mode="impulsive", dynamics=0, exponent=8, precompfname="EXOSIMS/Observatory/haloImpulsive", starShadeRadius = 10)
        targets = TargetList.TargetList(modules= {"StarCatalog": "EXOCAT1", "OpticalSystem": " ", "ZodiacalLight": "  ", "PostProcessing": " ", "Completeness": " ", "PlanetPopulation" : "KeplerLike1", "BackgroundSources" : " ", "PlanetPhysicalModel" : " "})
        dV = starshade.calculate_dV(targets, 5, np.array([13, 14, 17, 20]), np.ones((4, 50)) + np.random.rand(4,50), starshade.equinox)
        self.assertTrue(np.all(dV >= 0))
        self.assertTrue(dV.shape == np.ones((4, 50)).shape)

        starshade = KS(mode="impulsive", dynamics=0, exponent=8, precompfname="EXOSIMS/Observatory/haloImpulsive", starShadeRadius = 10)
        targets = TargetList.TargetList(modules= {"StarCatalog": "EXOCAT1", "OpticalSystem": " ", "ZodiacalLight": "  ", "PostProcessing": " ", "Completeness": " ", "PlanetPopulation" : "KeplerLike1", "BackgroundSources" : " ", "PlanetPhysicalModel" : " "})
        dV = starshade.calculate_dV(targets, 5, np.array([13]), np.ones((1, 50)) + np.random.rand(1,50), starshade.equinox)
        self.assertTrue(np.all(dV >= 0))
        self.assertTrue(dV.shape == np.ones((1, 50)).shape)





        

