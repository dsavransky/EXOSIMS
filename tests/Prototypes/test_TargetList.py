import unittest
import os
from EXOSIMS.Prototypes import TargetList
import numpy as np
from astropy import units as u
from astropy.time import Time
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
import json
import copy

r"""TargetList module unit tests

Paul Nunez, JPL, Aug. 2016
"""

# scriptfile = resource_path("test-scripts/template_prototype_testing.json")


class Test_TargetList_prototype(unittest.TestCase):
    def setUp(self):
        self.dev_null = open(os.devnull, "w")
        self.script = resource_path("test-scripts/template_minimal.json")
        with open(self.script) as f:
            self.spec = json.loads(f.read())
        self.spec["ntargs"] = 10  # generate fake targets list with 10 stars

    def tearDown(self):
        self.dev_null.close()

    def getTL(self, addkeys=None):
        spec = copy.deepcopy(self.spec)
        if addkeys:
            for key in addkeys:
                spec[key] = addkeys[key]

        # quiet the chatter at initialization
        with RedirectStreams(stdout=self.dev_null):
            self.targetlist = TargetList.TargetList(**spec)
        self.opticalsystem = self.targetlist.OpticalSystem
        self.planetpop = self.targetlist.PlanetPopulation

    def test_nan_filter(self):
        self.getTL()

        # First ensure that application of nan_filter initially does nothing
        n0 = len(self.targetlist.Vmag)
        self.targetlist.nan_filter()
        self.assertEqual(len(self.targetlist.Name), n0)

        # Introduce one nan value and check that it is removed
        self.targetlist.Vmag[2] = float("nan")
        self.targetlist.nan_filter()
        self.assertEqual(len(self.targetlist.Name), n0 - 1)

        # insert another nan for testing
        self.targetlist.dist[0] = float("nan")
        self.targetlist.nan_filter()
        self.assertEqual(len(self.targetlist.Name), n0 - 2)

    def test_binary_filter(self):
        self.getTL()

        n0 = self.targetlist.nStars

        # adding 3 binaries
        self.targetlist.Binary_Cut[1] = True
        self.targetlist.Binary_Cut[3] = True
        self.targetlist.Binary_Cut[5] = True
        self.targetlist.binary_filter()
        n1 = self.targetlist.nStars
        # 3 binaries should be removed
        self.assertEqual(n1, n0 - 3)

    def test_outside_IWA_filter(self):
        self.getTL()

        # initial application of IWA filter should do nothing
        n0 = self.targetlist.nStars
        self.targetlist.outside_IWA_filter()
        self.assertEqual(n0, self.targetlist.nStars)

        # Test filtering (need to have different distances)
        self.targetlist.dist = np.linspace(1, 5, self.targetlist.nStars) * u.pc
        self.opticalsystem.IWA = 0.5 * u.arcsec
        n_expected = len(
            np.where(
                np.tan(0.5 * u.arcsec) * self.targetlist.dist
                < np.max(self.planetpop.rrange)
            )[0]
        )
        self.targetlist.outside_IWA_filter()
        self.assertEqual(self.targetlist.nStars, n_expected)

        # now test with scaleOrbits
        self.planetpop.scaleOrbits = True
        self.targetlist.L[0] = 1e-3
        self.targetlist.outside_IWA_filter()
        self.assertEqual(self.targetlist.nStars, n_expected - 1)

        # Test limiting case where everything would be removed
        self.opticalsystem.IWA = 100 * u.arcsec
        with self.assertRaises(IndexError):
            self.targetlist.outside_IWA_filter()

    def test_vis_mag_filter(self):
        self.getTL()

        # initial application of filter should do nothing
        n0 = self.targetlist.nStars
        self.targetlist.vis_mag_filter(np.inf)
        self.assertEqual(n0, self.targetlist.nStars)

        # now populate different Vmags
        self.targetlist.Vmag[0] = 9
        self.targetlist.Vmag[5] = 10
        self.targetlist.vis_mag_filter(5)
        self.assertEqual(self.targetlist.nStars, n0 - 2)

        # Test limiting case
        with self.assertRaises(IndexError):
            self.targetlist.vis_mag_filter(-1)

    def test_dmag_filter(self):
        self.getTL()

        # test initial null filter
        n0 = self.targetlist.nStars
        self.targetlist.intCutoff_dMag = np.repeat(np.inf, self.targetlist.nStars)
        self.targetlist.max_dmag_filter()
        self.assertEqual(n0, self.targetlist.nStars)

        # Test removing single target
        self.targetlist.intCutoff_dMag[0] = 0
        self.targetlist.max_dmag_filter()
        self.assertEqual(self.targetlist.nStars, n0 - 1)

        # Test limiting case of intCutoff_dMag = 0
        self.targetlist.intCutoff_dMag = np.repeat(0.0, self.targetlist.nStars)
        with self.assertRaises(IndexError):
            self.targetlist.max_dmag_filter()

        # Test limiting case that distance to a star is (effectively) infinite
        # turmon: changed from inf to 1e8 because inf causes a confusing RuntimeWarning
        self.targetlist.intCutoff_dMag = np.repeat(30, self.targetlist.nStars)
        self.planetpop.rrange = np.array([1e8, 1e8]) * u.AU
        with self.assertRaises(IndexError):
            self.targetlist.max_dmag_filter()

    def test_completeness_filter(self):
        self.getTL()

        n0 = self.targetlist.nStars
        self.targetlist.completeness_filter()
        self.assertEqual(self.targetlist.nStars, n0)

        # test removing one target
        self.targetlist.intCutoff_comp[0] = self.targetlist.Completeness.minComp / 2
        self.targetlist.completeness_filter()
        self.assertEqual(self.targetlist.nStars, n0 - 1)

        # Test limiting case of minComp = 1.0
        self.targetlist.Completeness.minComp = 1.0
        with self.assertRaises(IndexError):
            self.targetlist.completeness_filter()

    def test_life_expectancy_filter(self):
        self.getTL()

        # test default removal of BV < 0.3 (hard-coded)
        n0 = self.targetlist.nStars
        self.targetlist.BV = np.repeat(0.5, self.targetlist.nStars)
        self.targetlist.life_expectancy_filter()
        self.assertEqual(n0, self.targetlist.nStars)

        # Test removing single target
        self.targetlist.BV[0] = 0
        self.targetlist.life_expectancy_filter()
        self.assertEqual(self.targetlist.nStars, n0 - 1)

        # test remove all
        self.targetlist.BV = np.repeat(0, self.targetlist.nStars)
        with self.assertRaises(IndexError):
            self.targetlist.life_expectancy_filter()

    def test_main_sequence_filter(self):
        self.getTL()

        # initial application should do nothing
        n0 = self.targetlist.nStars
        self.targetlist.main_sequence_filter()
        self.assertEqual(n0, self.targetlist.nStars)

        # test remove one
        self.targetlist.BV[0] = 10
        self.targetlist.MV[0] = 10
        self.targetlist.main_sequence_filter()
        self.assertEqual(n0 - 1, self.targetlist.nStars)

        # test remove all
        self.targetlist.BV = np.repeat(10, self.targetlist.nStars)
        self.targetlist.MV = np.repeat(10, self.targetlist.nStars)
        with self.assertRaises(IndexError):
            self.targetlist.main_sequence_filter()

    def test_stellar_mass(self):
        self.getTL()

        # Test with absolute magnitue of the sun
        self.targetlist.MV = np.array([4.83])
        self.targetlist.stellar_mass()
        # Should give 1 solar mass approximately
        np.testing.assert_allclose(
            self.targetlist.MsEst[0], 1.05865 * u.solMass, rtol=1e-5, atol=0
        )
        # Relative tolerance is 0.07
        np.testing.assert_allclose(
            self.targetlist.MsTrue[0], 1.05865 * u.solMass, rtol=0.07, atol=0
        )

    def test_fgk_filter(self):
        self.getTL()

        n0 = self.targetlist.nStars
        self.targetlist.fgk_filter()
        self.assertEqual(n0, self.targetlist.nStars)

        # check remove 1
        self.targetlist.Spec[0] = "B0II"
        self.targetlist.fgk_filter()
        self.assertEqual(n0 - 1, self.targetlist.nStars)

    def test_revise_lists(self):
        self.getTL()

        # Check that passing all indices does not change list
        # and that coordinates are in degrees
        i0 = range(len(self.targetlist.Name))
        self.targetlist.revise_lists(i0)
        self.assertEqual(len(i0), len(self.targetlist.Name))
        # Check to see that only 3 elements are retained
        i1 = np.array([1, 5, 8])
        self.targetlist.revise_lists(i1)
        self.assertEqual(len(i1), len(self.targetlist.Name))
        # Check to see that passing no indices yields an emply list
        i2 = []
        with self.assertRaises(IndexError):
            self.targetlist.revise_lists(i2)

    def test_fillPhotometry(self):
        """
        Filling in photometry should result in no nulls in Imag
        """

        self.getTL(addkeys={"fillPhotometry": True, "fillMissingBandMags": True})

        self.assertTrue(self.targetlist.fillPhotometry)
        self.assertTrue(self.targetlist.fillMissingBandMags)

        self.assertTrue(
            np.all(self.targetlist.Imag != 0)
            and np.all(~np.isnan(self.targetlist.Imag))
        )

    def test_completeness_specs(self):
        """
        Test completeness_specs logic
        """
        self.getTL()
        # test case where no completeness specs given
        self.assertEqual(
            self.targetlist.PlanetPopulation.__class__.__name__,
            self.targetlist.Completeness.PlanetPopulation.__class__.__name__,
        )

        # test case where completeness specs given
        self.getTL(
            addkeys={
                "completeness_specs": {
                    "modules": {
                        "PlanetPopulation": "EarthTwinHabZone1",
                        "PlanetPhysicalModel": "PlanetPhysicalModel",
                    }
                }
            }
        )

        self.assertNotEqual(
            self.targetlist.PlanetPopulation.__class__.__name__,
            self.targetlist.Completeness.PlanetPopulation.__class__.__name__,
        )

    def test_starprop(self):
        """
        Test starprop outputs
        """

        self.getTL()
        # setting up 1-dim and multi-dim arrays
        timeRange = np.arange(2000.5, 2019.5, 5)  # 1x4 time array
        # 1x5 time array, same size as sInds later
        timeRangeEql = np.linspace(2000.5, 2019.5, 5)
        # time Quantity arrays
        t_ref = Time(timeRange[0], format="jyear")  # 1x1 time array
        t_refArray = Time(timeRange, format="jyear")  # 1x4 time array
        # 1x5 time array, equal to sInds size
        t_refEqual = Time(timeRangeEql, format="jyear")
        # 1x5 time array, all elements are equal
        t_refCopy = Time(np.tile(timeRange[0], 5), format="jyear")

        # sInd arrays
        sInd = np.array([0])
        sInds = np.array([0, 1, 2, 3, 4])

        # testing Static Stars (set up as a default)
        r_targSSBothSingle = self.targetlist.starprop(sInd, t_ref)  # should be 1x3
        r_targSSMultSinds = self.targetlist.starprop(sInds, t_ref)  # should be 5x3
        r_targSSMultBoth = self.targetlist.starprop(
            sInds, t_refArray
        )  # should be 5x4x3
        r_targSSEqualBoth = self.targetlist.starprop(sInds, t_refEqual)  # should be 5x3
        r_targSSCopyTimes = self.targetlist.starprop(
            sInd, t_refCopy
        )  # should be 1x3 (equal defaults to 1)

        self.assertEqual(r_targSSBothSingle.shape, (1, 3))
        self.assertEqual(r_targSSMultSinds.shape, (sInds.size, 3))
        self.assertEqual(r_targSSMultBoth.shape, (t_refArray.size, sInds.size, 3))
        self.assertEqual(r_targSSEqualBoth.shape, (sInds.size, 3))
        self.assertEqual(r_targSSCopyTimes.shape, (1, 3))

        # testing without Static Stars
        self.targetlist.starprop_static = None
        r_targBothSingle = self.targetlist.starprop(sInd, t_ref)
        r_targMultSinds = self.targetlist.starprop(sInds, t_ref)
        r_targMultTimes = self.targetlist.starprop(sInd, t_refArray)  # should be 5x3
        r_targMultBoth = self.targetlist.starprop(sInds, t_refArray)
        r_targEqualBoth = self.targetlist.starprop(sInds, t_refEqual)
        r_targCopyTimes = self.targetlist.starprop(sInd, t_refCopy)

        self.assertEqual(r_targBothSingle.shape, (1, 3))
        self.assertEqual(r_targMultSinds.shape, (sInds.size, 3))
        self.assertEqual(r_targMultTimes.shape, (t_refArray.size, 3))
        self.assertEqual(r_targMultBoth.shape, (t_refArray.size, sInds.size, 3))
        self.assertEqual(r_targEqualBoth.shape, (sInds.size, 3))
        self.assertEqual(r_targCopyTimes.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
