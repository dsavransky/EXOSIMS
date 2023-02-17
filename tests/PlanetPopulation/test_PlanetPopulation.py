import unittest
import EXOSIMS
import EXOSIMS.Prototypes.PlanetPopulation
import EXOSIMS.PlanetPopulation
import pkgutil
from EXOSIMS.util.get_module import get_module
import numpy as np
import os
from tests.TestSupport.Utilities import RedirectStreams
import sys
from io import StringIO


class TestPlanetPopulation(unittest.TestCase):
    """

    Global PlanetPopulation tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """

    def setUp(self):

        self.dev_null = open(os.devnull, "w")

        self.spec = {"modules": {"PlanetPhysicalModel": "PlanetPhysicalModel"}}

        modtype = getattr(
            EXOSIMS.Prototypes.PlanetPopulation.PlanetPopulation, "_modtype"
        )
        pkg = EXOSIMS.PlanetPopulation
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + "."
        ):
            if not is_pkg:
                mod = get_module(module_name.split(".")[-1], modtype)
                self.assertTrue(
                    mod._modtype is modtype, "_modtype mismatch for %s" % mod.__name__
                )
                self.allmods.append(mod)

    def tearDown(self):
        self.dev_null.close()

    def test_honor_arange(self):
        """
        Tests that the input range for semi-major axis is properly set.
        """

        exclude_setrange = [
            "EarthTwinHabZone1",
            "EarthTwinHabZone2",
            "JupiterTwin",
            "AlbedoByRadiusDulzPlavchan",
            "DulzPlavchan",
            "EarthTwinHabZone1SDET",
            "EarthTwinHabZone3",
            "EarthTwinHabZoneSDET",
            "Brown2005EarthLike",
        ]

        arangein = np.sort(np.random.rand(2) * 10.0)

        for mod in self.allmods:
            if mod.__name__ not in exclude_setrange:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(arange=arangein, **self.spec)

                # test that the correct input arange is used
                self.assertTrue(
                    obj.arange[0].value == arangein[0],
                    "sma low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.arange[1].value == arangein[1],
                    "sma high bound set failed for %s" % mod.__name__,
                )

                # test that an incorrect input arange is corrected
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(arange=arangein[::-1], **self.spec)
                self.assertTrue(
                    obj.arange[0].value == arangein.min(),
                    "sma low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.arange[1].value == arangein.max(),
                    "sma high bound set failed for %s" % mod.__name__,
                )

    def test_honor_erange(self):
        """
        Tests that the input range for eccentricity is properly set.
        """

        exclude_setrange = [
            "EarthTwinHabZone1",
            "EarthTwinHabZone1SDET",
            "EarthTwinHabZone3",
            "EarthTwinHabZoneSDET",
        ]

        tmp = np.random.rand(1) * 0.5
        erangein = np.hstack((tmp, np.random.rand(1) * 0.5 + 0.5))

        for mod in self.allmods:
            if mod.__name__ not in exclude_setrange:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(erange=erangein, **self.spec)

                # test that the correct input erange is used
                self.assertTrue(
                    obj.erange[0] == erangein[0],
                    "e low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.erange[1] == erangein[1],
                    "e high bound set failed for %s" % mod.__name__,
                )

                # test that an incorrect input erange is corrected
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(erange=erangein[::-1], **self.spec)
                self.assertTrue(
                    obj.erange[0] == erangein.min(),
                    "e low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.erange[1] == erangein.max(),
                    "e high bound set failed for %s" % mod.__name__,
                )

    def test_honor_constrainOrbits(self):
        """
        Test that constrainOrbits is consistently applied

        Generated orbital radii must be within the rrange, which
        is the original arange.
        """

        exclude_check = ["Guimond2019"]

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(constrainOrbits=True, **self.spec)
            self.assertTrue(
                obj.constrainOrbits, "constrainOrbits not set for %s" % mod.__name__
            )
            self.assertTrue(
                np.all(obj.arange == obj.rrange),
                "arange and rrange do not match with constrainOrbits set for %s"
                % mod.__name__,
            )

            # ignore any cases where gen_plan_params is inherited
            if (mod.__name__ not in exclude_check) and (
                "gen_plan_params" in mod.__dict__
            ):
                x = 10000
                a, e, p, Rp = obj.gen_plan_params(x)
                rp = a * (1.0 + e)
                rm = a * (1.0 - e)
                high = (obj.rrange[1] - rp).to("AU").value
                low = (rm - obj.rrange[0]).to("AU").value
                self.assertTrue(
                    np.all(high > -1e-12),
                    "constrainOrbits high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(low > -1e-12),
                    "constrainOrbits low bound failed for %s" % mod.__name__,
                )

    def test_honor_prange(self):
        """
        Tests that the input range for albedo is properly set.
        """

        exclude_setrange = [
            "EarthTwinHabZone1",
            "EarthTwinHabZone2",
            "JupiterTwin",
            "AlbedoByRadius",
            "AlbedoByRadiusDulzPlavchan",
            "EarthTwinHabZone1SDET",
            "EarthTwinHabZone3",
            "EarthTwinHabZoneSDET",
            "Brown2005EarthLike",
        ]

        tmp = np.random.rand(1) * 0.5
        prangein = np.hstack((tmp, np.random.rand(1) * 0.5 + 0.5))

        for mod in self.allmods:
            if mod.__name__ not in exclude_setrange:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(prange=prangein, **self.spec)

                # test that the correct input prange is used
                self.assertTrue(
                    obj.prange[0] == prangein[0],
                    "p low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.prange[1] == prangein[1],
                    "p high bound set failed for %s" % mod.__name__,
                )

                # test that incorrect input prange is corrected
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(prange=prangein[::-1], **self.spec)
                self.assertTrue(
                    obj.prange[0] == prangein.min(),
                    "p low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.prange[1] == prangein.max(),
                    "p high bound set failed for %s" % mod.__name__,
                )

    def test_honor_Rprange(self):
        """
        Tests that the input range for planet radius is properly set
        and is used when generating radius samples.
        """

        exclude_setrange = [
            "EarthTwinHabZone1",
            "EarthTwinHabZone2",
            "JupiterTwin",
            "AlbedoByRadiusDulzPlavchan",
            "DulzPlavchan",
            "EarthTwinHabZone1SDET",
            "EarthTwinHabZone3",
            "EarthTwinHabZoneSDET",
            "Brown2005EarthLike",
            "Guimond2019",
        ]

        Rprangein = np.sort(np.random.rand(2) * 10.0)

        for mod in self.allmods:
            if mod.__name__ not in exclude_setrange:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(Rprange=Rprangein, **self.spec)

                # test that the correct input Rprange is used
                self.assertTrue(
                    obj.Rprange[0].value == Rprangein[0],
                    "Rp low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.Rprange[1].value == Rprangein[1],
                    "Rp high bound set failed for %s" % mod.__name__,
                )

                # test that incorrect input Rprange is corrected
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(Rprange=Rprangein[::-1], **self.spec)
                self.assertTrue(
                    obj.Rprange[0].value == Rprangein.min(),
                    "Rp low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.Rprange[1].value == Rprangein.max(),
                    "Rp high bound set failed for %s" % mod.__name__,
                )

    def test_honor_Mprange(self):
        """
        Tests that the input range for planet mass is properly set
        and is used when generating mass samples.
        """

        exclude_setrange = [
            "EarthTwinHabZone1",
            "EarthTwinHabZone2",
            "JupiterTwin",
            "AlbedoByRadiusDulzPlavchan",
            "DulzPlavchan",
            "EarthTwinHabZone1SDET",
            "EarthTwinHabZone3",
            "EarthTwinHabZoneSDET",
            "Brown2005EarthLike",
        ]
        exclude_checkrange = ["KeplerLike1"]

        Mprangein = np.sort(np.random.rand(2) * 10.0)

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(Mprange=Mprangein, **self.spec)

            # test that the input Mprange is used
            if mod.__name__ not in exclude_setrange:
                self.assertTrue(
                    obj.Mprange[0].value == Mprangein[0],
                    "Mp low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.Mprange[1].value == Mprangein[1],
                    "Mp high bound set failed for %s" % mod.__name__,
                )

            # test that incorrect input Mprange is corrected
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(Mprange=Mprangein[::-1], **self.spec)
            if mod.__name__ not in exclude_setrange:
                self.assertTrue(
                    obj.Mprange[0].value == Mprangein.min(),
                    "Mp low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.Mprange[1].value == Mprangein.max(),
                    "Mp high bound set failed for %s" % mod.__name__,
                )

            # test that generated values honor range
            if (mod.__name__ not in exclude_checkrange) and (
                "gen_mass" in mod.__dict__
            ):
                x = 10000
                Mp = obj.gen_mass(x)
                self.assertEqual(
                    len(Mp),
                    x,
                    "Incorrect number of samples generated for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(Mp.value <= obj.Mprange[1].value),
                    "Mp high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(Mp.value >= obj.Mprange[0].value),
                    "Mp low bound failed for %s" % mod.__name__,
                )

    def test_honor_wrange(self):
        """
        Tests that the input range for arg or periapse is properly set.
        """

        exclude_setrange = []
        exclude_checkrange = []

        tmp = np.random.rand(1) * 180
        wrangein = np.hstack((tmp, np.random.rand(1) * 180 + 180))

        for mod in self.allmods:
            if (
                (mod.__name__ not in exclude_setrange)
                and (mod.__name__ not in exclude_checkrange)
                and ("gen_angles" in mod.__dict__)
            ):
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(wrange=wrangein, **self.spec)

                # test that the correct input wrange is used
                self.assertTrue(
                    obj.wrange[0].value == wrangein[0],
                    "w low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.wrange[1].value == wrangein[1],
                    "w high bound set failed for %s" % mod.__name__,
                )

                # test that incorrect input wrange is corrected
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(wrange=wrangein[::-1], **self.spec)
                self.assertTrue(
                    obj.wrange[0].value == wrangein.min(),
                    "w low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.wrange[1].value == wrangein.max(),
                    "w high bound set failed for %s" % mod.__name__,
                )

    def test_honor_Irange(self):
        """
        Tests that the input range for inclination is properly set.
        """

        exclude_setrange = []
        exclude_checkrange = []

        tmp = np.random.rand(1) * 90
        Irangein = np.hstack((tmp, np.random.rand(1) * 90 + 90))

        for mod in self.allmods:
            if (
                (mod.__name__ not in exclude_setrange)
                and (mod.__name__ not in exclude_checkrange)
                and ("gen_angles" in mod.__dict__)
            ):
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(Irange=Irangein, **self.spec)

                # test that the correct input Irange is used
                self.assertTrue(
                    obj.Irange[0].value == Irangein[0],
                    "I low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.Irange[1].value == Irangein[1],
                    "I high bound set failed for %s" % mod.__name__,
                )

                # test that incorrect input Irange is corrected
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(Irange=Irangein[::-1], **self.spec)
                self.assertTrue(
                    obj.Irange[0].value == Irangein.min(),
                    "I low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.Irange[1].value == Irangein.max(),
                    "I high bound set failed for %s" % mod.__name__,
                )

    def test_honor_Orange(self):
        """
        Tests that the input range for long. of ascending node is properly set.
        """

        exclude_setrange = []
        exclude_checkrange = []

        tmp = np.random.rand(1) * 180
        Orangein = np.hstack((tmp, np.random.rand(1) * 180 + 180))

        for mod in self.allmods:
            if (
                (mod.__name__ not in exclude_setrange)
                and (mod.__name__ not in exclude_checkrange)
                and ("gen_angles" in mod.__dict__)
            ):
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(Orange=Orangein, **self.spec)

                # test that the correct input Orange is used
                self.assertTrue(
                    obj.Orange[0].value == Orangein[0],
                    "O low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.Orange[1].value == Orangein[1],
                    "O high bound set failed for %s" % mod.__name__,
                )

                # test that incorrect input Orange is corrected
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(Orange=Orangein[::-1], **self.spec)
                self.assertTrue(
                    obj.Orange[0].value == Orangein.min(),
                    "O low bound set failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    obj.Orange[1].value == Orangein.max(),
                    "O high bound set failed for %s" % mod.__name__,
                )

    def test_gen_plan_params(self):
        """
        Test that gen_plan_params returns the correct number of samples
        and values are within the min and max values specified.
        """

        exclude_setrange = ["EarthTwinHabZone1SDET", "EarthTwinHabZoneSDET"]

        # number of samples to generate
        num = 10000
        for mod in self.allmods:
            if mod.__name__ not in exclude_setrange:
                if "gen_plan_params" in mod.__dict__:
                    with RedirectStreams(stdout=self.dev_null):
                        obj = mod(**self.spec)
                    a, e, p, Rp = obj.gen_plan_params(num)
                    # check each sampled parameter
                    self.assertEqual(
                        len(a),
                        num,
                        "Incorrect number of samples generated for %s" % mod.__name__,
                    )
                    self.assertTrue(
                        np.all(a <= obj.arange[1]),
                        "a high bound failed for %s" % mod.__name__,
                    )
                    self.assertTrue(
                        np.all(a >= obj.arange[0]),
                        "a low bound failed for %s" % mod.__name__,
                    )
                    self.assertEqual(
                        len(e),
                        num,
                        "Incorrect number of samples generated for %s" % mod.__name__,
                    )
                    self.assertTrue(
                        np.all(e <= obj.erange[1]),
                        "e high bound failed for %s" % mod.__name__,
                    )
                    self.assertTrue(
                        np.all(e >= obj.erange[0]),
                        "e low bound failed for %s" % mod.__name__,
                    )
                    self.assertEqual(
                        len(p),
                        num,
                        "Incorrect number of samples generated for %s" % mod.__name__,
                    )
                    self.assertTrue(
                        np.all(p <= obj.prange[1]),
                        "p high bound failed for %s" % mod.__name__,
                    )
                    self.assertTrue(
                        np.all(p >= obj.prange[0]),
                        "p low bound failed for %s" % mod.__name__,
                    )
                    self.assertEqual(
                        len(Rp),
                        num,
                        "Incorrect number of samples generated for %s" % mod.__name__,
                    )
                    self.assertTrue(
                        np.all(Rp <= obj.Rprange[1]),
                        "Rp high bound failed for %s" % mod.__name__,
                    )
                    self.assertTrue(
                        np.all(Rp >= obj.Rprange[0]),
                        "Rp low bound failed for %s" % mod.__name__,
                    )

    def test_gen_angles(self):
        """
        Test that  gen_angles returns the correct number of samples
        and values are within the min and max specified.
        """

        # number of samples to generate
        num = 10000
        for mod in self.allmods:
            if "gen_angles" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(**self.spec)
                I, O, w = obj.gen_angles(num)
                self.assertEqual(
                    len(I),
                    num,
                    "Incorrect number of samples generated for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(I <= obj.Irange[1]),
                    "I high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(I >= obj.Irange[0]),
                    "I low bound failed for %s" % mod.__name__,
                )
                self.assertEqual(
                    len(O),
                    num,
                    "Incorrect number of samples generated for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(O <= obj.Orange[1]),
                    "O high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(O >= obj.Orange[0]),
                    "O low bound failed for %s" % mod.__name__,
                )
                self.assertEqual(
                    len(w),
                    num,
                    "Incorrect number of samples generated for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(w <= obj.wrange[1]),
                    "w high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(w >= obj.wrange[0]),
                    "w low bound failed for %s" % mod.__name__,
                )

    def test_dist_eccen_from_sma(self):
        """
        Test that eccentricities generating radii outside of arange
        have zero probability.
        """

        for mod in self.allmods:
            if "dist_eccen_from_sma" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(**self.spec)

                x = 10000
                a, e, p, Rp = obj.gen_plan_params(x)

                f = obj.dist_eccen_from_sma(e, a)
                self.assertTrue(
                    np.all(f[a * (1 + e) > obj.rrange[1]] == 0),
                    "dist_eccen_from_sma low bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(f[a * (1 - e) < obj.rrange[0]] == 0),
                    "dist_eccen_from_sma high bound failed for %s" % mod.__name__,
                )

    def test_dist_sma(self):
        """
        Test that smas outside of the range have zero probability

        """

        for mod in self.allmods:
            if "dist_sma" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    pp = mod(**self.spec)

                a = np.logspace(
                    np.log10(pp.arange[0].to("AU").value / 10.0),
                    np.log10(pp.arange[1].to("AU").value * 10.0),
                    100,
                )

                fa = pp.dist_sma(a)
                self.assertTrue(
                    np.all(fa[a < pp.arange[0].to("AU").value] == 0),
                    "dist_sma high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(fa[a > pp.arange[1].to("AU").value] == 0),
                    "dist_sma low bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(
                        fa[
                            (a >= pp.arange[0].to("AU").value)
                            & (a <= pp.arange[1].to("AU").value)
                        ]
                        >= 0.0
                    ),
                    "dist_sma generates negative densities within range for %s"
                    % mod.__name__,
                )

    def test_dist_eccen(self):
        """
        Test that eccentricities outside of the range have zero probability

        """
        for mod in self.allmods:
            if "dist_eccen" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    pp = mod(**self.spec)

                e = np.linspace(pp.erange[0] - 1, pp.erange[1] + 1, 100)

                fe = pp.dist_eccen(e)
                self.assertTrue(
                    np.all(fe[e < pp.erange[0]] == 0),
                    "dist_eccen high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(fe[e > pp.erange[1]] == 0),
                    "dist_eccen low bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(fe[(e >= pp.erange[0]) & (e <= pp.erange[1])] > 0),
                    "dist_eccen generates zero probabilities within range for %s"
                    % mod.__name__,
                )

    def test_dist_albedo(self):
        """
        Test that albedos outside of the range have zero probability

        """

        exclude_mods = ["KeplerLike1", "AlbedoByRadiusDulzPlavchan", "DulzPlavchan"]
        for mod in self.allmods:
            if (mod.__name__ not in exclude_mods) and ("dist_albedo" in mod.__dict__):
                with RedirectStreams(stdout=self.dev_null):
                    pp = mod(**self.spec)

                p = np.linspace(pp.prange[0] - 1, pp.prange[1] + 1, 100)

                fp = pp.dist_albedo(p)
                self.assertTrue(
                    np.all(fp[p < pp.prange[0]] == 0),
                    "dist_albedo high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(fp[p > pp.prange[1]] == 0),
                    "dist_albedo low bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(fp[(p >= pp.prange[0]) & (p <= pp.prange[1])] > 0),
                    "dist_albedo generates zero probabilities within range for %s"
                    % mod.__name__,
                )

    def test_dist_radius(self):
        """
        Test that radii outside of the range have zero probability

        """
        for mod in self.allmods:
            if "dist_radius" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    pp = mod(**self.spec)

                Rp = np.logspace(
                    np.log10(pp.Rprange[0].to("earthRad").value / 10.0),
                    np.log10(pp.Rprange[1].to("earthRad").value * 100.0),
                    100,
                )

                fr = pp.dist_radius(Rp)
                self.assertTrue(
                    np.all(fr[Rp < pp.Rprange[0].to("earthRad").value] == 0),
                    "dist_radius high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(fr[Rp > pp.Rprange[1].to("earthRad").value] == 0),
                    "dist_radius high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(
                        fr[
                            (Rp >= pp.Rprange[0].to("earthRad").value)
                            & (Rp <= pp.Rprange[1].to("earthRad").value)
                        ]
                        > 0
                    ),
                    "dist_radius generates zero probabilities within range for %s"
                    % mod.__name__,
                )

    def test_dist_mass(self):
        """
        Test that masses outside of the range have zero probability

        """

        for mod in self.allmods:
            if "dist_mass" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    pp = mod(**self.spec)

                Mp = np.logspace(
                    np.log10(pp.Mprange[0].value / 10.0),
                    np.log10(pp.Mprange[1].value * 100.0),
                    100,
                )

                fr = pp.dist_mass(Mp)
                self.assertTrue(
                    np.all(fr[Mp < pp.Mprange[0].value] == 0),
                    "dist_mass high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(fr[Mp > pp.Mprange[1].value] == 0),
                    "dist_mass high bound failed for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(
                        fr[(Mp >= pp.Mprange[0].value) & (Mp <= pp.Mprange[1].value)]
                        > 0
                    )
                )

    def test_dist_sma_radius(self):
        """
        Test that sma and radius values outside of the range have zero probability
        """

        for mod in self.allmods:
            if "dist_sma_radius" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    pp = mod(**self.spec)

                a = np.logspace(
                    np.log10(pp.arange[0].value / 10.0),
                    np.log10(pp.arange[1].value * 100),
                    100,
                )
                Rp = np.logspace(
                    np.log10(pp.Rprange[0].value / 10.0),
                    np.log10(pp.Rprange[1].value * 100),
                    100,
                )

                aa, RR = np.meshgrid(a, Rp)

                fr = pp.dist_sma_radius(aa, RR)
                self.assertTrue(
                    np.all(fr[aa < pp.arange[0].value] == 0),
                    "dist_sma_radius low bound failed on sma for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(fr[aa > pp.arange[1].value] == 0),
                    "dist_sma_radius high bound failed on sma for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(fr[RR < pp.Rprange[0].value] == 0),
                    "dist_sma_radius low bound failed on radius for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(fr[RR > pp.Rprange[1].value] == 0),
                    "dist_sma_radius high bound failed on radius for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(
                        fr[
                            (aa > pp.arange[0].value)
                            & (aa < pp.arange[1].value)
                            & (RR > pp.Rprange[0].value)
                            & (RR < pp.Rprange[1].value)
                        ]
                        > 0
                    ),
                    "dist_sma_radius is improper pdf for %s" % mod.__name__,
                )

    def test_str(self):
        """
        Test __str__ method, for full coverage and check that all modules have
        required attributes.
        """
        atts_list = [
            "PlanetPhysicalModel",
            "arange",
            "erange",
            "Irange",
            "Orange",
            "wrange",
            "prange",
            "Rprange",
            "Mprange",
            "rrange",
            "scaleOrbits",
            "constrainOrbits",
            "eta",
        ]
        for mod in self.allmods:
            if "__str__" not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**self.spec)
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            # call __str__ method
            result = obj.__str__()
            # examine what was printed
            contents = sys.stdout.getvalue()
            self.assertEqual(type(contents), type(""))
            # attributes from ICD
            for att in atts_list:
                self.assertIn(
                    att, contents, "{} missing for {}".format(att, mod.__name__)
                )
            sys.stdout.close()
            # it also returns a string, which is not necessary
            self.assertEqual(type(result), type(""))
            # put stdout back
            sys.stdout = original_stdout
