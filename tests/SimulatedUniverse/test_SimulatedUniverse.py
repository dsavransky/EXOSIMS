import unittest
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
import EXOSIMS.SimulatedUniverse
from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
from EXOSIMS.Prototypes.TargetList import TargetList
from EXOSIMS.util.get_module import get_module
import os
import pkgutil
import numpy as np
import astropy.units as u
import astropy.constants as const
import json
import copy
import sys
from io import StringIO


class TestSimulatedUniverse(unittest.TestCase):
    """

    Global SimulatedUniverse tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """

    def setUp(self):

        self.dev_null = open(os.devnull, "w")
        self.script = resource_path("test-scripts/template_minimal.json")
        with open(self.script) as f:
            self.spec = json.loads(f.read())

        modtype = getattr(SimulatedUniverse, "_modtype")
        pkg = EXOSIMS.SimulatedUniverse
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

    def update_spec(self, mod, addkeys=None):
        """Patch spec for specific module families"""

        spec = copy.deepcopy(self.spec)
        spec["modules"]["PlanetPhysicalModel"] = "FortneyMarleyCahoyMix1"
        # no need for a full catalog - just make sure it has multiple stars
        spec["ntargs"] = 100
        # also set eta to 10 so you ensure that some planets are generated
        spec["eta"] = 10
        spec["modules"]["SimulatedUniverse"] = mod.__name__
        if "Kepler" in mod.__name__:
            spec["modules"]["PlanetPopulation"] = "KeplerLike1"
            spec["scaleOrbits"] = True
        elif "KnownRV" in mod.__name__:
            spec["modules"]["PlanetPopulation"] = "KnownRVPlanets"
            spec["modules"]["TargetList"] = "KnownRVPlanetsTargetList"
        elif "SAG13" in mod.__name__:
            spec["modules"]["PlanetPopulation"] = "SAG13"
            spec["Rprange"] = [1, 10]
            spec["scaleOrbits"] = True
        elif "DulzPlavchan" in mod.__name__:
            spec["modules"]["PlanetPopulation"] = "DulzPlavchan"
        elif "SolarSystem" in mod.__name__:
            spec["modules"]["PlanetPopulation"] = "SolarSystem"

        if addkeys:
            for key in addkeys:
                spec[key] = addkeys[key]

        return spec

    def instantiate_mod(self, mod, addkeys=None):
        """For small sized targetlist, there's a non-negligible chance of producing
        zero planets.  Allow up to 3 redos"""
        obj = None
        for j in range(3):
            try:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(**self.update_spec(mod, addkeys=addkeys))
                break
            except ValueError:
                pass

        assert obj, f"Could not instantiate {mod.__name__} after 3 tries."

        return obj

    def test_init(self):
        """
        Test of initialization and __init__.

        Because some implementations depend on a specific planet population,
        there needs to be additional logic in the setup
        """

        req_atts = [
            "plan2star",
            "a",
            "e",
            "I",
            "O",
            "w",
            "Min",
            "M0",
            "Rp",
            "Mp",
            "p",
            "r",
            "v",
            "d",
            "s",
            "phi",
            "nEZ",
            "dMag",
            "WA",
        ]

        for mod in self.allmods:
            obj = self.instantiate_mod(mod)

            # verify that all attributes are there
            for att in req_atts:
                self.assertTrue(hasattr(obj, att), f"Attribute {att} missing for {mod}")

            # planet properties must all be the same size
            self.assertTrue(
                len(obj.a)
                == len(obj.e)
                == len(obj.I)
                == len(obj.O)
                == len(obj.w)
                == len(obj.M0)
                == len(obj.Rp)
                == len(obj.Mp)
                == len(obj.p)
                == len(obj.d)
                == len(obj.s)
                == len(obj.phi)
                == len(obj.nEZ)
                == len(obj.dMag)
                == len(obj.WA)
                == obj.nPlans,
                "Planet parameters do not have all same lengths in %s" % mod.__name__,
            )

            # r and v must be nx3
            self.assertEqual(
                obj.r.shape,
                (obj.nPlans, 3),
                "r has incorrect shape in %s" % mod.__name__,
            )
            self.assertEqual(
                obj.v.shape,
                (obj.nPlans, 3),
                "v has incorrect shape in %s" % mod.__name__,
            )

            # basic sanity checks
            self.assertTrue(
                np.all(
                    np.linalg.norm(obj.r.to("AU").value, axis=1) == obj.d.to("AU").value
                ),
                "r and d do not match magnitudes in %s" % mod.__name__,
            )
            self.assertTrue(
                np.all(obj.s <= obj.d),
                "Projected separation exceeds orbital radius in %s" % mod.__name__,
            )

            # if module has its own propagator, spin first planet forward by one
            # period and check that it returns to starting position
            if "propag_system" in mod.__dict__:
                sInd = obj.plan2star[0]
                pInds = np.where(obj.plan2star == sInd)[0]
                pInd = pInds[0]

                Ms = obj.TargetList.MsTrue[[sInd]]
                Mp = obj.Mp[pInd]
                mu = (const.G * (Mp + Ms)).to("AU3/day2")
                dt = np.sqrt(4 * np.pi**2.0 * obj.a[pInd] ** 3.0 / mu).to(u.day)

                r0 = obj.r[pInd].copy()
                v0 = obj.v[pInd].copy()

                obj.propag_system(sInd, dt)
                np.testing.assert_allclose(
                    r0,
                    obj.r[pInd],
                    err_msg="propagated r mismatch in %s" % mod.__name__,
                )
                np.testing.assert_allclose(
                    v0,
                    obj.v[pInd],
                    err_msg="propagated r mismatch in %s" % mod.__name__,
                )

    def test_honor_scaleOrbit(self):
        """
        Test that scaleOrbit flag is honored

        Because some implementations depend on a specific planet population,
        there needs to be additional logic in the setup
        """

        whitelist = [
            "KeplerLikeUniverse",
            "KnownRVPlanetsUniverse",
            "SAG13Universe",
            "SolarSystemUniverse",
        ]
        for mod in self.allmods:
            if mod.__name__ in whitelist:
                continue

            obj = self.instantiate_mod(mod, addkeys={"scaleOrbits": True})

            self.assertTrue(
                obj.PlanetPopulation.scaleOrbits,
                "scaleOrbits not set in %s" % mod.__name__,
            )

            aeff = obj.a / np.sqrt(obj.TargetList.L[obj.plan2star])
            self.assertTrue(
                np.all(aeff <= obj.PlanetPopulation.arange[1]),
                "scaled sma out of bounds in %s" % mod.__name__,
            )
            self.assertTrue(
                np.all(aeff >= obj.PlanetPopulation.arange[0]),
                "scaled sma out of bounds in %s" % mod.__name__,
            )

    def test_Honor_fixedPlanPerStar(self):
        """
        Test that fixed PlanPerStar flag passes through integers and None
        """

        # If fixedPlanPerStar is not Defined
        SU = self.instantiate_mod(SimulatedUniverse)
        self.assertTrue(SU.fixedPlanPerStar is None)

        # For 1 star
        SU = self.instantiate_mod(
            SimulatedUniverse, addkeys={"fixedPlanPerStar": 1, "ntargs": 1}
        )
        self.assertTrue(SU.fixedPlanPerStar == 1)
        self.assertTrue(SU.plan2star == np.unique(SU.plan2star))
        self.assertTrue(SU.TargetList.nStars * SU.fixedPlanPerStar == SU.nPlans)
        self.assertTrue(SU.nPlans == 1)  # for this specific test instance

        # For multiple stars and multiple planets per star
        nPlanPerStar = 10
        SU = self.instantiate_mod(
            SimulatedUniverse, addkeys={"fixedPlanPerStar": nPlanPerStar}
        )
        self.assertTrue(SU.fixedPlanPerStar == nPlanPerStar)
        self.assertTrue(SU.nPlans == SU.TargetList.nStars * SU.fixedPlanPerStar)
        self.assertTrue(len(SU.plan2star) == SU.TargetList.nStars * SU.fixedPlanPerStar)

    def test_honor_Min(self):
        """
        Test that gen_M0 assigns constant or random value to mean anomaly

        Because some implementations depend on a specific planet population,
        there needs to be additional logic in the setup
        """

        whitelist = ["KnownRVPlanetsUniverse"]
        for mod in self.allmods:
            if mod.__name__ in whitelist:
                continue

            # Test Min = None first
            obj = self.instantiate_mod(mod)
            if obj.nPlans > 1:
                self.assertTrue(
                    obj.M0[0] != obj.M0[1], "Initial M0 must be randomly set"
                )

            # Test Min = 20
            obj = self.instantiate_mod(mod, addkeys={"Min": 20})

            self.assertTrue(
                np.all(obj.M0.to("deg").value == 20), "Initial M0 must be 20"
            )

    def test_set_planet_phase(self):
        """
        Test that set_planet_phase places planets at the correct phase angle

        Because some implementations depend on a specific planet population,
        there needs to be additional logic in the setup
        """
        whitelist = ["KnownRVPlanetsUniverse"]

        for mod in self.allmods:
            if (mod.__name__ in whitelist) or ("set_planet_phase" not in mod.__dict__):
                continue

            obj = self.instantiate_mod(mod)

            # attempt to set planet phase to pi/4
            obj.set_planet_phase(np.pi / 4.0)
            betas = np.arccos(obj.r[:, 2] / obj.d)
            val1 = np.abs(betas.to("rad").value - np.pi / 4.0)
            val2 = np.abs(betas.to("rad").value - np.pi / 2.0)
            inds1 = np.where(val1 < 1e-4)[0]
            inds2 = np.where(val2 < 1e-4)[0]
            num = len(inds1) + len(inds2)

            self.assertTrue(num == obj.nPlans, "Phase angles not set correctly")

    def test_dump_systems(self):
        """
        Test that dump_systems returns a dictionary with correct keys and values
        """

        # required dictionary keys
        req_keys = [
            "a",
            "e",
            "I",
            "O",
            "w",
            "M0",
            "Mp",
            "mu",
            "Rp",
            "p",
            "plan2star",
            "star",
        ]

        # missing attributes from req_keys
        matts = ["mu", "star"]

        SU = self.instantiate_mod(SimulatedUniverse)

        test_dict = SU.dump_systems()
        for key in req_keys:
            self.assertIn(
                key,
                test_dict,
                "Key %s not in dictionary produced by dump_systems" % key,
            )
            if key not in matts:
                self.assertTrue(
                    np.all(test_dict[key] == getattr(SU, key)),
                    "Value(s) for %s not same produced by dump_systems" % key,
                )

    def test_load_systems(self):
        """
        Test that load systems loads correctly
        """

        SU = self.instantiate_mod(SimulatedUniverse)

        # dictionary of planetary parameter keys
        param_keys = [
            "a",
            "e",
            "I",
            "O",
            "w",
            "M0",
            "Mp",
            "Rp",
            "p",
            "plan2star",
            "nEZ",
        ]
        systems = {
            "a": np.array([5.0]) * u.AU,
            "e": np.array([0.0]),
            "I": np.array([0.0]) * u.deg,
            "O": np.array([0.0]) * u.deg,
            "w": np.array([0.0]) * u.deg,
            "M0": np.array([0.0]) * u.deg,
            "Mp": np.array([300.0]) * u.earthMass,
            "Rp": np.array([10.0]) * u.earthRad,
            "p": np.array([0.6]),
            "plan2star": np.array([0], dtype=int),
            "nEZ": np.array([1.0]),
        }
        SU.load_systems(systems)
        for key in param_keys:
            self.assertTrue(
                systems[key] == getattr(SU, key),
                "Value for %s not assigned with load_systems" % key,
            )

    def test_revise_planets_list(self):
        """
        Test that revise_planets_list filters correctly
        """

        SU = self.instantiate_mod(SimulatedUniverse, addkeys={"Rprange": [1, 20]})

        # keep planets > 4 R_earth
        pInds = np.where(SU.Rp >= 4 * u.R_earth)[0]
        SU.revise_planets_list(pInds)
        self.assertTrue(
            np.all(SU.Rp >= 4 * u.R_earth),
            "revise_planets_list does not filter correctly",
        )

    def test_revise_stars_list(self):
        """
        Test that revise_stars_list filters correctly
        """

        SU = self.instantiate_mod(SimulatedUniverse)

        # star indices to keep
        sInds = np.arange(0, 10, dtype=int)
        SU.revise_stars_list(sInds)
        names = SU.TargetList.Name[sInds]

        # check correct stars in targetlist
        self.assertTrue(
            np.all(names == SU.TargetList.Name),
            "revise_stars_list does not select correct stars",
        )
        # check that planets are only assigned to stars in filtered list
        pInds = set(SU.plan2star)
        sInds = set(sInds)
        self.assertTrue(
            pInds.issubset(sInds),
            "revise_stars_list does not assign planets only to filtered stars",
        )

    def test_str(self):
        """
        Test __str__ method, for full coverage and check that all modules have
        required attributes.
        """

        atts_list = [
            "StarCatalog",
            "PlanetPopulation",
            "PlanetPhysicalModel",
            "OpticalSystem",
            "ZodiacalLight",
            "BackgroundSources",
            "PostProcessing",
            "Completeness",
            "TargetList",
            "nPlans",
            "plan2star",
            "sInds",
            "a",
            "e",
            "I",
            "O",
            "w",
            "Min",
            "M0",
            "p",
            "Rp",
            "Mp",
            "r",
            "v",
            "d",
            "s",
            "phi",
            "nEZ",
            "dMag",
            "WA",
            "phiIndex",
        ]

        for mod in self.allmods:
            if "__str__" not in mod.__dict__:
                continue

            obj = self.instantiate_mod(mod)

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
