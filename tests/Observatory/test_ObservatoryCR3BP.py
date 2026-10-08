import copy
import shutil
import tempfile
import unittest
import warnings

import astropy.units as u
import numpy as np
from astropy.time import Time
from orbit_gen.setup.system_config import CanonicalUnits

from EXOSIMS.Observatory.ObservatoryCR3BP import ObservatoryCR3BP


class TestOrbitRotationMath(unittest.TestCase):
    """Calls the real orbit() method against fully controlled inputs, so
    expected outputs can be hand-derived rather than compared against
    another piece of code.
    """

    def _build_stub_observatory(self, mu_star, r_synodic, r_primary_au, r_secondary_au):
        obs = ObservatoryCR3BP.__new__(ObservatoryCR3BP)
        obs.canonical_units = CanonicalUnits(
            dist_km=(1 * u.AU).to_value(u.km), time_days=58.0
        )
        obs.orbit_epoch = Time(
            np.array(0.0, ndmin=1, dtype=float), format="mjd", scale="tai"
        )
        obs.j2000_jd = Time(2000.0, format="jyear", scale="tai").jd
        obs.julian_century = 36525.0
        obs.obe = (
            lambda TDB: 23.439279
            - 1.0130102 * TDB
            - 5.086e-8 * (TDB**2.0)
            + 5.565e-7 * (TDB**3.0)
            + 1.6e-10 * (TDB**4.0)
            + 1.21e-11 * (TDB**5.0)
        )

        class _StubOrbit:
            def __init__(self, period, mu_star):
                self.period = period
                self.mu_star = mu_star

        obs.observatory_orbit = _StubOrbit(period=3.0, mu_star=mu_star)

        obs.r_interp = lambda t: np.tile(
            np.asarray(r_synodic).reshape(3, 1), (1, np.atleast_1d(t).size)
        )

        obs.primary_name = "Sun"
        obs.secondary_name = "Earth"

        def _stub_ephemeris(currentTime, bodyname, eclip=True):
            n = np.atleast_1d(currentTime.mjd).size
            pos = r_primary_au if bodyname == obs.primary_name else r_secondary_au
            return np.tile(pos, (n, 1)) * u.AU

        obs.solarSystem_body_position = _stub_ephemeris
        return obs

    def test_rotation_and_barycenter(self):
        """Tested at several non-trivial angles, since a sign error
        would still pass at theta0 = 0.
        """
        mu_star = 3e-6
        r_synodic = np.array([1.01, 0.0, 0.0])

        for theta0_deg in [0, 37, 90, 200, 359]:
            with self.subTest(theta0_deg=theta0_deg):
                theta0 = np.radians(theta0_deg)
                r_primary_au = np.array([0.0, 0.0, 0.0])
                r_secondary_au = np.array([np.cos(theta0), np.sin(theta0), 0.0])

                obs = self._build_stub_observatory(
                    mu_star, r_synodic, r_primary_au, r_secondary_au
                )

                t = Time(65000.0, format="mjd")

                r_eclip = obs.orbit(t, eclip=True)
                lon = (
                    np.degrees(np.arctan2(r_eclip[0, 1].value, r_eclip[0, 0].value))
                    % 360
                )
                radius = np.linalg.norm(r_eclip.to_value(u.AU)[0])

                self.assertAlmostEqual(lon, theta0_deg % 360, places=6)
                self.assertAlmostEqual(radius, 1.01 + mu_star, places=6)

                # eclip=False is a pure rotation on top of the eclip
                # result (obliquity), so it must preserve vector norm --
                # true regardless of the specific (inherited, not ours)
                # obliquity formula being correct
                r_equat = obs.orbit(t, eclip=False)
                self.assertAlmostEqual(
                    np.linalg.norm(r_equat.to_value(u.AU)[0]), radius, places=10
                )

    def test_time_wrapping(self):
        """The synodic-frame position handed to the rotation step must be
        the same at t = epoch + phase, t = epoch + period + phase,
        t = epoch + 2*period + phase -- i.e. the modulo wrapping is doing
        its job.
        """
        obs = self._build_stub_observatory(
            mu_star=3e-6,
            r_synodic=np.array([0.0, 0.0, 0.0]),  # overwritten below
            r_primary_au=np.array([0.0, 0.0, 0.0]),
            r_secondary_au=np.array([1.0, 0.0, 0.0]),
        )
        obs.observatory_orbit.period = 3.0

        calls = []

        def _tracking_r_interp(t):
            calls.append(np.atleast_1d(t).copy())
            return np.tile(np.array([[1.01], [0.0], [0.0]]), (1, np.atleast_1d(t).size))

        obs.r_interp = _tracking_r_interp

        epoch_mjd = obs.orbit_epoch.to_value("mjd")[0]
        period_days = obs.observatory_orbit.period * obs.canonical_units.time_days
        phase_days = 0.37 * period_days
        expected_phase_canonical = phase_days / obs.canonical_units.time_days

        for n in range(3):
            t = Time(epoch_mjd + n * period_days + phase_days, format="mjd")
            obs.orbit(t, eclip=True)

        for t_canonical in calls:
            self.assertAlmostEqual(t_canonical[0], expected_phase_canonical, places=6)


class TestSelectOrbitByPeriod(unittest.TestCase):
    """Exercises _select_orbit_by_period directly -- our own selection
    algorithm -- with a periods array we control. No real continuation
    needed.
    """

    def setUp(self):
        self.cu = CanonicalUnits(dist_km=(1 * u.AU).to_value(u.km), time_days=58.0)
        self.obs = ObservatoryCR3BP.__new__(ObservatoryCR3BP)

    def test_selects_closest_period(self):
        periods_days = np.array([50.0, 100.0, 150.0, 200.0])
        periods_canonical = periods_days / self.cu.time_days
        idx = self.obs._select_orbit_by_period(periods_canonical, 145.0, self.cu)
        self.assertEqual(idx, 2)  # 150 is closest to 145

    def test_warns_when_far_from_target(self):
        periods_days = np.array([50.0, 100.0])
        periods_canonical = periods_days / self.cu.time_days
        with self.assertWarns(UserWarning):
            self.obs._select_orbit_by_period(periods_canonical, 1000.0, self.cu)

    def test_no_warning_when_close(self):
        periods_days = np.array([50.0, 100.0, 150.0])
        periods_canonical = periods_days / self.cu.time_days
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.obs._select_orbit_by_period(periods_canonical, 149.0, self.cu)


MIN_SPECS = {
    "modules": {
        "PlanetPopulation": " ",
        "StarCatalog": " ",
        "OpticalSystem": " ",
        "ZodiacalLight": " ",
        "BackgroundSources": " ",
        "PlanetPhysicalModel": " ",
        "Observatory": " ",
        "TimeKeeping": " ",
        "PostProcessing": " ",
        "Completeness": " ",
        "TargetList": " ",
        "SimulatedUniverse": " ",
        "SurveySimulation": " ",
        "SurveyEnsemble": " ",
    },
    "scienceInstruments": [{"name": "imager"}],
    "starlightSuppressionSystems": [{"name": "coronagraph"}],
}

REQUIRED_ATTS = [
    "canonical_units",
    "orbit_epoch",
    "orbit_period_days",
    "observatory_orbit",
    "primary_name",
    "secondary_name",
    "r_interp",
    "v_interp",
]


class TestObservatoryCR3BP(unittest.TestCase):
    """Builds one real ObservatoryCR3BP in setUpClass, shared across all
    test methods here -- continuation runs at most once for this class."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="orbit_cr3bp_cache_")
        cls.obs = ObservatoryCR3BP(
            system="SE",
            preset="SE_L2_Northern_Halo",
            max_solutions=3,
            max_iter=200,
            cachedir=cls.tmpdir,
            **copy.deepcopy(MIN_SPECS),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_required_attributes(self):
        """Guards against the docstring's Attributes list drifting from
        what __init__ actually sets."""
        for att in REQUIRED_ATTS:
            self.assertTrue(hasattr(self.obs, att), f"Missing attribute {att}")

    def test_orbit_shape_and_type_scalar(self):
        """Mirrors EXOSIMS's own generic Observatory.test_orbit contract:
        a scalar-valued Time must still come back as shape (1, 3)."""
        t_ref = Time(2027.0, format="jyear")
        r_sc = self.obs.orbit(t_ref)
        self.assertEqual(type(r_sc), type(1.0 * u.km))
        self.assertEqual(r_sc.shape, (1, 3))

    def test_orbit_shape_multi(self):
        t_ref = Time(np.array([2027.0, 2027.5, 2028.0]), format="jyear")
        r_sc = self.obs.orbit(t_ref)
        self.assertEqual(r_sc.shape, (3, 3))

    def test_orbit_finite(self):
        t_ref = Time(np.linspace(2027.0, 2030.0, 20), format="jyear")
        r_sc = self.obs.orbit(t_ref)
        self.assertTrue(np.all(np.isfinite(r_sc.value)))

    def test_orbit_period_boundary_no_crash(self):
        """t = epoch + exactly N * period is the modulo edge case: rare
        for arbitrary mission times, but exactly what this test walks
        into deliberately. Should not raise interp1d's bounds_error.
        """
        epoch_mjd = self.obs.orbit_epoch.to_value("mjd")[0]
        period_days = self.obs.orbit_period_days.to_value(u.day)
        for n in range(1, 4):
            with self.subTest(n=n):
                t = Time(epoch_mjd + n * period_days, format="mjd")
                r_sc = self.obs.orbit(t)
                self.assertTrue(np.all(np.isfinite(r_sc.value)))

    def test_orbit_stays_near_secondary(self):
        """Whole-pipeline sanity check: our combination of a real
        generated trajectory with our rotation/barycenter code should
        stay near the secondary body, not put the spacecraft somewhere
        nonsensical. Band is generous on purpose -- this is a smoke
        test.
        """
        epoch_mjd = self.obs.orbit_epoch.to_value("mjd")[0]
        period_days = self.obs.orbit_period_days.to_value(u.day)
        times = Time(epoch_mjd + np.linspace(0, 2 * period_days, 30), format="mjd")

        r_sc = self.obs.orbit(times, eclip=True)
        r_earth = self.obs.solarSystem_body_position(
            times, self.obs.secondary_name, eclip=True
        )

        dist_au = np.linalg.norm((r_sc - r_earth).to_value(u.AU), axis=1)
        self.assertTrue(np.all(dist_au > 0.001))
        self.assertTrue(np.all(dist_au < 0.05))


class TestDesiredPeriodSelection(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="orbit_cr3bp_cache2_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_desired_period_selects_closest(self):
        kwargs = dict(
            system="SE",
            preset="SE_L2_Northern_Halo",
            max_solutions=5,
            max_iter=200,
            cachedir=self.tmpdir,
        )
        baseline = ObservatoryCR3BP(**kwargs, **copy.deepcopy(MIN_SPECS))
        target_days = baseline.orbit_period_days.to_value(u.day)

        selected = ObservatoryCR3BP(
            desired_period=target_days, **kwargs, **copy.deepcopy(MIN_SPECS)
        )
        got_days = selected.orbit_period_days.to_value(u.day)
        self.assertLess(abs(got_days - target_days), 0.1 * target_days)


class TestCacheConsistency(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="orbit_cr3bp_cache3_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cache_hit_matches_fresh_run(self):
        """Building the same config twice -- second time hitting the
        on-disk npz cache instead of rerunning continuation -- must
        produce an Observatory identical to the fresh one: same period,
        same mu_star (including its *type*, which catches the
        numpy-scalar-vs-float mismatch between the cache-hit and
        cache-miss branches), and same orbit() output.
        """
        kwargs = dict(
            system="SE",
            preset="SE_L2_Northern_Halo",
            max_solutions=3,
            max_iter=200,
            cachedir=self.tmpdir,
        )

        fresh = ObservatoryCR3BP(**kwargs, **copy.deepcopy(MIN_SPECS))
        cached = ObservatoryCR3BP(**kwargs, **copy.deepcopy(MIN_SPECS))

        self.assertAlmostEqual(
            fresh.observatory_orbit.period, cached.observatory_orbit.period
        )
        self.assertAlmostEqual(
            fresh.observatory_orbit.mu_star, cached.observatory_orbit.mu_star
        )
        self.assertEqual(
            np.ndim(fresh.observatory_orbit.mu_star),
            0,
            "fresh mu_star is not scalar",
        )
        self.assertEqual(
            np.ndim(cached.observatory_orbit.mu_star),
            0,
            "cached mu_star is not scalar -- likely an unindexed array "
            "from np.load on the cache-hit path",
        )

        t_ref = Time(2027.0, format="jyear")
        r_fresh = fresh.orbit(t_ref)
        r_cached = cached.orbit(t_ref)
        self.assertTrue(
            np.allclose(r_fresh.to_value(u.AU), r_cached.to_value(u.AU), atol=1e-10)
        )

        # Periodicity: states[0] and states[-1] are one full period apart
        # and must coincide for an actually-periodic orbit. atol is set
        # relative to the (default, unoverridden here) corrector
        # convergence tolerance eps=1e-6 -- that residual, not the much
        # tighter solve_ivp integration tolerance, is what limits how
        # exactly the orbit closes, so the check needs headroom above it.
        #
        # On `fresh`, this exercises orbit-gen's own corrector (their
        # suite already covers this -- kept here only as a cheap belt-
        # and-suspenders check, not the point of this test).
        #
        # On `cached`, this is the one that matters: it's checking OUR
        # reconstruction code (Orbit(x0=state[0], z0=state[2],
        # vy0=state[4], ...) followed by .propagate()) actually
        # reproduces a closed orbit -- e.g. it would catch pulling the
        # wrong state index (state[1]=y0, always ~0 by symmetry, instead
        # of state[2]=z0), which orbit-gen's own tests have no way to
        # know to check, since this reconstruction is entirely our code.
        periodicity_atol = 1e-4
        self.assertTrue(
            np.allclose(
                fresh.observatory_orbit.states[0],
                fresh.observatory_orbit.states[-1],
                atol=periodicity_atol,
            ),
            "fresh orbit does not close -- orbit-gen corrector issue",
        )
        self.assertTrue(
            np.allclose(
                cached.observatory_orbit.states[0],
                cached.observatory_orbit.states[-1],
                atol=periodicity_atol,
            ),
            "cached orbit does not close -- likely a bug in the "
            "cache-hit IC reconstruction (wrong state index, wrong "
            "period, etc.), since this is our own code, not orbit-gen's",
        )


if __name__ == "__main__":
    unittest.main()
