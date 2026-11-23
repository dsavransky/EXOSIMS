import unittest

import astropy.units as u
import numpy as np

from EXOSIMS.Observatory.Lazuli import Lazuli


class TestLazuli(unittest.TestCase):
    """Check of the Lazuli frame against JPL Horizons."""

    def test_frame_against_horizons(self):
        """Compare the frame with one built from the Moon's state from JPL Horizons.

        The Moon's position and velocity relative to Earth (AU, AU/day, ICRF) are
        from JPL Horizons, ephemeris DE441 (Park et al. 2021, AJ 161, 105), at
        JD 2460575.75 TDB. That is the default orbit_epoch (MJD 60575.25 TAI) read
        as TDB, because spk_body passes the TAI Julian date to the kernel
        unchanged. If that is fixed, use JD 2460575.750372482 TDB.
        """
        obs = Lazuli()
        moon_pos = np.array(
            [1.338053912906314e-03, 1.822599099506684e-03, 9.823649380852890e-04]
        )
        moon_vel = np.array(
            [-4.955465506388118e-04, 3.161956664126426e-04, 1.788300697370541e-04]
        )

        x = moon_pos / np.linalg.norm(moon_pos)
        z = np.cross(moon_pos, moon_vel) / np.linalg.norm(np.cross(moon_pos, moon_vel))
        y = np.cross(z, x)
        ic = obs.observatory_orbit.states[0, :3]
        mu_star = obs.observatory_orbit.mu_star
        offset = (ic + [mu_star, 0, 0]) * obs.canonical_units.dist_km
        expected = offset[0] * x + offset[1] * y + offset[2] * z

        t = obs.orbit_epoch
        r_obs = obs.orbit(t, eclip=False)
        r_earth = obs.solarSystem_body_position(t, "Earth", eclip=False)
        np.testing.assert_allclose(
            (r_obs - r_earth).to_value(u.km)[0], expected, atol=0.1
        )
