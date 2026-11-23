import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from orbit_gen import Orbit
from orbit_gen.setup.system_config import CanonicalUnits
from orbit_gen.utils import units

from EXOSIMS.Observatory.ObservatoryCR3BP import ObservatoryCR3BP
from EXOSIMS.Prototypes.Observatory import Observatory


class Lazuli(ObservatoryCR3BP):
    """Observatory on the Lazuli 3:1 lunar-resonant highly elliptical orbit.

    The orbit proposed for the Lazuli mission: perigee and apogee altitudes of about
    70,000 and 285,000 km, an inclination of 29 degrees to the ecliptic, and three loops
    per lunar month. It is modeled in the Earth-Moon CR3BP, starting at perigee at an
    altitude of 70,000 km. The next perigee altitudes are 71,060 and 71,122 km, and the
    apogee altitudes are 284,588, 287,125, and 284,638 km. That trajectory is not
    periodic in the CR3BP, so it is propagated once for its return time T, and
    repeated.

    T is the time at which the trajectory, after three loops, next crosses the
    y = 0 plane in the same direction near its starting point, where it comes
    back to within 236 km and 56 m/s of its start. The initial y velocity is
    chosen such that T is one sidereal month, which keeps the orbit in 3:1
    resonance with the ephemeris Moon. Each repeat jumps by that much, which
    implies station-keeping.

    As in ``ObservatoryCR3BP`` for the Earth-Moon system, the synodic frame is
    placed at each time with the ephemeris Moon: x along the Earth-Moon line and
    z along the normal to the Moon's instantaneous orbit plane. The position is
    measured from Earth. The orbit is 32.935 degrees to the Moon's orbit plane,
    so its inclination to the ecliptic depends on where the Moon is on its orbit
    when the orbit starts. The initial condition is for a start at MJD 60575.25
    (the default ``orbit_epoch``), when the Moon is 50.9 degrees past its
    ascending node, which gives 29.000 degrees to the ecliptic with the JPL
    ephemeris (29.13 degrees with static ephemerides). The same orbit started at
    other times is between 27.6 and 38.2 degrees to the ecliptic, so
    ``orbit_epoch`` must be 60575.25 and ``orbit_epoch_phase`` must be 0. At
    ``orbit_epoch``, the observatory is at this perigee, on the opposite side of
    Earth from the Moon at 28.77 degrees north ecliptic latitude.

    The inclination to the ecliptic varies between 28.12 and 30.50 degrees in
    the first repeat, because the frame follows the Moon's actual, non-uniform
    motion. At the start of each repeat over 5 years, it is between 27.65 and
    29.63 degrees, as the Moon's orbit plane moves.

    The ``__init__`` and ``orbit`` methods of ``ObservatoryCR3BP`` are overloaded.
    ``__init__`` runs the Observatory prototype's constructor instead of
    ``ObservatoryCR3BP``'s, so no family of orbits is generated, the
    ``ObservatoryCR3BP`` inputs are not used, and ``v_interp`` is not set.

    Args:
        **specs:
            :ref:`sec:inputspec`

    """

    def __init__(self, **specs):
        # run prototype constructor first, not ObservatoryCR3BP's, which generates
        # a family of orbits by continuation
        Observatory.__init__(self, **specs)

        self.primary_name = "Earth"
        self.secondary_name = "Moon"

        # enforce initial condition to enforce correct inclination
        epoch_mjd = 60575.25
        assert self.orbit_epoch.mjd[0] == epoch_mjd and self.orbit_epoch_phase == 0, (
            f"Lazuli's orbit is only valid for the default orbit_epoch = {epoch_mjd} "
            "and orbit_epoch_phase = 0."
        )

        # GM of Earth and Moon in km^3/s^2, from the SPICE kernel gm_de440.tpc
        gm_earth = 398600.4355070226
        gm_moon = 4902.800118457549

        x0 = -0.17891306361798687
        z0 = 0.1080266156516575
        vy0 = -2.6410936777460554
        period = 6.291718496318983
        mu_star = gm_moon / (gm_earth + gm_moon)

        self.canonical_units = CanonicalUnits(
            dist_km=384400.0,
            time_days=np.sqrt(384400.0**3 / (gm_earth + gm_moon)) / 86400,
        )

        self.observatory_orbit = Orbit(
            x0=x0,
            z0=z0,
            vy0=vy0,
            period=period,
            mu_star=mu_star,
            canonical_units=self.canonical_units,
        )
        self.observatory_orbit.propagate()

        self.orbit_period_days = units.time_to_dimensional(period, self.canonical_units)

        self.r_interp = interp1d(
            self.observatory_orbit.times,
            self.observatory_orbit.states[:, 0:3].T,
            kind="cubic",
            assume_sorted=True,
        )

    def orbit(self, currentTime, eclip=False):
        """Finds observatory orbit positions vector in heliocentric equatorial
        (default) or ecliptic frame for current time (MJD).

        Args:
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD. May be a single time or an
                array of n times.
            eclip (bool):
                Boolean used to switch to heliocentric ecliptic frame.
                Defaults to False, corresponding to heliocentric equatorial
                frame.

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Observatory orbit positions vector in AU. nx3
        """

        mjdtime = np.array(currentTime.mjd, ndmin=1)
        dt_days = mjdtime - self.orbit_epoch.to_value("mjd")
        t_canonical = units.time_to_canonical(dt_days * u.d, self.canonical_units)
        t_canonical = t_canonical % self.observatory_orbit.period

        r_synodic = self.r_interp(t_canonical).T

        # observatory position relative to Earth
        r_synodic_earth = r_synodic + [self.observatory_orbit.mu_star, 0, 0]
        r_inertial_earth = self.synodic_to_ecliptic(r_synodic_earth, currentTime)
        r_inertial_earth_au = units.pos_to_dimensional(
            r_inertial_earth, self.canonical_units
        ).to(u.AU)

        r_earth = self.solarSystem_body_position(currentTime, "Earth", eclip=True)
        r_obs = r_earth + r_inertial_earth_au

        if not eclip:
            r_obs = self.eclip2equat(r_obs, currentTime)

        return r_obs
