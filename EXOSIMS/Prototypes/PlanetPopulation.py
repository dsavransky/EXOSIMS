import copy
import numbers

import astropy.units as u
import numpy as np

from EXOSIMS.util.get_dirs import get_cache_dir
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.keyword_fun import get_all_args
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util._numpy_compat import copy_if_needed


class PlanetPopulation(object):
    r""":ref:`PlanetPopulation` Prototype

    Args:
        arange (list(float)):
            [Min, Max] semi-major axis (in AU). Defaults to [0.1,100.]
        erange (list(float)):
            [Min, Max] eccentricity. Defaults to [0.01,0.99]
        Irange (list(float)):
            [Min, Max] inclination (in degrees). Defaults to [0.,180.]
        Orange (list(float)):
            [Min, Max] longitude of the ascending node (in degrees).
            Defaults to [0.,360.]
        wrange (list(float)):
            [Min, Max] argument of periapsis. Defaults to [0.,360.]
        prange (list(float)):
            [Min, Max] geometric albedo. Defaults to [0.1,0.6]
        Rprange (list(float)):
            [Min, Max] planet radius (in Earth radii). Defaults to [1.,30.]
        Mprange (list(float)):
            [Min, Max] planet mass (in Earth masses). Defaults to [1.,4131.]
        scaleOrbits (bool):
            Scale orbits by :math:`\sqrt{L}` where :math:`L` is the stellar luminosity.
            This has the effect of matching insolation distnaces and preserving the
            habitable zone of the population.  Defaults to False.
        constrainOrbits (bool):
            Do not allow orbits where orbital radius can exceed the ``arange`` limits.
            Defaults to False
        eta (float):
            Overall occurrence rate of the population.  The expected number of planets
            per target star. Must be strictly positive, but may be greater than 1
            (if more than 1 planet is expected per star, on average). Defaults to 0.1.
        cachedir (str, optional):
            Full path to cachedir.
            If None (default) use default (see :ref:`EXOSIMSCACHE`)
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        _outspec (dict):
            :ref:`sec:outspec`
        arange (astropy.units.quantity.Quantity):
            [Min, Max] semi-major axis
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)
        constrainOrbits (bool):
            Do not allow orbits where orbital radius can exceed the ``arange`` limits.
        erange (numpy.ndarray):
            [Min, Max] eccentricity.
        eta (float):
            Overall occurrence rate of the population.  The expected number of planets
            per target star. Must be strictly positive, but may be greater than 1
            (if more than 1 planet is expected per star, on average).
        Irange (astropy.units.quantity.Quantity):
            [Min, Max] inclination
        Mprange (astropy.units.quantity.Quantity):
            [Min, Max] planet mass
        Orange (astropy.units.quantity.Quantity):
            [Min, Max] longitude of the ascending node
        pfromRp (bool):
            Albedo is dependent on planetary radius
        PlanetPhysicalModel (:ref:`PlanetPhysicalModel`):
            Planet physical model object
        prange (numpy.ndarray):
            [Min, Max] geometric albedo.
        Rprange (astropy.units.quantity.Quantity):
            [Min, Max] planet radius
        rrange (astropy.units.quantity.Quantity):
            [Min, Max] orbital radius
        scaleOrbits (bool):
            Scale orbits by :math:`\sqrt{L}` where :math:`L` is the stellar luminosity.
            This has the effect of matching insolation distnaces and preserving the
            habitable zone of the population.
        wrange (astropy.units.quantity.Quantity):
            [Min, Max] argument of periapsis.

    """

    _modtype = "PlanetPopulation"

    def __init__(
        self,
        arange=[0.1, 100.0],
        erange=[0.01, 0.99],
        Irange=[0.0, 180.0],
        Orange=[0.0, 360.0],
        wrange=[0.0, 360.0],
        prange=[0.1, 0.6],
        Rprange=[1.0, 30.0],
        Mprange=[1.0, 4131.0],
        scaleOrbits=False,
        constrainOrbits=False,
        eta=0.1,
        cachedir=None,
        **specs
    ):

        # start the outspec
        self._outspec = {}

        # get the cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        specs["cachedir"] = self.cachedir

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

        # check range of parameters
        self.arange = self.checkranges(arange, "arange") * u.AU
        self.erange = self.checkranges(erange, "erange")
        self.Irange = self.checkranges(Irange, "Irange") * u.deg
        self.Orange = self.checkranges(Orange, "Orange") * u.deg
        self.wrange = self.checkranges(wrange, "wrange") * u.deg
        self.prange = self.checkranges(prange, "prange")
        self.Rprange = self.checkranges(Rprange, "Rprange") * u.earthRad
        self.Mprange = self.checkranges(Mprange, "Mprange") * u.earthMass

        assert isinstance(scaleOrbits, bool), "scaleOrbits must be boolean"
        # scale planetary orbits by sqrt(L)
        self.scaleOrbits = scaleOrbits

        assert isinstance(constrainOrbits, bool), "constrainOrbits must be boolean"
        # constrain planetary orbital radii to sma range
        self.constrainOrbits = constrainOrbits
        assert isinstance(eta, numbers.Number) and (
            eta > 0
        ), "eta must be strictly positive"
        # global occurrence rate defined as expected number of planets per
        # star in a given universe
        self.eta = eta

        # populate outspec with all inputs
        kws = get_all_args(self.__class__)
        ignore_kws = ["self", "cachedir"]
        kws = list((set(kws) - set(ignore_kws)))
        for att in kws:
            if att not in ["vprint", "_outspec"]:
                dat = copy.copy(self.__dict__[att])
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat

        # albedo is independent of planetary radius range
        self.pfromRp = False

        # derive orbital radius range
        ar = self.arange.to("AU").value
        er = self.erange
        if self.constrainOrbits:
            self.rrange = [ar[0], ar[1]] * u.AU
        else:
            self.rrange = [ar[0] * (1.0 - er[1]), ar[1] * (1.0 + er[1])] * u.AU

        # define prototype distributions of parameters (uniform and log-uniform)
        self.uniform = lambda x, v: np.array(
            (np.array(x) >= v[0]) & (np.array(x) <= v[1]), dtype=float, ndmin=1
        ) / (v[1] - v[0])
        self.logunif = lambda x, v: np.array(
            (np.array(x) >= v[0]) & (np.array(x) <= v[1]), dtype=float, ndmin=1
        ) / (x * np.log(v[1] / v[0]))

        # import PlanetPhysicalModel
        self.PlanetPhysicalModel = get_module(
            specs["modules"]["PlanetPhysicalModel"], "PlanetPhysicalModel"
        )(**specs)

    def checkranges(self, var, name):
        """Helper function provides asserts on all 2 element lists of ranges

        Args:
            var (list):
                2-element list
            name (str):
                Variable name

        Returns:
            list:
                Sorted input variable

        Raises AssertionError on test fail.

        """

        # reshape var
        assert len(var) == 2, "%s must have two elements," % name
        var = np.array([float(v) for v in var])

        # check values
        if name in ["arange", "Rprange", "Mprange"]:
            assert np.all(var > 0), "%s values must be strictly positive" % name
        if name in ["erange", "prange"]:
            assert np.all(var >= 0) and np.all(var <= 1), (
                "%s values must be between 0 and 1" % name
            )

        # the second element must be greater or equal to the first
        if var[1] < var[0]:
            var = var[::-1]

        return var

    def __str__(self):
        """String representation of the Planet Population object

        When the command 'print' is used on the Planet Population object, this
        method will print the attribute values contained in the object"""

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Planet Population class object attributes"

    def gen_input_check(self, n):
        """
        Helper function checks that input is integer, casts to int, is >= 0

        Args:
            n (float):
                An integer to validate

        Returns:
            int:
                The input integer as an integer

        Raises AssertionError on test fail.

        """
        assert (
            isinstance(n, numbers.Number) and float(n).is_integer()
        ), "Input must be an integer value."
        assert n >= 0, "Input must be nonnegative"

        return int(n)

    def gen_mass(self, n):
        """Generate planetary mass values in units of Earth mass.

        The prototype provides a log-uniform distribution between the minimum and
        maximum values.

        Args:
            n (int):
                Number of samples to generate

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Planet mass values in units of Earth mass.

        """
        n = self.gen_input_check(n)
        Mpr = self.Mprange.to("earthMass").value
        Mp = (
            np.exp(np.random.uniform(low=np.log(Mpr[0]), high=np.log(Mpr[1]), size=n))
            * u.earthMass
        )

        return Mp

    def gen_angles(self, n, commonSystemPlane=False, commonSystemPlaneParams=None):
        """Generate inclination, longitude of the ascending node, and argument
        of periapse in degrees

        The prototype generates inclination as sinusoidally distributed and
        longitude of the ascending node and argument of periapse as uniformly
        distributed.

        Args:
            n (int):
                Number of samples to generate
            commonSystemPlane (bool):
                Generate delta inclinations from common orbital plane rather than
                fully independent inclinations and Omegas.  Defaults False.  If True,
                commonSystemPlaneParams must be supplied.
            commonSystemPlaneParams (None or list):
                4 element list of [I mean, I standard deviation,
                                   O mean, O standard deviation]
                in units of degrees, describing the distribution of
                inclinations and Omegas relative to a common orbital plane.
                Ignored if commonSystemPlane is False.

        Returns:
            tuple:
                I (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Inclination in units of degrees OR deviation in inclination (deg)
                O (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Longitude of the ascending node (deg)
                w (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Argument of periapsis (deg)

        """
        n = self.gen_input_check(n)
        # inclination
        C = 0.5 * (np.cos(self.Irange[0]) - np.cos(self.Irange[1]))
        if commonSystemPlane:
            assert (
                len(commonSystemPlaneParams) == 4
            ), "commonSystemPlaneParams must be a four-element list"
            I = (  # noqa: E741
                np.random.normal(
                    loc=commonSystemPlaneParams[0],
                    scale=commonSystemPlaneParams[1],
                    size=n,
                )
                * u.deg
            )
            O = (  # noqa: E741
                np.random.normal(
                    loc=commonSystemPlaneParams[2],
                    scale=commonSystemPlaneParams[3],
                    size=n,
                )
                * u.deg
            )
        else:
            I = (  # noqa: E741
                np.arccos(np.cos(self.Irange[0]) - 2.0 * C * np.random.uniform(size=n))
            ).to("deg")
            # longitude of the ascending node
            Or = self.Orange.to("deg").value
            O = np.random.uniform(low=Or[0], high=Or[1], size=n) * u.deg  # noqa: E741

        # argument of periapse
        wr = self.wrange.to("deg").value
        w = np.random.uniform(low=wr[0], high=wr[1], size=n) * u.deg

        return I, O, w

    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)

        The prototype generates semi-major axis and planetary radius with
        log-uniform distributions and eccentricity and geometric albedo with
        uniform distributions.

        Args:
            n (int):
                Number of samples to generate

        Returns:
            tuple:
                a (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Semi-major axis in units of AU
                e (~numpy.ndarray(float)):
                    Eccentricity
                p (~numpy.ndarray(float)):
                    Geometric albedo
                Rp (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Planetary radius in units of earthRad

        """
        n = self.gen_input_check(n)
        # generate samples of semi-major axis
        ar = self.arange.to("AU").value
        # check if constrainOrbits == True for eccentricity
        if self.constrainOrbits:
            # restrict semi-major axis limits
            arcon = np.array(
                [ar[0] / (1.0 - self.erange[0]), ar[1] / (1.0 + self.erange[0])]
            )
            a = (
                np.exp(
                    np.random.uniform(
                        low=np.log(arcon[0]), high=np.log(arcon[1]), size=n
                    )
                )
                * u.AU
            )
            tmpa = a.to("AU").value
            # upper limit for eccentricity given sma
            elim = np.zeros(len(a))
            amean = np.mean(ar)
            elim[tmpa <= amean] = 1.0 - ar[0] / tmpa[tmpa <= amean]
            elim[tmpa > amean] = ar[1] / tmpa[tmpa > amean] - 1.0
            elim[elim > self.erange[1]] = self.erange[1]
            elim[elim < self.erange[0]] = self.erange[0]

            # uniform distribution
            e = np.random.uniform(low=self.erange[0], high=elim, size=n)
        else:
            a = (
                np.exp(np.random.uniform(low=np.log(ar[0]), high=np.log(ar[1]), size=n))
                * u.AU
            )
            e = np.random.uniform(low=self.erange[0], high=self.erange[1], size=n)

        # generate geometric albedo
        pr = self.prange
        p = np.random.uniform(low=pr[0], high=pr[1], size=n)
        # generate planetary radius
        Rpr = self.Rprange.to("earthRad").value
        Rp = (
            np.exp(np.random.uniform(low=np.log(Rpr[0]), high=np.log(Rpr[1]), size=n))
            * u.earthRad
        )

        return a, e, p, Rp

    def dist_eccen_from_sma(self, e, a):
        """Probability density function for eccentricity constrained by
        semi-major axis, such that orbital radius always falls within the
        provided sma range.

        The prototype provides a uniform distribution between the minimum and
        maximum allowable values.

        Args:
            e (~numpy.ndarray(float)):
                Eccentricity values
            a (~numpy.ndarray(float)):
                Semi-major axis value in AU. Not an astropy quantity.

        Returns:
            ~numpy.ndarray(float):
                Probability density of eccentricity constrained by semi-major axis

        """

        # cast a and e to array
        e = np.array(e, ndmin=1, copy=copy_if_needed)
        a = np.array(a, ndmin=1, copy=copy_if_needed)
        # if a is length 1, copy a to make the same shape as e
        if a.ndim == 1 and len(a) == 1:
            a = a * np.ones(e.shape)

        # unitless sma range
        ar = self.arange.to("AU").value
        arcon = np.array(
            [ar[0] / (1.0 - self.erange[0]), ar[1] / (1.0 + self.erange[0])]
        )
        # upper limit for eccentricity given sma
        elim = np.zeros(a.shape)
        amean = np.mean(arcon)
        elim[a <= amean] = 1.0 - ar[0] / a[a <= amean]
        elim[a > amean] = ar[1] / a[a > amean] - 1.0
        elim[elim > self.erange[1]] = self.erange[1]
        elim[elim < self.erange[0]] = self.erange[0]

        # if e and a are two arrays of different size, create a 2D grid
        if a.size not in [1, e.size]:
            elim, e = np.meshgrid(elim, e)
        f = np.zeros(e.shape)
        mask = np.where((a >= arcon[0]) & (a <= arcon[1]))
        f[mask] = self.uniform(e[mask], (self.erange[0], elim[mask]))

        return f

    def dist_sma(self, a):
        """Probability density function for semi-major axis in AU

        The prototype provides a log-uniform distribution between the minimum
        and maximum values.

        Args:
            a (~numpy.ndarray(float)):
                Semi-major axis value(s) in AU. Not an astropy quantity.

        Returns:
            ~numpy.ndarray(float):
                Semi-major axis probability density

        """

        return self.logunif(a, self.arange.to("AU").value)

    def dist_eccen(self, e):
        """Probability density function for eccentricity

        The prototype provides a uniform distribution between the minimum and
        maximum values.

        Args:
            e (~numpy.ndarray(float)):
                Eccentricity value(s)

        Returns:
            ~numpy.ndarray(float):
                Eccentricity probability density

        """

        return self.uniform(e, self.erange)

    def dist_albedo(self, p):
        """Probability density function for albedo

        The prototype provides a uniform distribution between the minimum and
        maximum values.

        Args:
            p (~numpy.ndarray(float)):
                Albedo value(s)

        Returns:
            ~numpy.ndarray(float):
                Albedo probability density

        """

        return self.uniform(p, self.prange)

    def dist_radius(self, Rp):
        """Probability density function for planetary radius in Earth radius

        The prototype provides a log-uniform distribution between the minimum
        and maximum values.

        Args:
            Rp (~numpy.ndarray(float)):
                Planetary radius value(s) in Earth radius. Not an astropy quantity.

        Returns:
            ~numpy.ndarray(float):
                Planetary radius probability density

        """

        return self.logunif(Rp, self.Rprange.to("earthRad").value)

    def dist_mass(self, Mp):
        """Probability density function for planetary mass in Earth mass

        The prototype provides an unbounded power law distribution. Note
        that this should really be a function of a density model and the radius
        distribution for all implementations that use it.

        Args:
            Mp (~numpy.ndarray(float)):
                Planetary mass value(s) in Earth mass. Not an astropy quantity.

        Returns:
            ~numpy.ndarray(float):
                Planetary mass probability density

        """

        Mearth = np.array(Mp, ndmin=1) * u.earthMass

        tmp = ((Mearth >= self.Mprange[0]) & (Mearth <= self.Mprange[1])).astype(float)
        Mjup = Mearth.to("jupiterMass").value

        return tmp * Mjup ** (-1.3)
