from EXOSIMS.PlanetPopulation.DulzPlavchan import DulzPlavchan
import astropy.units as u
import numpy as np
from EXOSIMS.util._numpy_compat import copy_if_needed


class AlbedoByPhaseSma(DulzPlavchan):
    """Planet Population module based on occurrence rate tables from Shannon Dulz and
    Peter Plavchan.

    NOTE: This assigns constant albedo based on radius ranges. When
    ``use_spectrum`` is enabled, it can also return wavelength-dependent albedos
    using a precomputed lookup table.

    Attributes:
        SAG13coeffs (float 4x2 ndarray):
            Coefficients used by the SAG13 broken power law. The 4 lines
            correspond to Gamma, alpha, beta, and the minimum radius.
        Gamma (float ndarray):
            Gamma coefficients used by SAG13 broken power law.
        alpha (float ndarray):
            Alpha coefficients used by SAG13 broken power law.
        beta (float ndarray):
            Beta coefficients used by SAG13 broken power law.
        Rplim (float ndarray):
            Minimum radius used by SAG13 broken power law.
        SAG13starMass (astropy Quantity):
            Assumed stellar mass corresponding to the given set of coefficients.
        mu (astropy Quantity):
            Gravitational parameter associated with SAG13starMass.
        Ca (float 2x1 ndarray):
            Constants used for sampling.
        ps (float nx1 ndarray):
            Constant geometric albedo values.
        Rb (float (n-1)x1 ndarray):
            Planetary radius break points for albedos in earthRad.
        Rbs (float (n+1)x1 ndarray):
            Planetary radius break points with 0 padded on left and np.inf
            padded on right
        use_spectrum (bool):
            Set True to enable spectral albedo lookup
        spectral_albedo_table (dict):
            Table describing spectral albedo as a function of phase angle,
            semi-major axis, and wavelength. Requires keys 'phi', 'a',
            ''wavelength'', and ''p''.

    """

    def __init__(
        self,
        starMass=1.0,
        occDataPath=None,
        esigma=0.175 / np.sqrt(np.pi / 2.0),
        ps=[0.2, 0.5],
        Rb=[1.4],
        use_spectrum=False,
        spectral_albedo_table=None,
        **specs,
    ):
        self.ps = np.array(ps, ndmin=1, copy=copy_if_needed)
        self.Rb = np.array(Rb, ndmin=1, copy=copy_if_needed)

        #################################### newly added
        self.use_spectrum = use_spectrum
        self._spectral_phi_grid = None
        self._spectral_phi_unit = None
        self._spectral_a_grid = None
        self._spectral_a_unit = None
        self._spectral_wavelength_grid = None
        self._spectral_wavelength_unit = None
        self._spectral_albedo_grid = None

        if spectral_albedo_table is not None:
            self._initialize_spectral_albedo_table(spectral_albedo_table)
        elif self.use_spectrum:
            raise ValueError(
                "use_spectrum is True but no spectral_albedo_table was provided"
            )
        ################################ end newly added
        specs["prange"] = [np.min(ps), np.max(ps)]
        DulzPlavchan.__init__(
            self, starMass=starMass, occDataPath=occDataPath, esigma=esigma, **specs
        )

        # check to ensure proper inputs
        assert (
            len(self.ps) - len(self.Rb) == 1
        ), "input albedos must have one more element than break radii"
        self.Rbs = np.hstack((0.0, self.Rb, np.inf))

        # albedo is constant for planetary radius range
        self.pfromRp = True

    def _initialize_spectral_albedo_table(self, table):
        try:
            phi_values = table["phi"]
            sma_values = table["a"]
            wavelength_values = table["wavelength"]
            albedo_values = table["p"]
        except KeyError as err:
            raise ValueError(
                "spectral_albedo_table must include 'phi', 'a', 'wavelength', and 'p' keys"
            ) from err

        phi_unit = self._parse_unit(table.get("phi_unit"))
        sma_unit = self._parse_unit(table.get("a_unit"))
        wavelength_unit = self._parse_unit(table.get("wavelength_unit"))

        self._spectral_phi_grid, self._spectral_phi_unit = self._prepare_axis(
            phi_values, phi_unit, "phi"
        )
        self._spectral_a_grid, self._spectral_a_unit = self._prepare_axis(
            sma_values, sma_unit, "a"
        )
        (
            self._spectral_wavelength_grid,
            self._spectral_wavelength_unit,
        ) = self._prepare_axis(wavelength_values, wavelength_unit, "wavelength")

        albedo_array = np.array(
            albedo_values, ndmin=3, copy=copy_if_needed, dtype=float
        )
        expected_shape = (
            self._spectral_phi_grid.size,
            self._spectral_a_grid.size,
            self._spectral_wavelength_grid.size,
        )
        if albedo_array.shape != expected_shape:
            raise ValueError(
                "spectral_albedo_table['p'] must have shape {} but has {}".format(
                    expected_shape, albedo_array.shape
                )
            )

        self._spectral_albedo_grid = albedo_array

    @staticmethod
    def _parse_unit(raw_unit):
        if raw_unit is None:
            return None
        if isinstance(raw_unit, u.UnitBase):
            return raw_unit
        return u.Unit(raw_unit)

    def _prepare_axis(self, values, override_unit, name):
        if isinstance(values, u.Quantity):
            if override_unit is not None:
                axis = np.array(
                    values.to(override_unit).value,
                    ndmin=1,
                    copy=copy_if_needed,
                    dtype=float,
                )
                axis_unit = override_unit
            else:
                axis = np.array(
                    values.value,
                    ndmin=1,
                    copy=copy_if_needed,
                    dtype=float,
                )
                axis_unit = values.unit
        else:
            axis = np.array(
                values, ndmin=1, copy=copy_if_needed, dtype=float
            )
            axis_unit = override_unit

        if axis.ndim != 1:
            raise ValueError(
                f"spectral albedo axis '{name}' must be one-dimensional"
            )
        if axis.size < 2:
            raise ValueError(
                f"spectral albedo axis '{name}' must contain at least two points"
            )
        if not np.all(np.diff(axis) > 0):
            raise ValueError(
                f"spectral albedo axis '{name}' must be strictly increasing"
            )

        return axis.astype(float, copy=False), axis_unit

    def _convert_axis_value(self, value, axis_unit, name):
        if isinstance(value, u.Quantity):
            if axis_unit is None:
                raise ValueError(
                    f"Received a quantity for {name}, but the spectral table axis"
                    f" '{name}' is unitless. Provide '{name}_unit' in the table or"
                    " pass a unitless value."
                )
            scalar = value.to_value(axis_unit)
        else:
            scalar_array = np.array(value, ndmin=1, dtype=float)
            if scalar_array.size != 1:
                raise ValueError(
                    f"Expected scalar for {name} but received array with"
                    f" {scalar_array.size} elements"
                )
            scalar = float(scalar_array[0])

        return scalar

    @staticmethod
    def _cast_albedo_output(original, value):
        if np.isscalar(original):
            return type(original)(value)

        original_array = np.asanyarray(original)
        if original_array.shape == ():
            return original_array.dtype.type(value)

        return np.full_like(original_array, value, dtype=original_array.dtype)

    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and
        planetary radius (earthRad)

        Semi-major axis and planetary radius are jointly distributed.
        Eccentricity is a Rayleigh distribution. Albedo is a constant value
        based on planetary radius.

        Args:
            n (integer):
                Number of samples to generate

        Returns:
            tuple:
            a (astropy Quantity array):
                Semi-major axis in units of AU
            e (float ndarray):
                Eccentricity
            p (float ndarray):
                Geometric albedo
            Rp (astropy Quantity array):
                Planetary radius in units of earthRad

        """
        n = self.gen_input_check(n)
        # generate semi-major axis and planetary radius samples
        a, Rp = self.gen_sma_radius(n)

        # check for constrainOrbits == True for eccentricity samples
        # constants
        C1 = np.exp(-(self.erange[0] ** 2) / (2.0 * self.esigma**2))
        ar = self.arange.to("AU").value
        if self.constrainOrbits:
            # restrict semi-major axis limits
            arcon = np.array(
                [ar[0] / (1.0 - self.erange[0]), ar[1] / (1.0 + self.erange[0])]
            )
            # clip sma values to sma range
            sma = np.clip(a.to("AU").value, arcon[0], arcon[1])
            # upper limit for eccentricity given sma
            elim = np.zeros(len(sma))
            amean = np.mean(ar)
            elim[sma <= amean] = 1.0 - ar[0] / sma[sma <= amean]
            elim[sma > amean] = ar[1] / sma[sma > amean] - 1.0
            elim[elim > self.erange[1]] = self.erange[1]
            elim[elim < self.erange[0]] = self.erange[0]
            # additional constant
            C2 = C1 - np.exp(-(elim**2) / (2.0 * self.esigma**2))
            a = sma * u.AU
        else:
            C2 = self.enorm
        e = self.esigma * np.sqrt(-2.0 * np.log(C1 - C2 * np.random.uniform(size=n)))
        # generate albedo from planetary radius
        p = self.get_p_from_Rp(Rp)

        return a, e, p, Rp

    def get_p_from_Rp(self, Rp):
        """Generate constant albedos for radius ranges

        Args:
            Rp (astropy Quantity array):
                Planetary radius with units of earthRad

        Returns:
            float ndarray:
                Albedo values

        """
        Rp = np.array(Rp.to("earthRad").value, ndmin=1, copy=copy_if_needed)
        p = np.zeros(Rp.shape)
        for i in range(len(self.Rbs) - 1):
            mask = np.where((Rp >= self.Rbs[i]) & (Rp < self.Rbs[i + 1]))
            p[mask] = self.ps[i]

        return p

    def get_p_from_phi_a(self, mode, p, phi, a, wavelength):
        """Return an albedo appropriate for the observing mode and wavelength.

        Args:
            mode (dict):
                Observing mode dictionary containing key ``detectionMode`` that
                identifies the default detection mode.
            p (float or ndarray):
                Baseline geometric albedo returned by :func:`get_p_from_Rp`.
            phi (float or Quantity):
                Planet phase angle used for the spectral lookup.
            a (float or Quantity):
                Semi-major axis used for the spectral lookup.
            wavelength (float or Quantity):
                Wavelength at which to evaluate the spectral albedo.

        Returns:
            float or ndarray:
                Albedo value corresponding to the supplied mode and wavelength.

        """
        if (
            mode.get("detectionMode", False)
            or not self.use_spectrum
            or self._spectral_albedo_grid is None
        ):
            return p

        phi_value = self._convert_axis_value(phi, self._spectral_phi_unit, "phi")
        a_value = self._convert_axis_value(a, self._spectral_a_unit, "a")
        wavelength_value = self._convert_axis_value(
            wavelength, self._spectral_wavelength_unit, "wavelength"
        )

        phi_value = np.clip(
            phi_value, self._spectral_phi_grid[0], self._spectral_phi_grid[-1]
        )
        a_value = np.clip(
            a_value, self._spectral_a_grid[0], self._spectral_a_grid[-1]
        )
        wavelength_value = np.clip(
            wavelength_value,
            self._spectral_wavelength_grid[0],
            self._spectral_wavelength_grid[-1],
        )

        phi_interp = np.empty(
            (self._spectral_a_grid.size, self._spectral_wavelength_grid.size)
        )
        for j in range(self._spectral_a_grid.size):
            phi_interp[j, :] = np.interp(
                phi_value,
                self._spectral_phi_grid,
                self._spectral_albedo_grid[:, j, :],
                left=self._spectral_albedo_grid[0, j, :],
                right=self._spectral_albedo_grid[-1, j, :],
            )

        albedo_vs_wavelength = np.empty(self._spectral_wavelength_grid.size)
        for k in range(self._spectral_wavelength_grid.size):
            albedo_vs_wavelength[k] = np.interp(
                a_value,
                self._spectral_a_grid,
                phi_interp[:, k],
                left=phi_interp[0, k],
                right=phi_interp[-1, k],
            )

        albedo_value = np.interp(
            wavelength_value,
            self._spectral_wavelength_grid,
            albedo_vs_wavelength,
            left=albedo_vs_wavelength[0],
            right=albedo_vs_wavelength[-1],
        )

        return self._cast_albedo_output(p, float(albedo_value))
