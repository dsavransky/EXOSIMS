import importlib.resources
import warnings

from EXOSIMS.PlanetPopulation.DulzPlavchan import DulzPlavchan
import astropy.units as u
import numpy as np
from EXOSIMS.util._numpy_compat import copy_if_needed


class AlbedoByPhaseSma(DulzPlavchan):
    """Planet population module based on occurrence rate tables from Shannon Dulz and Peter Plavchan, with albedo as a function of phase angle and semi-major axis

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
    """

    def __init__(
        self,
        starMass=1.0,
        occDataPath=None,
        esigma=0.175 / np.sqrt(np.pi / 2.0),
        ps=[0.2, 0.5],
        Rb=[1.4],
        use_spectrum=False,
        **specs,
    ):
        """Initialize the planet population and optional spectral albedo lookup."""
        input_ranges = {
            name: np.array(specs[name], copy=copy_if_needed)
            for name in ("arange", "prange", "Rprange", "Mprange")
            if name in specs
        }
        self.ps = np.array(ps, ndmin=1, copy=copy_if_needed)
        self.Rb = np.array(Rb, ndmin=1, copy=copy_if_needed)

        self.use_spectrum = use_spectrum
        self._beta_grid = np.round(np.linspace(0, 180, num=15), 3)
        self._sma_grid = np.round(np.linspace(0.9, 1.7, num=15), 3)
        self._albedo_table_cache = {}
        self._albedo_directory = importlib.resources.files(
            "EXOSIMS.PlanetPhysicalModel"
        ).joinpath("exosims_albedo")
        specs.setdefault("prange", [np.min(ps), np.max(ps)])
        DulzPlavchan.__init__(
            self, starMass=starMass, occDataPath=occDataPath, esigma=esigma, **specs
        )

        if "arange" in input_ranges:
            self.arange = self.checkranges(input_ranges["arange"], "arange") * u.AU
            ar = self.arange.to("AU").value
            if self.constrainOrbits:
                self.rrange = [ar[0], ar[1]] * u.AU
            else:
                self.rrange = [
                    ar[0] * (1.0 - self.erange[1]),
                    ar[1] * (1.0 + self.erange[1]),
                ] * u.AU
            self._outspec["arange"] = self.arange.value
            self._outspec["rrange"] = self.rrange.value
        if "prange" in input_ranges:
            self.prange = self.checkranges(input_ranges["prange"], "prange")
            self._outspec["prange"] = self.prange
        if "Rprange" in input_ranges:
            self.Rprange = (
                self.checkranges(input_ranges["Rprange"], "Rprange") * u.earthRad
            )
            self._outspec["Rprange"] = self.Rprange.value
        if "Mprange" in input_ranges:
            self.Mprange = (
                self.checkranges(input_ranges["Mprange"], "Mprange") * u.earthMass
            )
            self._outspec["Mprange"] = self.Mprange.value

        # check to ensure proper inputs
        assert (
            len(self.ps) - len(self.Rb) == 1
        ), "input albedos must have one more element than break radii"
        self.Rbs = np.hstack((0.0, self.Rb, np.inf))

        # albedo is constant for planetary radius range
        self.pfromRp = True

    def gen_plan_params(self, n):
        """Generate semi-major axis (AU), eccentricity, geometric albedo, and planetary radius (earthRad).

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
        """Generate constant albedos for radius ranges.

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

    def get_p_from_phi_a(self, mode, beta, a, wl_dependency=None):
        """Return spectral albedo for phase angle, semi-major axis, and bandpass.

        Args:
            mode (dict):
                Observing mode
            beta (float or Quantity):
                Planet phase angle used for the spectral lookup.
            a (float or Quantity):
                Semi-major axis used for the spectral lookup.
            wl_dependency (int, optional):
                Wavelength-bin index. If omitted, use the full observing bandpass.

        Returns:
            float or ndarray:
                Band-averaged albedo value for the supplied beta and semi-major axis.

        """
        if not self.use_spectrum:
            raise ValueError(
                "get_p_from_phi_a requires use_spectrum=True for spectral albedo lookup"
            )

        lam = mode["lam"].to(u.um)
        directory = self._albedo_directory

        # mode wavelength of interest in microns (used as fallback)
        lam_value = u.Quantity(lam).to_value(u.um)

        if isinstance(beta, u.Quantity):
            beta_values = beta.to_value(u.deg)
        else:
            beta_values = beta
        if isinstance(a, u.Quantity):
            sma_values = a.to_value(u.AU)
        else:
            sma_values = a

        beta_scalar = np.ndim(beta_values) == 0
        sma_scalar = np.ndim(sma_values) == 0

        beta_array = np.asarray(beta_values, dtype=float)
        sma_array = np.asarray(sma_values, dtype=float)

        if beta_array.shape != sma_array.shape:
            raise ValueError("beta and a must have the same shape")

        beta_flat = beta_array.reshape(-1)
        sma_flat = sma_array.reshape(-1)

        beta_clipped = np.clip(beta_flat, self._beta_grid[0], self._beta_grid[-1])
        sma_clipped = np.clip(sma_flat, self._sma_grid[0], self._sma_grid[-1])

        # Interpolate between the four neighboring beta/SMA table files.
        beta_upper_idx = np.searchsorted(self._beta_grid, beta_clipped, side="left")
        beta_upper_idx = np.clip(beta_upper_idx, 0, self._beta_grid.size - 1)
        beta_exact = np.isclose(
            beta_clipped, self._beta_grid[beta_upper_idx], atol=1e-6
        )
        beta_lower_idx = np.where(
            beta_exact,
            beta_upper_idx,
            np.clip(beta_upper_idx - 1, 0, self._beta_grid.size - 1),
        )
        beta_lower_vals = self._beta_grid[beta_lower_idx]
        beta_upper_vals = self._beta_grid[beta_upper_idx]

        sma_upper_idx = np.searchsorted(self._sma_grid, sma_clipped, side="left")
        sma_upper_idx = np.clip(sma_upper_idx, 0, self._sma_grid.size - 1)
        sma_exact = np.isclose(
            sma_clipped, self._sma_grid[sma_upper_idx], atol=1e-6
        )
        sma_lower_idx = np.where(
            sma_exact,
            sma_upper_idx,
            np.clip(sma_upper_idx - 1, 0, self._sma_grid.size - 1),
        )
        sma_lower_vals = self._sma_grid[sma_lower_idx]
        sma_upper_vals = self._sma_grid[sma_upper_idx]

        beta_lower_keys = np.char.mod("%.3f", beta_lower_vals)
        beta_upper_keys = np.char.mod("%.3f", beta_upper_vals)
        sma_lower_keys = np.char.mod("%.3f", sma_lower_vals)
        sma_upper_keys = np.char.mod("%.3f", sma_upper_vals)

        required_keys = set()
        required_keys.update(zip(beta_lower_keys, sma_lower_keys))
        required_keys.update(zip(beta_lower_keys, sma_upper_keys))
        required_keys.update(zip(beta_upper_keys, sma_lower_keys))
        required_keys.update(zip(beta_upper_keys, sma_upper_keys))

        albedo_lookup = {}
        for key in required_keys:
            if key not in self._albedo_table_cache:
                beta_label, sma_label = key
                file_path = directory.joinpath(
                    f"alb_phi{beta_label}_sma{sma_label}.dat"
                )
                with file_path.open("r", encoding="utf-8") as handle:
                    data = np.loadtxt(handle, dtype=float)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                self._albedo_table_cache[key] = (
                    data[:, 0],
                    data[:, 1],
                )
            wavelengths, albedos = self._albedo_table_cache[key]
            if wl_dependency is None:
                throughput = mode["bandpass"](wavelengths * u.um).value
            else:
                throughput = mode["bandpass_wl"]["bin" + str(wl_dependency)](
                    wavelengths * u.um
                ).value
            denom = np.trapz(throughput, wavelengths)
            if denom > 0 and np.isfinite(denom):
                numerator = np.trapz(albedos * throughput, wavelengths)
                albedo_lookup[key] = numerator / denom
            else:
                warnings.warn(
                    "Albedo table bandpass has zero throughput; falling back to "
                    "interpolation at the mode wavelength.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                albedo_lookup[key] = np.interp(
                    lam_value,
                    wavelengths,
                    albedos,
                    left=albedos[0],
                    right=albedos[-1],
                )

        count = beta_flat.size
        albedo_ll = np.empty(count)
        albedo_lu = np.empty(count)
        albedo_ul = np.empty(count)
        albedo_uu = np.empty(count)

        for idx in range(count):
            bl_key = beta_lower_keys[idx]
            bu_key = beta_upper_keys[idx]
            al_key = sma_lower_keys[idx]
            au_key = sma_upper_keys[idx]

            albedo_ll[idx] = albedo_lookup[(bl_key, al_key)]
            albedo_lu[idx] = albedo_lookup[(bl_key, au_key)]
            albedo_ul[idx] = albedo_lookup[(bu_key, al_key)]
            albedo_uu[idx] = albedo_lookup[(bu_key, au_key)]

        with np.errstate(divide="ignore", invalid="ignore"):
            a_weight = np.divide(
                sma_clipped - sma_lower_vals,
                sma_upper_vals - sma_lower_vals,
                out=np.zeros_like(sma_clipped),
                where=(sma_upper_vals > sma_lower_vals),
            )
            beta_weight = np.divide(
                beta_clipped - beta_lower_vals,
                beta_upper_vals - beta_lower_vals,
                out=np.zeros_like(beta_clipped),
                where=(beta_upper_vals > beta_lower_vals),
            )

        lower_interp = albedo_ll + (albedo_lu - albedo_ll) * a_weight
        upper_interp = albedo_ul + (albedo_uu - albedo_ul) * a_weight
        albedo_flat = lower_interp + (upper_interp - lower_interp) * beta_weight

        albedo_array = albedo_flat.reshape(beta_array.shape)
        if beta_scalar and sma_scalar:
            return float(albedo_flat[0])

        return albedo_array
