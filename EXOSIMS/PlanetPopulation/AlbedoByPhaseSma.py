import importlib.resources

from EXOSIMS.PlanetPopulation.DulzPlavchan import DulzPlavchan
import astropy.units as u
import numpy as np
from EXOSIMS.util._numpy_compat import copy_if_needed


class AlbedoByPhaseSma(DulzPlavchan):
    """Planet Population module based on occurrence rate tables from Shannon Dulz and
    Peter Plavchan.

    NOTE: This assigns constant albedo based on radius ranges. When
    ''use_spectrum'' is enabled, it can also return wavelength-dependent albedos
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
        self.ps = np.array(ps, ndmin=1, copy=copy_if_needed)
        self.Rb = np.array(Rb, ndmin=1, copy=copy_if_needed)

        #################################### newly added, new flag
        self.use_spectrum = use_spectrum
        self._beta_grid = np.round(np.linspace(0, 180, num=15), 3)
        self._sma_grid = np.round(np.linspace(0.9, 1.7, num=15), 3)
        self._albedo_table_cache = {}
        self._albedo_directory = importlib.resources.files(
            "EXOSIMS.PlanetPhysicalModel"
        ).joinpath("exosims_albedo")
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
    
    ###### new function
    def get_p_from_phi_a(self, mode, beta, a, wl_dependency=None):
        """Return the corresponding albedo for a (beta, a) pair and observing wavelength

        Args:
            mode (dict):
                Observing mode
            beta (float or Quantity):
                Planet phase angle used for the spectral lookup.
            a (float or Quantity):
                Semi-major axis used for the spectral lookup.
            wavelength (float or Quantity):
                Wavelength at which to evaluate the spectral albedo.

        Returns:
            float or ndarray:
                Albedo value corresponding to the supplied phi, a, mode[wavelength].

        """
        lam = mode["lam"].to(u.um)
        #folder that contains all of the albedo tables
        directory = self._albedo_directory
        
        # mode wavelength of interest in microns (used as fallback)
        lam_value = u.Quantity(lam).to_value(u.um)

        # check for units
        if isinstance(beta, u.Quantity):
            beta_values = beta.to_value(u.deg)
        else:
            beta_values = beta
        if isinstance(a, u.Quantity):
            sma_values = a.to_value(u.AU)
        else:
            sma_values = a

        # check whether input is scalar or array of beta, sma
        beta_scalar = np.ndim(beta_values) == 0
        sma_scalar = np.ndim(sma_values) == 0

        beta_array = np.asarray(beta_values, dtype=float)
        sma_array = np.asarray(sma_values, dtype=float)

        if beta_array.shape != sma_array.shape:
            raise ValueError("beta and a must have the same shape")

        beta_flat = beta_array.reshape(-1)
        sma_flat = sma_array.reshape(-1)

        # ensure we are in the allowable range
        beta_clipped = np.clip(beta_flat, self._beta_grid[0], self._beta_grid[-1])
        sma_clipped = np.clip(sma_flat, self._sma_grid[0], self._sma_grid[-1])

        # find the 2 values that enclose the input value of beta, sma
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

        # save as keys for caching
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
        # if it hasnt been cached yet, cache it now
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
            # get cached values, whether we just cached it now or beforehand
            # integrate against the observing bandpass to get band-averaged albedo
            wavelengths, albedos = self._albedo_table_cache[key]
            if wl_dependency is None:
                throughput = mode["bandpass"](wavelengths * u.um).value
            else:
                throughput = mode["bandpass_wl"]["bin" + str(wl_dependency)](wavelengths * u.um).value   
            denom = np.trapz(throughput, wavelengths)
            if denom > 0 and np.isfinite(denom):
                numerator = np.trapz(albedos * throughput, wavelengths)
                albedo_lookup[key] = numerator / denom
            else:
                print("WARNING, got 0 in the denominator, will interpolate albedo table", wl_dependency)
                albedo_lookup[key] = np.interp(
                    lam_value,
                    wavelengths,
                    albedos,
                    left=albedos[0],
                    right=albedos[-1],
                )
        # initialize the albedo arrays (lower and upper values for both beta and sma ranges)
        count = beta_flat.size
        albedo_ll = np.empty(count)
        albedo_lu = np.empty(count)
        albedo_ul = np.empty(count)
        albedo_uu = np.empty(count)

        # fill in the albedo arrays
        for idx in range(count):
            bl_key = beta_lower_keys[idx]
            bu_key = beta_upper_keys[idx]
            al_key = sma_lower_keys[idx]
            au_key = sma_upper_keys[idx]

            albedo_ll[idx] = albedo_lookup[(bl_key, al_key)]
            albedo_lu[idx] = albedo_lookup[(bl_key, au_key)]
            albedo_ul[idx] = albedo_lookup[(bu_key, al_key)]
            albedo_uu[idx] = albedo_lookup[(bu_key, au_key)]

        # do the weighted averaging between upper and lower values. Get the weights here.
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

        # get a single albedo array based on weighted averaging from the weights above
        lower_interp = albedo_ll + (albedo_lu - albedo_ll) * a_weight
        upper_interp = albedo_ul + (albedo_uu - albedo_ul) * a_weight
        albedo_flat = lower_interp + (upper_interp - lower_interp) * beta_weight

        albedo_array = albedo_flat.reshape(beta_array.shape)
        if beta_scalar and sma_scalar:
            return float(albedo_array[0])

        return albedo_array
