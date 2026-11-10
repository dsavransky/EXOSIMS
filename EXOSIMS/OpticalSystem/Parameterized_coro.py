import astropy.units as u
import numpy as np

from EXOSIMS.OpticalSystem.Nemati import Nemati


class Parameterized_coro(Nemati):
    """Nemati optical system with analytic parameterized coronagraph performance curves.

    In a starlight suppression system, use ``equation_ct`` for core throughput,
    ``equation_occ`` for occulter transmission, or ``equation_cc`` for core contrast.
    ``core_thruput`` and ``occ_trans`` use ``ctmax`` and ``alpha``;
    ``core_contrast`` uses ``rciwa``, ``beta``, and ``gamma``. ``IWA_Ber`` sets
    the curve transition in the system's input angle units. ``occ_trans`` is
    ``core_thruput / 0.65``, clipped between 0 and 1.

    Uses standard Nemati behavior unless an analytic equation is selected.
    """

    def populate_starlightSuppressionSystems_extra(self):
        """Register analytic coronagraph parameters."""
        super().populate_starlightSuppressionSystems_extra()
        self.allowed_starlightSuppressionSystem_kws += [
            "IWA_Ber",
            "ctmax",
            "alpha",
            "rciwa",
            "beta",
            "gamma",
        ]

    def get_coro_param(
        self,
        syst,
        param_name,
        fill=0.0,
        expected_ndim=None,
        expected_first_dim=None,
        min_val=None,
        max_val=None,
        interp_kind="linear",
        update_WAs=True,
    ):
        """Load normal coronagraph parameters or analytic curve strings."""
        token = syst.get(param_name)
        if not isinstance(token, str) or token not in {
            "equation_ct",
            "equation_occ",
            "equation_cc",
        }:
            return super().get_coro_param(
                syst,
                param_name,
                fill=fill,
                expected_ndim=expected_ndim,
                expected_first_dim=expected_first_dim,
                min_val=min_val,
                max_val=max_val,
                interp_kind=interp_kind,
                update_WAs=update_WAs,
            )

        iwa = syst["IWA_Ber"]
        if not isinstance(iwa, u.Quantity):
            iwa = iwa * syst["input_angle_unit_value"]
        iwa = iwa.to_value(u.arcsec)
        owa = syst["OWA"].to_value(u.arcsec)
        if iwa <= 0 or owa <= 0:
            raise ValueError("IWA_Ber and OWA must be positive.")
        lam0 = syst["lam"]
        step = min(0.0005, owa / 1000.0)
        if token in {"equation_ct", "equation_occ"}:
            ctmax = syst["ctmax"]
            alpha = syst["alpha"]
        else:
            rciwa = syst["rciwa"]
            beta = syst["beta"]
            gamma = syst["gamma"]

        def func(lam, s):
            # Scale working angles to the coronagraph's reference wavelength.
            lam_ratio = lam0.value / lam.to_value(lam0.unit)
            scaled_WA = np.array(
                s.to_value(u.arcsec) * lam_ratio,
                ndmin=1,
                dtype=float,
            )
            out = np.full(scaled_WA.shape, fill, dtype=float)
            valid = (scaled_WA >= step) & (scaled_WA <= owa)
            WA = scaled_WA[valid]
            D = np.empty_like(WA)
            below_iwa = WA < iwa
            if token in {"equation_ct", "equation_occ"}:
                D[below_iwa] = ctmax / 2.0 * (WA[below_iwa] / iwa) ** (-alpha)
                D[~below_iwa] = ctmax / 2.0 * (2.0 - (WA[~below_iwa] / iwa) ** alpha)
                D = np.clip(D, 0.0, None)
                if token == "equation_occ":
                    D = np.clip(D / 0.65, 0.0, 1.0)
            else:
                D[below_iwa] = rciwa * (WA[below_iwa] / iwa) ** beta
                D[~below_iwa] = rciwa * (WA[~below_iwa] / iwa) ** gamma
            out[valid] = D
            return out

        # Check the curve against the bounds required by the parent loader.
        WA = np.arange(step, owa + step, step) * u.arcsec
        D = func(lam0, WA)
        if (min_val is not None and np.any(D < min_val)) or (
            max_val is not None and np.any(D > max_val)
        ):
            raise ValueError(f"{param_name} is outside its allowed range.")
        syst[param_name] = func
        return syst
