# -*- coding: utf-8 -*-
import astropy.units as u
import numpy as np

from EXOSIMS.OpticalSystem.Nemati import Nemati


class Nemati_ct_iwa_2xbg(Nemati):
    """Nemati optical system variant with analytic IWA curves and 2x background."""

    def Cp_Cb_Csp(self, TL, sInds, fZ, JEZ, dMag, WA, mode, returnExtra=False, TK=None):
        """Calculate count rates with the background variance doubled."""

        C_p, _, C_sp, C_extra = super().Cp_Cb_Csp(
            TL, sInds, fZ, JEZ, dMag, WA, mode, returnExtra=True, TK=TK
        )
        inst = mode["inst"]
        ENF2 = inst["ENF"] ** 2
        k_SZ = (
            1.0 + 1.0 / (10 ** (0.4 * self.ref_dMag) * self.ref_Time)
            if self.ref_Time > 0
            else 1.0
        )
        k_det = 1.0 + self.ref_Time

        background = k_SZ * ENF2 * (
            C_extra["C_sr"] + C_extra["C_z"] + C_extra["C_ez"] + C_extra["C_bl"]
        ) + k_det * (
            ENF2 * (C_extra["C_dc"] + C_extra["C_cc"]) + C_extra["C_rn"]
        )
        C_b = 2.0 * background + ENF2 * C_extra["C_p0"]

        if returnExtra:
            return C_p, C_b, C_sp, C_extra
        return C_p, C_b, C_sp

    def Cp_Cb_Csp_Cstar_wl(self, TL, ZL, sInds, fZ, JEZ, dMag, WA, mode):
        """Calculate wavelength-binned rates with the 2x background policy."""

        C_star_wl, C_p_wl, C_b_wl, C_sp_wl = super().Cp_Cb_Csp_Cstar_wl(
            TL, ZL, sInds, fZ, JEZ, dMag, WA, mode
        )
        ENF2 = mode["inst"]["ENF"] ** 2
        background_wl = C_b_wl
        if not mode["detectionMode"]:
            background_wl = background_wl - ENF2 * C_p_wl
        C_b_wl = 2.0 * background_wl + ENF2 * C_p_wl

        return C_star_wl, C_p_wl, C_b_wl, C_sp_wl

    def dMag_per_intTime_x0(self, TL, sInds, fZ, JEZ, WA, mode, TK, intTime):
        """Initial dMag guesses consistent with the 2x background count model."""

        inst = mode["inst"]
        syst = mode["syst"]
        lam = mode["lam"]
        tmp_dMags = np.full(len(sInds), 25)
        _, _, Csp, C_extra = self.Cp_Cb_Csp(
            TL, sInds, fZ, JEZ, tmp_dMags, WA, mode, returnExtra=True, TK=TK
        )
        _Csp = Csp.to_value(self.inv_s)

        k_SZ = (
            1.0 + 1.0 / (10 ** (0.4 * self.ref_dMag) * self.ref_Time)
            if self.ref_Time > 0
            else 1.0
        )
        k_det = 1.0 + self.ref_Time
        ENF2 = inst["ENF"] ** 2
        SNR = mode["SNR"]

        _Cstar = C_extra["C_star"].to_value(self.inv_s)
        _Csr = C_extra["C_sr"].to_value(self.inv_s)
        _Cz = C_extra["C_z"].to_value(self.inv_s)
        _Cez = C_extra["C_ez"].to_value(self.inv_s)
        _Cbl = C_extra["C_bl"].to_value(self.inv_s)
        _Cdc = C_extra["C_dc"].to_value(self.inv_s)
        _Ccc = C_extra["C_cc"].to_value(self.inv_s)
        _Crn = C_extra["C_rn"].to_value(self.inv_s)
        Npix = C_extra["Npix"]
        _intTime = intTime.to_value(u.s)

        a0 = ENF2
        a1 = 2.0 * (
            k_SZ * ENF2 * (_Csr + _Cz + _Cez + _Cbl) + k_det * ENF2 * _Cdc
        )
        if self.texp_flag:
            # Frame-time terms are proportional to C_p0 when texp = 1/(10*C_p0).
            a0 += 2.0 * 10.0 * k_det * Npix * (ENF2 * inst["CIC"] + inst["sread"])
        else:
            a1 += 2.0 * k_det * (ENF2 * _Ccc + _Crn)

        radDos = mode["radDos"]
        _texp = inst["texp"].to_value(u.s)
        with np.errstate(divide="ignore", invalid="ignore"):
            approx_Cp0 = _Cstar * 10 ** (-0.4 * 23) * syst["core_thruput"](lam, WA)
            phConv = np.clip(
                ((approx_Cp0 + _Csr + _Cz + _Cez) / Npix * _texp), 1, None
            )
            approx_Cp_inf = _Cstar * 10 ** (-0.4 * 30) * syst["core_thruput"](
                lam, WA
            )
            phConv_inf = np.clip(
                ((approx_Cp_inf + _Csr + _Cz + _Cez) / Npix * _texp), 1, None
            )

        with np.errstate(invalid="ignore"):
            NCTE = 1.0 + (radDos / 4.0) * 0.51296 * (np.log10(phConv) + 0.0147233)
            NCTE_inf = 1.0 + (radDos / 4.0) * 0.51296 * (
                np.log10(phConv_inf) + 0.0147233
            )
        A = (inst["PCeff"] * NCTE * _intTime / SNR) ** 2
        B = -a0 * _intTime
        C = -a1 * _intTime
        if not mode["syst"]["occulter"]:
            C -= (_Csp * _intTime) ** 2

        C_p01 = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)
        core_thruput = syst["core_thruput"](lam, WA)
        dMag0 = -2.5 * np.log10(C_p01 / (_Cstar * core_thruput))

        C_p0inf = SNR * _Csp / (inst["PCeff"] * NCTE_inf)
        if not mode["syst"]["occulter"]:
            dMag_inf = -2.5 * np.log10(C_p0inf / (_Cstar * core_thruput))
        else:
            dMag_inf = np.full(len(sInds), np.inf)

        return dMag0, dMag_inf

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
        """Load normal coro params, plus this variant's analytic curve strings."""

        assert param_name in syst, f"{param_name} not found in system {syst['name']}."
        if isinstance(syst[param_name], str) and syst[param_name] in {
            "equation_ct",
            "equation_occ",
            "equation_cc",
        }:
            syst[param_name] = self._build_equation_coro_param(
                syst, param_name, fill, min_val, max_val
            )
            return syst

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

    def _build_equation_coro_param(self, syst, param_name, fill, min_val, max_val):
        """Create the analytic core throughput, occ_trans, or contrast curve."""

        token = syst[param_name]
        iwa_val = self._angle_to_arcsec_value(syst["IWA_Ber"], syst)
        owa_val = self._angle_to_arcsec_value(syst["OWA"], syst)
        assert iwa_val > 0, f"IWA_Ber for system {syst['name']} must be positive."
        assert owa_val > 0, f"OWA for system {syst['name']} must be positive."

        if token in {"equation_ct", "equation_occ"}:
            ctmax = syst["ctmax"]
            alpha = syst["alpha"]

            def evaluate_curve(WA):
                D = np.empty_like(WA)
                below_iwa = WA < iwa_val
                D[below_iwa] = ctmax / 2.0 * (WA[below_iwa] / iwa_val) ** (-alpha)
                D[~below_iwa] = ctmax / 2.0 * (
                    2.0 - (WA[~below_iwa] / iwa_val) ** alpha
                )
                D = np.clip(D, 0.0, None)
                if token == "equation_occ":
                    D = np.clip(D / 0.65, 0.0, 1.0)
                return D

        else:
            rciwa = syst["rciwa"]
            beta = syst["beta"]
            gamma = syst["gamma"]

            def evaluate_curve(WA):
                D = np.empty_like(WA)
                below_iwa = WA < iwa_val
                D[below_iwa] = rciwa * (WA[below_iwa] / iwa_val) ** beta
                D[~below_iwa] = rciwa * (WA[~below_iwa] / iwa_val) ** gamma
                return D

        step = min(0.0005, owa_val / 1000.0)
        WA = np.arange(step, owa_val + step, step)
        D = evaluate_curve(WA)

        if min_val is not None:
            assert np.all(D >= min_val), (
                f"{param_name} in {syst['name']} may not have values less than "
                f"{min_val}."
            )
        if max_val is not None:
            assert np.all(D <= max_val), (
                f"{param_name} in {syst['name']} may not have values greater than "
                f"{max_val}."
            )

        lam0_val = syst["lam"].value
        lam0_unit = syst["lam"].unit

        def func(lam, s):
            lam_ratio = lam0_val / lam.to_value(lam0_unit)
            s_scaled_as = np.array(
                s.to_value(u.arcsec) * lam_ratio, ndmin=1, dtype=float
            )
            out = np.full(s_scaled_as.shape, fill, dtype=float)
            valid = (s_scaled_as >= step) & (s_scaled_as <= owa_val)
            if np.any(valid):
                out[valid] = evaluate_curve(s_scaled_as[valid])
            return out

        return func

    def _angle_to_arcsec_value(self, value, syst):
        """Return an angle input as an arcsec scalar."""

        if isinstance(value, u.Quantity):
            return value.to_value(u.arcsec)
        return (float(value) * syst["input_angle_unit_value"]).to_value(u.arcsec)
