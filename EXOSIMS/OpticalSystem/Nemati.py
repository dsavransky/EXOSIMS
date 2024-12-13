# -*- coding: utf-8 -*-
import warnings

import astropy.units as u
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar, minimize
from tqdm import tqdm
from EXOSIMS.util._numpy_compat import copy_if_needed


from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem


class Nemati(OpticalSystem):
    r"""Nemati Optical System class

    Optical System Module based on [Nemati2014]_.

    Args:
        CIC (float):
            Default clock-induced-charge (in electrons/pixel/read).  Only used
            when not set in science instrument definition. Defaults to 1e-3
        radDos (float):
            Default radiation dose.   Only used when not set in mode definition.
            Specific defintion depends on particular optical system. Defaults to 0.
        PCeff (float):
            Default photon counting efficiency.  Only used when not set
            in science instrument definition. Defaults to 0.8
        ENF (float):
            Default excess noise factor.  Only used when not set
            in science instrument definition. Defaults to 1.
        ref_dMag (float):
            Reference star :math:`\Delta\mathrm{mag}` for reference differential
            imaging.  Defaults to 3.  Unused if ``ref_Time`` input is 0
        ref_Time (float):
            Faction of time used on reference star imaging. Must be between 0 and 1.
            Defaults to 0
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        default_vals_extra (dict):
            Dictionary of input values to be filled in as defaults in the instrument,
            starlight supporession system and observing modes. These values are specific
            to this module.
        ref_dMag (float):
            Reference star :math:`\Delta\mathrm{mag}` for reference differential
            imaging. Unused if ``ref_Time`` input is 0
        ref_Time (float):
            Faction of time used on reference star imaging.

    """

    def __init__(
        self, CIC=1e-3, radDos=0, PCeff=0.8, ENF=1, ref_dMag=3, ref_Time=0, **specs
    ):
        self.ref_dMag = float(ref_dMag)  # reference star dMag for RDI
        self.ref_Time = float(ref_Time)  # fraction of time spent on ref star for RDI

        # package inputs for use in popoulate*_extra
        self.default_vals_extra = {
            "CIC": CIC,
            "radDos": radDos,
            "PCeff": PCeff,
            "ENF": ENF,
        }

        # call upstream init
        OpticalSystem.__init__(self, **specs)

        # add local defaults to outspec
        for k in self.default_vals_extra:
            self._outspec[k] = self.default_vals_extra[k]

    def populate_scienceInstruments_extra(self):
        """Add Nemati-specific keywords to scienceInstruments"""
        newatts = [
            "CIC",  # clock-induced-charge
            "ENF",  # excess noise factor
            "PCeff",  # photon counting efficiency
        ]
        self.allowed_scienceInstrument_kws += newatts

        for ninst, inst in enumerate(self.scienceInstruments):
            for att in newatts:
                inst[att] = float(inst.get(att, self.default_vals_extra[att]))
                self._outspec["scienceInstruments"][ninst][att] = inst[att]

    def populate_observingModes_extra(self):
        """Add Nemati-specific observing mode keywords"""

        self.allowed_observingMode_kws.append("radDos")

        for nmode, mode in enumerate(self.observingModes):
            # radiation dosage, goes from 0 (beginning of mission) to 1 (end of mission)
            mode["radDos"] = float(
                mode.get("radDos", self.default_vals_extra["radDos"])
            )
            self._outspec["observingModes"][nmode]["radDos"] = mode["radDos"]

    def Cp_Cb_Csp(self, TL, sInds, fZ, JEZ, dMag, WA, mode, returnExtra=False, TK=None):
        """Calculates electron count rates for planet signal, background noise,
        and speckle residuals.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            dMag (~numpy.ndarray(float)):
                Differences in magnitude between planets and their host star
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            returnExtra (bool):
                Optional flag, default False, set True to return additional rates for
                validation
            TK (:ref:`TimeKeeping`, optional):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.


        Returns:
            tuple:
                C_p (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Planet signal electron count rate in units of 1/s
                C_b (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Background noise electron count rate in units of 1/s
                C_sp (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Residual speckle spatial structure (systematic error)
                    in units of 1/s

        """
        # grab all count rates
        C_star, C_p0, C_sr, C_z, C_ez, C_dc, C_bl, Npix = self.Cp_Cb_Csp_helper(
            TL, sInds, fZ, JEZ, dMag, WA, mode
        )

        inst = mode["inst"]

        # exposure time
        if self.texp_flag:
            with np.errstate(divide="ignore", invalid="ignore"):
                texp = 1 / C_p0 / 10  # Use 1/C_p0 as frame time for photon counting
        else:
            texp = inst["texp"]
        # readout noise
        C_rn = Npix * inst["sread"] / texp

        # clock-induced-charge
        C_cc = Npix * inst["CIC"] / texp

        # C_p = PLANET SIGNAL RATE
        # photon counting efficiency
        PCeff = inst["PCeff"]
        # radiation dosage
        radDos = mode["radDos"]
        # photon-converted 1 frame (minimum 1 photon)
        # there may be zeros in the denominator. Suppress the resulting warning:
        with np.errstate(divide="ignore", invalid="ignore"):
            phConv = np.clip(
                ((C_p0 + C_sr + C_z + C_ez) / Npix * texp).decompose().value, 1, None
            )
        # net charge transfer efficiency
        with np.errstate(invalid="ignore"):
            NCTE = 1.0 + (radDos / 4.0) * 0.51296 * (np.log10(phConv) + 0.0147233)
        # planet signal rate
        C_p = C_p0 * PCeff * NCTE
        # possibility of Npix=0 may lead C_p to be nan.  Change these to zero instead.
        C_p[np.isnan(C_p)] = 0 / u.s

        # C_b = NOISE VARIANCE RATE
        # corrections for Ref star Differential Imaging e.g. dMag=3 and 20% time on ref
        # k_SZ for speckle and zodi light, and k_det for detector
        k_SZ = (
            1.0 + 1.0 / (10 ** (0.4 * self.ref_dMag) * self.ref_Time)
            if self.ref_Time > 0
            else 1.0
        )
        k_det = 1.0 + self.ref_Time
        # calculate Cb
        ENF2 = inst["ENF"] ** 2
        C_b = k_SZ * ENF2 * (C_sr + C_z + C_ez + C_bl) + k_det * (
            ENF2 * (C_dc + C_cc) + C_rn
        )
        # for characterization, Cb must include the planet
        if not (mode["detectionMode"]):
            C_b = C_b + ENF2 * C_p0
            C_sp = C_sr * TL.PostProcessing.ppFact_char(WA) * self.stabilityFact
        else:
            # C_sp = spatial structure to the speckle including post-processing
            #        contrast factor and stability factor
            C_sp = C_sr * TL.PostProcessing.ppFact(WA) * self.stabilityFact

        if returnExtra:
            # organize components into an optional fourth result
            C_extra = dict(
                C_sr=C_sr.to("1/s"),
                C_z=C_z.to("1/s"),
                C_ez=C_ez.to("1/s"),
                C_dc=C_dc.to("1/s"),
                C_cc=C_cc.to("1/s"),
                C_rn=C_rn.to("1/s"),
                C_star=C_star.to("1/s"),
                C_p0=C_p0.to("1/s"),
                C_bl=C_bl.to("1/s"),
                Npix=Npix,
            )
            return C_p.to("1/s"), C_b.to("1/s"), C_sp.to("1/s"), C_extra
        else:
            return C_p.to("1/s"), C_b.to("1/s"), C_sp.to("1/s")

    def calc_intTime(self, TL, sInds, fZ, JEZ, dMag, WA, mode, TK=None):
        """Finds integration times of target systems for a specific observing
        mode (imaging or characterization), based on Nemati 2014 (SPIE).

        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            dMag (float ndarray):
                Differences in magnitude between planets and their host star
            WA (astropy Quantity array):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            TK (TimeKeeping object):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.

        Returns:
            intTime (astropy Quantity array):
                Integration times in units of day

        """

        # electron counts
        C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, JEZ, dMag, WA, mode, TK=TK)

        # get SNR threshold
        SNR = mode["SNR"]
        # calculate integration time based on Nemati 2014
        with np.errstate(divide="ignore", invalid="ignore"):
            if mode["syst"]["occulter"] is False:
                intTime = np.true_divide(
                    SNR**2.0 * C_b, (C_p**2.0 - (SNR * C_sp) ** 2.0)
                ).to("day")
            else:
                intTime = np.true_divide(SNR**2.0 * C_b, (C_p**2.0)).to("day")
        # infinite and NAN are set to zero
        intTime[np.isinf(intTime) | np.isnan(intTime)] = np.nan
        # negative values are set to zero
        intTime[intTime.value < 0.0] = np.nan

        return intTime

    def calc_dMag_per_intTime(
        self, intTimes, TL, sInds, fZ, JEZ, WA, mode, C_b=None, C_sp=None, TK=None
    ):
        """Finds achievable dMag for one integration time per star in the input
        list at one working angle.

        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light for each star in sInds
                in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (astropy Quantity array):
                Working angle for each star in sInds in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            TK (TimeKeeping object):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.

        Returns:
            dMag (ndarray):
                Achievable dMag for given integration time and working angle

        """
        # Calculate the analytic values for the dMag and the singularity
        # dMag value (which functions as a bound)
        dMag_x0s, sing_x0s = self.dMag_per_intTime_x0(
            TL, sInds, fZ, JEZ, WA, mode, TK, intTimes
        )
        # Loop over every star given and numerically refine the dMag
        dMags = np.zeros(len(sInds))
        for i, int_time in enumerate(tqdm(intTimes, delay=2)):
            if int_time == 0:
                warnings.warn(
                    "calc_dMag_per_int_time got an intTime=0 input, nan returned"
                )
                dMags[i] = np.nan
                continue
            if (WA[i] > mode["OWA"]) or (WA[i] < mode["IWA"]):
                warnings.warn(
                    "calc_dMag_per_int_time got WA not in [IWA, OWA], nan returned"
                )
                dMags[i] = np.nan
                continue
            # Parameters for this star
            s = [sInds[i]]

            args_denom = (TL, s, fZ[i].ravel(), JEZ[i].ravel(), WA[i].ravel(), mode, TK)
            args_intTime = (*args_denom, int_time.ravel())

            # Refine the singularity dMag value with root finding
            singularity_res = root_scalar(
                self.int_time_denom_obj,
                x0=sing_x0s[i],
                args=args_denom,
                bracket=[0, 50],
            )
            singularity_dMag = singularity_res.root
            if int_time == np.inf:
                dMag = singularity_dMag
            else:
                # Adjust the lower bounds until we have proper convergence
                star_vmag = TL.Vmag[sInds[i]]
                test_lb_subractions = [2, 10]
                converged = False
                for j, lb_subtraction in enumerate(test_lb_subractions):
                    initial_lower_bound = np.clip(
                        singularity_dMag - lb_subtraction - star_vmag,
                        5,  # Lower bound (of the lower bound) of 5
                        dMag_x0s[i] - 2,  # Upper bound of 2 under the analytic value
                    )
                    lb_adjustment = 0
                    while not converged:
                        dMag_lb = initial_lower_bound + lb_adjustment

                        if dMag_lb > singularity_dMag:
                            if j == len(test_lb_subractions):
                                raise ValueError(
                                    (
                                        "No dMag convergence for"
                                        f" {mode["instName"]}, sInds {sInds[i]}, "
                                        f"int_times {int_time}, and WA {WA[i]}"
                                    )
                                )
                            else:
                                break
                        dMag_min_res = minimize(
                            self.dMag_per_intTime_obj,
                            dMag_x0s[i],
                            args=args_intTime,
                            bounds=[(dMag_lb, singularity_dMag)],
                            method="SLSQP",
                        )

                        # Some times minimize_scalar returns the x value in an
                        # array and sometimes it doesn't, idk why
                        if isinstance(dMag_min_res["x"], np.ndarray):
                            dMag = dMag_min_res["x"][0]
                        else:
                            dMag = dMag_min_res["x"]

                        # Check if the returned time difference is greater than 5%
                        # of the true int time, if it is then raise the lower bound
                        # and try again. Also, if it converges to the lower bound
                        # then raise the lower bound and try again
                        time_diff = dMag_min_res["fun"]
                        if (time_diff > int_time.to(u.day).value / 20) or (
                            np.abs(dMag - dMag_lb) < 0.01
                        ):
                            lb_adjustment += 1
                        else:
                            converged = True
            dMags[i] = dMag
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        errs = np.abs(dMags - dMag_x0s)
        perc_errs = errs * 100 / dMags

        fig = plt.figure(figsize=(9, 4.5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

        # Left subplot
        ax0 = fig.add_subplot(gs[0])
        ax0.hist(perc_errs, bins=100)
        ax0.set_xlabel("$\Delta$mag$_0$ Percent Error (%)")
        ax0.set_ylabel("Star Count")

        # Right subplot
        ax1 = fig.add_subplot(gs[1])
        scatter = ax1.scatter(dMags, perc_errs, c=TL.Vmag, cmap="viridis")
        ax1.set_xlabel("$\Delta$mag$_0$")
        ax1.set_ylabel("$\Delta$mag$_0$ Percent Error (%)")

        # Colorbar
        cbar_ax = fig.add_subplot(gs[2])
        cbar = fig.colorbar(scatter, cax=cbar_ax, label="Vmag")

        # Add a super title
        fig.suptitle("Analytic vs Numerical $\Delta$mag$_0$ (Full Roman ETC)")
        plt.savefig("Error_hist.png", dpi=300)
        # fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        # axs[0].hist(perc_errs, bins=100)
        # axs[0].set_xlabel("$\Delta$mag$_0$ Percent Error (%)")
        # axs[0].set_ylabel("Star Count")
        #
        # axs[1].scatter(dMags, perc_errs, c=TL.Vmag)
        # axs[1].set_xlabel("$\Delta$mag$_0$")
        # axs[1].set_ylabel("$\Delta$mag$_0$ Percent Error (%)")
        # # Adjust the subplots and add a colorbar
        # fig.subplots_adjust(right=0.8)
        # fig.colorbar(axs[1].collections[0], ax=axs[1], label="Vmag")
        # fig.suptitle("Analytic vs Numerical $\Delta$mag$_0$ (Full Roman ETC)")
        # axs[0].set_title("Percent Error Histogram")
        breakpoint()
        # plt.hist(perc_errs, bins=100)

        return dMags

    def ddMag_dt(
        self, intTimes, TL, sInds, fZ, JEZ, WA, mode, C_b=None, C_sp=None, TK=None
    ):
        """Finds derivative of achievable dMag with respect to integration time

        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light for each star in sInds
                in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (astropy Quantity array):
                Working angle for each star in sInds in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            TK (TimeKeeping object):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.

        Returns:
            ddMagdt (ndarray):
                Derivative of achievable dMag with respect to integration time

        """

        # cast sInds, WA, fZ, fEZ, and intTimes to arrays
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        WA = np.array(WA.value, ndmin=1) * WA.unit
        fZ = np.array(fZ.value, ndmin=1) * fZ.unit
        JEZ = np.array(JEZ.value, ndmin=1) * JEZ.unit
        intTimes = np.array(intTimes.value, ndmin=1) * intTimes.unit
        assert len(intTimes) == len(sInds), "intTimes and sInds must be same length"
        assert len(JEZ) == len(sInds), "JEZ must be an array of length len(sInds)"
        assert len(fZ) == len(sInds), "fZ must be an array of length len(sInds)"
        assert len(WA) == len(sInds), "WA must be an array of length len(sInds)"

        rough_dMag = np.zeros(len(sInds)) + 25.0
        if (C_b is None) or (C_sp is None):
            _, C_b, C_sp = self.Cp_Cb_Csp(
                TL, sInds, fZ, JEZ, rough_dMag, WA, mode, TK=TK
            )
        ddMagdt = (
            2.5
            / (2.0 * np.log(10.0))
            * (C_b / (C_b * intTimes + (C_sp * intTimes) ** 2.0)).to("1/s").value
        )

        return ddMagdt / u.s

    def dMag_per_intTime_obj(self, dMag, *args):
        """
        Objective function for calc_dMag_per_intTime's minimize_scalar function
        that uses calc_intTime from Nemati and then compares the value to the
        true intTime value

        Args:
            dMag (~numpy.ndarray(float)):
                dMag being tested
            *args:
                all the other arguments that calc_intTime needs

        Returns:
            ~numpy.ndarray(float):
                Absolute difference between true and evaluated integration time in days.
        """
        TL, sInds, fZ, JEZ, WA, mode, TK, true_intTime = args
        est_intTime = self.calc_intTime(TL, sInds, fZ, JEZ, dMag, WA, mode, TK)
        abs_diff = np.abs(true_intTime.to("day").value - est_intTime.to("day").value)
        return abs_diff

    def calc_saturation_dMag(self, TL, sInds, fZ, JEZ, WA, mode, TK=None):
        """
        This calculates the delta magnitude for each target star that
        corresponds to an infinite integration time.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            TK (:ref:`TimeKeeping`, optional):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.

        Returns:
            ~numpy.ndarray(float):
                Maximum achievable dMag for  each target star
        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        _, sing_x0s = self.dMag_per_intTime_x0(
            TL, sInds, fZ, JEZ, WA, mode, TK, np.ones(len(sInds)) * u.d
        )

        # TODO: revisit this if updating occulter noise floor model
        if mode["syst"].get("occulter"):
            saturation_dMag = np.full(shape=len(sInds), fill_value=np.inf)
        else:
            saturation_dMag = np.zeros(len(sInds))
            for i, sInd in enumerate(tqdm(sInds, desc="Calculating saturation_dMag")):
                args = (
                    TL,
                    [sInd],
                    fZ[i].ravel(),
                    JEZ[i].ravel(),
                    WA[i].ravel(),
                    mode,
                    TK,
                )
                singularity_res = root_scalar(
                    self.int_time_denom_obj, x0=sing_x0s[i], args=args, bracket=[0, 50]
                )
                singularity_dMag = singularity_res.root
                saturation_dMag[i] = singularity_dMag

        return saturation_dMag

    def int_time_denom_obj(self, dMag, *args):
        """
        Objective function for calc_dMag_per_intTime's calculation of the root
        of the denominator of calc_inTime to determine the upper bound to use
        for minimizing to find the correct dMag

        Args:
            dMag (~numpy.ndarray(float)):
                dMag being tested
            *args:
                all the other arguments that calc_intTime needs

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Denominator of integration time expression
        """
        TL, sInds, fZ, JEZ, WA, mode, TK = args
        C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, JEZ, dMag, WA, mode, TK=TK)
        denom = C_p.decompose().value ** 2 - (mode["SNR"] * C_sp.decompose().value) ** 2
        return denom

    def dMag_per_intTime_x0(self, TL, sInds, fZ, JEZ, WA, mode, TK, intTime):
        """
        This calculates the initial guess for the dMag for each target star
        that corresponds to an infinite integration time.
        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            TK (:ref:`TimeKeeping`, optional):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.
            intTime (astropy Quantity array):
                Integration time
        Returns:
            tuple:
                ~numpy.ndarray(float):
                    Initial guess for dMag for each target star
                ~numpy.ndarray(float):
                    Initial guess for dMag for each target star that corresponds
                    to an infinite integration time
        """
        inst = mode["inst"]
        syst = mode["syst"]
        lam = mode["lam"]
        tmp_dMags = np.full(len(sInds), 25)
        Cp, Cb, Csp, C_extra = self.Cp_Cb_Csp(
            TL, sInds, fZ, JEZ, tmp_dMags, WA, mode, returnExtra=True, TK=TK
        )

        k_SZ = (
            1.0 + 1.0 / (10 ** (0.4 * self.ref_dMag) * self.ref_Time)
            if self.ref_Time > 0
            else 1.0
        )
        k_det = 1.0 + self.ref_Time
        ENF2 = inst["ENF"] ** 2
        SNR = mode["SNR"]

        Cstar = C_extra["C_star"]
        Csr = C_extra["C_sr"]
        Cz = C_extra["C_z"]
        Cez = C_extra["C_ez"]
        Cbl = C_extra["C_bl"]
        Cdc = C_extra["C_dc"]
        Ccc = C_extra["C_cc"]
        Crn = C_extra["C_rn"]
        Npix = C_extra["Npix"]

        a0 = 0
        a1 = k_SZ * ENF2 * (Csr + Cz + Cez + Cbl) + k_det * ENF2 * Cdc
        if self.texp_flag:
            # When using texp_flag the frame time is 1/(10*C_p0) which affects
            # the clock induced charge and the read noise terms
            a0 += 10 * k_det * Npix(ENF2 * inst["CIC"] + inst["sread"])
        else:
            a1 += k_det * (ENF2 * Ccc + Crn)
        if mode["detectionMode"]:
            # Account for the direct addition of the planet signal to the
            # background noise
            a0 += ENF2

        # Calculating terms necessary for the inversion
        radDos = mode["radDos"]
        with np.errstate(divide="ignore", invalid="ignore"):
            # Making an approximation for the planet signal at 23 dMag
            approx_Cp0 = Cstar * 10 ** (-0.4 * 23) * syst["core_thruput"](lam, WA)
            phConv = np.clip(
                ((approx_Cp0 + Csr + Cz + Cez) / Npix * inst["texp"]).decompose().value,
                1,
                None,
            )
            # Form of phConv that includes a fainter planet signal to use for
            # the singularity calculation
            approx_Cp_inf = Cstar * 10 ** (-0.4 * 30) * syst["core_thruput"](lam, WA)
            phConv_inf = np.clip(
                ((approx_Cp_inf + Csr + Cz + Cez) / Npix * inst["texp"])
                .decompose()
                .value,
                1,
                None,
            )
        # net charge transfer efficiency
        with np.errstate(invalid="ignore"):
            NCTE = 1.0 + (radDos / 4.0) * 0.51296 * (np.log10(phConv) + 0.0147233)
            NCTE_inf = 1.0 + (radDos / 4.0) * 0.51296 * (
                np.log10(phConv_inf) + 0.0147233
            )
        A = (inst["PCeff"] * NCTE * intTime / SNR) ** 2
        B = -a0 * intTime
        C = -a1 * intTime
        if not mode["syst"]["occulter"]:
            # Account for speckle noise
            C -= (Csp * intTime) ** 2
        # First root (the positive one)
        C_p01 = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)

        core_thruput = syst["core_thruput"](lam, WA)
        dMag0 = -2.5 * np.log10(C_p01 / (Cstar * core_thruput))

        # Calculate the dMag corresponding to the singularity (infinite intTime)
        C_p0inf = SNR * Csp / (inst["PCeff"] * NCTE_inf)
        if not mode["syst"]["occulter"]:
            dMag_inf = -2.5 * np.log10(C_p0inf / (Cstar * core_thruput))
        else:
            # The occulter has no speckle noise in the current forumlation
            dMag_inf = np.fill(len(sInds), np.inf)

        return dMag0, dMag_inf
