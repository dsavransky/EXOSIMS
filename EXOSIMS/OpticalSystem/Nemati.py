# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
import astropy.units as u
import numpy as np
from scipy.optimize import minimize_scalar
from tqdm import tqdm


class Nemati(OpticalSystem):
    """Nemati Optical System class

    This class contains all variables and methods necessary to perform
    Optical System Module calculations in exoplanet mission simulation using
    the model from Nemati 2014.

    Args:
        specs:
            user specified values

    """

    def __init__(self, **specs):

        OpticalSystem.__init__(self, **specs)

    def calc_intTime(self, TL, sInds, fZ, fEZ, dMag, WA, mode, TK=None):
        """Finds integration times of target systems for a specific observing
        mode (imaging or characterization), based on Nemati 2014 (SPIE).

        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
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
        C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dMag, WA, mode, TK=TK)

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
        self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None
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
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light for each star in sInds
                in units of 1/arcsec2
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

        # cast sInds, WA, fZ, fEZ, and intTimes to arrays
        sInds = np.array(sInds, ndmin=1, copy=False)
        WA = np.array(WA.value, ndmin=1) * WA.unit
        fZ = np.array(fZ.value, ndmin=1) * fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1) * fEZ.unit
        intTimes = np.array(intTimes.value, ndmin=1) * intTimes.unit
        assert len(intTimes) == len(sInds), "intTimes and sInds must be same length"
        assert len(fEZ) == len(sInds), "fEZ must be an array of length len(sInds)"
        assert len(fZ) == len(sInds), "fZ must be an array of length len(sInds)"
        assert len(WA) == len(sInds), "WA must be an array of length len(sInds)"

        # get scienceInstrument and starlightSuppressionSystem
        inst = mode["inst"]
        syst = mode["syst"]

        # get mode wavelength
        lam = mode["lam"]
        # get mode fractional bandwidth
        BW = mode["BW"]  # noqa: F841
        # get mode bandwidth (including any IFS spectral resolving power)
        deltaLam = (
            lam / inst["Rs"] if "spec" in inst["name"].lower() else mode["deltaLam"]
        )

        # get star magnitude
        mV = TL.starMag(sInds, lam)

        # get signal to noise ratio
        SNR = mode["SNR"]

        # spectral flux density = F0 * A * Dlam * QE * T (attenuation due to optics)
        attenuation = inst["optics"] * syst["optics"]
        F_0 = TL.starF0(sInds, mode)
        C_F0 = F_0 * self.pupilArea * deltaLam * inst["QE"](lam) * attenuation

        # get core_thruput
        core_thruput = syst["core_thruput"](lam, WA)

        # get planet delta magnitude for calculation of Cb, which will be refined later
        rough_dMag = np.repeat(25, len(sInds))

        if (C_b is None) or (C_sp is None):
            _, C_b, C_sp = self.Cp_Cb_Csp(
                TL, sInds, fZ, fEZ, rough_dMag, WA, mode, TK=TK
            )
        intTimes[intTimes.value < 0.0] = 0.0
        tmp = np.nan_to_num(C_b / intTimes)
        assert all(tmp + C_sp**2.0 >= 0.0), "Invalid value in Nemati sqrt, "
        dMag = -2.5 * np.log10(
            (
                SNR
                * np.sqrt(tmp + C_sp**2.0)
                / (C_F0 * 10.0 ** (-0.4 * mV) * core_thruput * inst["PCeff"])
            )
            .decompose()
            .value
        )
        dMag[
            np.where(np.isnan(dMag))[0]
        ] = 0.0  # this is an error catch. if intTimes = 0, the dMag becomes infinite
        rough_dMags = dMag
        # Because Cb is a function of dMag, the rough dMags may be off by
        # ~10^-2, but it is useful as a center point for root-finding brackets

        dMags = np.zeros((len(sInds)))
        disable_bar = len(intTimes) == 1
        for i, int_time in enumerate(
            tqdm(
                intTimes,
                desc="Calculating dMag based on integration time",
                disable=disable_bar,
            )
        ):
            args = (
                TL,
                sInds[i],
                [fZ[i].value] * fZ.unit,
                [fEZ[i].value] * fEZ.unit,
                [WA[i].value] * WA.unit,
                mode,
                TK,
                [int_time.value] * int_time.unit,
            )
            dMag_min_res = minimize_scalar(
                self.dMag_per_intTime_obj,
                args=args,
                method="bounded",
                bounds=(rough_dMags[i] - 0.1, rough_dMags[i] + 0.1),
                options={"xatol": 1e-8, "disp": 0},
            )

            # Some times minimize_scalar returns the x value in an
            # array and sometimes it doesn't, idk why
            if isinstance(dMag_min_res["x"], np.ndarray):
                dMag = dMag_min_res["x"][0]
            else:
                dMag = dMag_min_res["x"]

            dMags[i] = dMag
        return np.array(dMags)

    def ddMag_dt(
        self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None
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
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light for each star in sInds
                in units of 1/arcsec2
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
        sInds = np.array(sInds, ndmin=1, copy=False)
        WA = np.array(WA.value, ndmin=1) * WA.unit
        fZ = np.array(fZ.value, ndmin=1) * fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1) * fEZ.unit
        intTimes = np.array(intTimes.value, ndmin=1) * intTimes.unit
        assert len(intTimes) == len(sInds), "intTimes and sInds must be same length"
        assert len(fEZ) == len(sInds), "fEZ must be an array of length len(sInds)"
        assert len(fZ) == len(sInds), "fZ must be an array of length len(sInds)"
        assert len(WA) == len(sInds), "WA must be an array of length len(sInds)"

        rough_dMag = np.zeros(len(sInds)) + 25.0
        if (C_b is None) or (C_sp is None):
            _, C_b, C_sp = self.Cp_Cb_Csp(
                TL, sInds, fZ, fEZ, rough_dMag, WA, mode, TK=TK
            )
        ddMagdt = (
            2.5
            / (2.0 * np.log(10.0))
            * (C_b / (C_b * intTimes + (C_sp * intTimes) ** 2.0)).to("1/s").value
        )

        return ddMagdt / u.s
