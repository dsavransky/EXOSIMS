# -*- coding: utf-8 -*-
import warnings

import astropy.units as u
import numpy as np
from scipy.optimize import minimize, root_scalar
from tqdm import tqdm
from synphot import Observation
from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
from EXOSIMS.util._numpy_compat import copy_if_needed


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

        # Strip units for computation speed
        # Underscores are used to indicate that the variable is unitless and
        # need to have units added back before returning
        _C_star = C_star.to_value(self.inv_s)
        _C_p0 = C_p0.to_value(self.inv_s)
        _C_sr = C_sr.to_value(self.inv_s)
        _C_z = C_z.to_value(self.inv_s)
        _C_ez = C_ez.to_value(self.inv_s)
        _C_dc = C_dc.to_value(self.inv_s)
        _C_bl = C_bl.to_value(self.inv_s)
        inst = mode["inst"]

        # exposure time
        if self.texp_flag:
            with np.errstate(divide="ignore", invalid="ignore"):
                texp = 1 / _C_p0 / 10  # Use 1/C_p0 as frame time for photon counting
        else:
            texp = inst["texp"].to_value(u.s)
        # readout noise
        _C_rn = Npix * inst["sread"] / texp

        # clock-induced-charge
        _C_cc = Npix * inst["CIC"] / texp

        # C_p = PLANET SIGNAL RATE
        # photon counting efficiency
        PCeff = inst["PCeff"]
        # radiation dosage
        radDos = mode["radDos"]
        # photon-converted 1 frame (minimum 1 photon)
        # there may be zeros in the denominator. Suppress the resulting warning:
        with np.errstate(divide="ignore", invalid="ignore"):
            phConv = np.clip(((_C_p0 + _C_sr + _C_z + _C_ez) / Npix * texp), 1, None)
        # net charge transfer efficiency
        with np.errstate(invalid="ignore"):
            NCTE = 1.0 + (radDos / 4.0) * 0.51296 * (np.log10(phConv) + 0.0147233)
        # planet signal rate
        _C_p = _C_p0 * PCeff * NCTE
        # possibility of Npix=0 may lead C_p to be nan.  Change these to zero instead.
        _C_p[np.isnan(_C_p)] = 0

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
        _C_b = k_SZ * ENF2 * (_C_sr + _C_z + _C_ez + _C_bl) + k_det * (
            ENF2 * (_C_dc + _C_cc) + _C_rn
        )
        # for characterization, Cb must include the planet
        if not (mode["detectionMode"]):
            _C_b = _C_b + ENF2 * _C_p0
            _C_sp = _C_sr * TL.PostProcessing.ppFact_char(WA) * self.stabilityFact
        else:
            # C_sp = spatial structure to the speckle including post-processing
            #        contrast factor and stability factor
            _C_sp = _C_sr * TL.PostProcessing.ppFact(WA) * self.stabilityFact

        if returnExtra:
            # organize components into an optional fourth result
            C_extra = dict(
                C_sr=_C_sr << self.inv_s,
                C_z=_C_z << self.inv_s,
                C_ez=_C_ez << self.inv_s,
                C_dc=_C_dc << self.inv_s,
                C_cc=_C_cc << self.inv_s,
                C_rn=_C_rn << self.inv_s,
                C_star=_C_star << self.inv_s,
                C_p0=_C_p0 << self.inv_s,
                C_bl=_C_bl << self.inv_s,
                Npix=Npix,
            )
            return _C_p << self.inv_s, _C_b << self.inv_s, _C_sp << self.inv_s, C_extra
        else:
            return _C_p << self.inv_s, _C_b << self.inv_s, _C_sp << self.inv_s
    def Cp_Cb_Csp_Cstar_wl(self, TL, ZL, sInds, fZ, JEZ, dMag, WA, mode):
        """Calculate wavelength-binned count rates for spectral DRM output.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            dMag (~numpy.ndarray(float)):
                Differences in magnitude between planets and their host star
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode

        Returns:
            tuple:
                C_star_wl (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Non-coronagraphic star count rate (1/s)
                C_p_wl (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Planet count rate (1/s)
                C_b_wl (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Background noise electron count rate in units of 1/s
                C_sp_wl (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Residual speckle spatial structure (systematic error)
                    in units of 1/s
        """

        # Wavelength-dependent conversions also depend on the observing mode bins.
        n_wl_bins = len(mode["bandpass_wl"])
        conv_key = (fZ.unit, JEZ.unit, mode["hex"], n_wl_bins)
        cache_conversions = conv_key not in self.unit_conv
        convs_added = False
        if cache_conversions:
            convs = {}
        else:
            convs = self.unit_conv[conv_key]

        # get scienceInstrument and starlightSuppressionSystem and wavelength
        inst = mode["inst"]
        syst = mode["syst"]
        lam = mode["lam"]
        _lam = lam.to_value(u.nm)
        _syst_lam = syst["lam"].to_value(u.nm)

        # coronagraph parameters
        occ_trans = syst["occ_trans"](lam, WA)
        core_thruput = syst["core_thruput"](lam, WA)
        Omega = syst["core_area"](lam, WA)

        # number of pixels per lenslet
        pixPerLens = inst["lenslSamp"] ** 2.0

        # number of detector pixels in the photometric aperture = Omega / theta^2
        # Npix = pixPerLens * (Omega / inst["pixelScale"] ** 2.0).decompose().value
        if cache_conversions or convs.get("Npix") is None:
            Npix = pixPerLens * (Omega / inst["pixelScale"] ** 2.0)
            if Npix[0].value != 0:
                convs["Npix"] = (
                    Npix[0].to_value(u.dimensionless_unscaled) / Npix[0].value
                )
                Npix = Npix.value * convs["Npix"]
                convs_added = True
        else:
            Npix = (
                pixPerLens
                * (Omega.value / inst["pixelScale"].value ** 2.0)
                * convs["Npix"]
            )

        # get stellar residual intensity in the planet PSF core
        # if core_mean_intensity is None, fall back to using core_contrast
        if syst["core_mean_intensity"] is None:
            core_contrast = syst["core_contrast"](lam, WA)
            core_intensity = core_contrast * core_thruput
        else:
            # if we're here, we're using the core mean intensity
            core_mean_intensity = syst["core_mean_intensity"](
                lam, WA, TL.diameter[sInds]
            )
            # also, if we're here, we must have a platescale defined
            # furthermore, if we're a coronagraph, we have to scale by wavelength
            scale_factor = _lam / _syst_lam if not (syst["occulter"]) else 1
            core_platescale = syst["core_platescale"] * scale_factor

            # core_intensity is the mean intensity times the number of map pixels
            if cache_conversions or convs.get("core_intensity") is None:
                core_intensity = core_mean_intensity * Omega / core_platescale**2
                if core_intensity[0].value != 0:
                    convs["core_intensity"] = (
                        core_intensity[0].to_value(u.dimensionless_unscaled)
                        / core_intensity[0].value
                    )
                    convs_added = True
            else:
                core_intensity = (
                    core_mean_intensity
                    * Omega.value
                    / core_platescale.value**2
                    * convs["core_intensity"]
                )

            # finally, if a contrast floor was set, make sure we're not violating it
            if syst["contrast_floor"] is not None:
                below_contrast_floor = (
                    core_intensity / core_thruput < syst["contrast_floor"]
                )
                core_intensity[below_contrast_floor] = (
                    syst["contrast_floor"] * core_thruput[below_contrast_floor]
                )

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)

        # Star fluxes per wavelength bin (ph/m^2/s)
        flux_star_wl = TL.starFlux(sInds, mode, wl_dependency=True)

        # ELECTRON COUNT RATES [ s^-1 ]
        # non-coronagraphic star counts
        # The bandpass is already split into bins, so do not include deltaLam here.
        QE_wl = mode["inst"]["QE"](mode["band_wavelengths"] * u.nm)
        a_eff = self.pupilArea * QE_wl * mode["attenuation"]
        if cache_conversions or convs.get("C_star_wl") is None:
            C_star_wl = flux_star_wl * a_eff
            if C_star_wl[0].value.any():
                convs["C_star_wl"] = (
                    C_star_wl[0].to_value(self.inv_s) / C_star_wl[0].value
                )
                C_star_wl = C_star_wl.value * convs["C_star_wl"] << self.inv_s
                convs_added = True
        else:
            C_star_wl = (
                flux_star_wl.value * a_eff.value * convs["C_star_wl"] << self.inv_s
            )
        _C_star_wl = C_star_wl.to_value(self.inv_s)
        band_wavelengths = mode["band_wavelengths"] * u.nm
        core_thruput_wl = np.array(
            [syst["core_thruput"](wl, WA).item() for wl in band_wavelengths]
        )
        if cache_conversions or convs.get("C_p0_wl") is None:
            C_p0_wl = C_star_wl * 10.0 ** (-0.4 * dMag) * core_thruput_wl
            if C_p0_wl[0].value.any():
                convs["C_p0_wl"] = C_p0_wl[0].to_value(self.inv_s) / C_p0_wl[0].value
                C_p0_wl = C_p0_wl.value * convs["C_p0_wl"] << self.inv_s
                convs_added = True
        else:
            C_p0_wl = (
                _C_star_wl
                * 10.0 ** (-0.4 * dMag)
                * core_thruput_wl
                * convs["C_p0_wl"]
                << self.inv_s
            )

        # starlight residual
        occ_trans_wl = np.array(
            [syst["occ_trans"](wl, WA).item() for wl in band_wavelengths]
        )
        if cache_conversions or convs.get("C_sr_wl") is None:
            C_sr_wl = C_star_wl * core_intensity
            if C_sr_wl[0].value.any():
                convs["C_sr_wl"] = C_sr_wl[0].to_value(self.inv_s) / C_sr_wl[0].value
                C_sr_wl = C_sr_wl.value * convs["C_sr_wl"] << self.inv_s
                convs_added = True
        else:
            C_sr_wl = _C_star_wl * core_intensity * convs["C_sr_wl"] << self.inv_s
        # zodiacal light
        delta_wl_bins = mode["wl_bins"][:, 1] - mode["wl_bins"][:, 0]
        bw_factor = (delta_wl_bins * u.nm / mode["bandpass"].equivwidth()).decompose()
        fZ_wl_factor = ZL.zodi_intensity_at_wavelength(
            band_wavelengths
        ) / ZL.zodi_intensity_at_wavelength(mode["lam"])
        if cache_conversions or convs.get("C_z_wl") is None:
            C_z_wl = (
                mode["F0"] * bw_factor * a_eff * fZ * fZ_wl_factor * Omega * occ_trans_wl
            )
            if C_z_wl[0].value.any():
                convs["C_z_wl"] = C_z_wl[0].to_value(self.inv_s) / C_z_wl[0].value
                C_z_wl = C_z_wl.value * convs["C_z_wl"] << self.inv_s
                convs_added = True
        else:
            C_z_wl = (
                mode["F0"].value
                * bw_factor.value
                * a_eff.value
                * fZ.value
                * fZ_wl_factor.value
                * Omega.value
                * occ_trans_wl
                * convs["C_z_wl"]
                << self.inv_s
            )

        # exozodiacal light
        dust_spectrum = TL.get_exozodi_spectrum(int(sInds[0]))
        intensity_mode = Observation(
            dust_spectrum, mode["bandpass"], force="taper"
        ).integrate()
        specific_intensity_mode = intensity_mode / mode["bandpass"].equivwidth()
        specific_intensity_mode_wl = u.Quantity(
            [
                Observation(
                    dust_spectrum, mode["bandpass_wl"][f"bin{j}"], force="taper"
                ).integrate()
                / mode["bandpass_wl"][f"bin{j}"].equivwidth()
                for j in range(n_wl_bins)
            ]
        )
        flambda_factor = specific_intensity_mode_wl / specific_intensity_mode
        if cache_conversions or convs.get("C_ez_wl") is None:
            if self.use_core_thruput_for_ez:
                C_ez_wl = (
                    JEZ * bw_factor * flambda_factor * a_eff * Omega * core_thruput_wl
                )
            else:
                C_ez_wl = (
                    JEZ * bw_factor * flambda_factor * a_eff * Omega * occ_trans_wl
                )
            if C_ez_wl[0].value.any():
                convs["C_ez_wl"] = C_ez_wl[0].to_value(self.inv_s) / C_ez_wl[0].value
                C_ez_wl = C_ez_wl.value * convs["C_ez_wl"] << self.inv_s
                convs_added = True
        else:
            if self.use_core_thruput_for_ez:
                C_ez_wl = (
                    JEZ.value
                    * bw_factor.value
                    * flambda_factor.value
                    * a_eff.value
                    * Omega.value
                    * core_thruput_wl
                    * convs["C_ez_wl"]
                    << self.inv_s
                )
            else:
                C_ez_wl = (
                    JEZ.value
                    * bw_factor.value
                    * flambda_factor.value
                    * a_eff.value
                    * Omega.value
                    * occ_trans_wl
                    * convs["C_ez_wl"]
                    << self.inv_s
                )
        # dark current
        C_dc = Npix * inst["idark"]
        # only calculate binary leak if you have a model and relevant data
        # in the targelist
        if hasattr(self, "binaryleakmodel") and all(
            hasattr(TL, attr)
            for attr in ["closesep", "closedm", "brightsep", "brightdm"]
        ):
            cseps = TL.closesep[sInds]
            cdms = TL.closedm[sInds]
            bseps = TL.brightsep[sInds]
            bdms = TL.brightdm[sInds]

            if cache_conversions:
                convs["seps"] = self.arcsec2rad
                convs["diam/lam"] = (1 * self.pupilDiam.unit / lam.unit).to_value(
                    u.dimensionless_unscaled
                )
            # don't double count where the bright star is the close star
            repinds = (cseps == bseps) & (cdms == bdms)
            bseps[repinds] = np.nan
            bdms[repinds] = np.nan

            crawleaks = self.binaryleakmodel(
                (
                    (cseps * convs["seps"])
                    / lam.value
                    * self.pupilDiam.value
                    * convs["diam/lam"]
                )
            )
            cleaks = crawleaks * 10 ** (-0.4 * cdms)
            cleaks[np.isnan(cleaks)] = 0

            brawleaks = self.binaryleakmodel(
                (
                    (bseps * convs["seps"])
                    / lam.value
                    * self.pupilDiam.value
                    * convs["diam/lam"]
                )
            )
            bleaks = brawleaks * 10 ** (-0.4 * bdms)
            bleaks[np.isnan(bleaks)] = 0

            C_bl_wl = (cleaks + bleaks) * C_star_wl * core_thruput << self.inv_s
        else:
            C_bl_wl = np.zeros(len(sInds)) << self.inv_s

        # exposure time
        if self.texp_flag:
            with np.errstate(divide="ignore", invalid="ignore"):
                texp = 1 / np.sum(C_p0_wl) / 10  # Use 1/C_p0 as frame time for photon counting
        else:
            texp = inst["texp"].to_value(u.s)
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
                (
                    (C_p0_wl.value + C_sr_wl.value + C_z_wl.value + C_ez_wl.value)
                    / Npix
                    * texp
                ),
                1,
                None,
            )
        # net charge transfer efficiency
        with np.errstate(invalid="ignore"):
            NCTE = 1.0 + (radDos / 4.0) * 0.51296 * (np.log10(phConv) + 0.0147233)
        # planet signal rate
        C_p_wl = C_p0_wl * PCeff * NCTE
        # possibility of Npix=0 may lead C_p to be nan.  Change these to zero instead.
        C_p_wl[np.isnan(C_p_wl)] = 0

        core_thruput = u.Quantity(core_thruput_wl, copy=False)
        if core_thruput.shape != C_p0_wl.shape:
            core_thruput = u.Quantity(
                np.broadcast_to(core_thruput.value, C_p_wl.shape),
                unit=core_thruput.unit,
            )

        k_SZ = (
            1.0 + 1.0 / (10 ** (0.4 * self.ref_dMag) * self.ref_Time)
            if self.ref_Time > 0
            else 1.0
        )
        k_det = 1.0 + self.ref_Time
        # calculate Cb
        ENF2 = inst["ENF"] ** 2
        C_b_wl = k_SZ * ENF2 * (C_sr_wl + C_z_wl + C_ez_wl + C_bl_wl) + k_det * (
            ENF2 * (C_dc + C_cc) + C_rn
        )

        if not (mode["detectionMode"]):
            C_b_wl = C_b_wl + ENF2 * C_p_wl
            C_sp_wl = C_sr_wl * TL.PostProcessing.ppFact_char(WA) * self.stabilityFact
        else:
            # C_sp = spatial structure to the speckle including post-processing
            #        contrast factor and stability factor
            C_sp_wl = C_sr_wl * TL.PostProcessing.ppFact(WA) * self.stabilityFact
        if cache_conversions or convs_added:
            self.unit_conv[conv_key] = convs

        return C_star_wl, C_p_wl, C_b_wl, C_sp_wl

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
        _C_p = C_p.to_value(self.inv_s)
        _C_b = C_b.to_value(self.inv_s)
        _C_sp = C_sp.to_value(self.inv_s)

        # get SNR threshold
        SNR = mode["SNR"]
        # calculate integration time based on Nemati 2014
        with np.errstate(divide="ignore", invalid="ignore"):
            if mode["syst"]["occulter"] is False:
                intTime = (
                    np.true_divide(SNR**2.0 * _C_b, (_C_p**2.0 - (SNR * _C_sp) ** 2.0))
                    * self.s2d
                )
            else:
                intTime = np.true_divide(SNR**2.0 * _C_b, (_C_p**2.0)) * self.s2d
        # infinite and NAN are set to zero
        intTime[np.isinf(intTime) | np.isnan(intTime)] = np.nan
        # negative values are set to zero
        intTime[intTime < 0.0] = np.nan

        return intTime << u.d

    def calc_dMag_per_intTime(
        self,
        intTimes,
        TL,
        sInds,
        fZ,
        JEZ,
        WA,
        mode,
        C_b=None,
        C_sp=None,
        TK=None,
        analytic_only=False,
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
            analytic_only (Bool):
                Returns the analytic solution without running root-finding

        Returns:
            dMag (ndarray):
                Achievable dMag for given integration time and working angle

        """
        # Calculate the analytic values for the dMag and the singularity
        # dMag value (which functions as a bound)
        dMag_x0s, sing_x0s = self.dMag_per_intTime_x0(
            TL, sInds, fZ, JEZ, WA, mode, TK, intTimes
        )
        if analytic_only:
            return dMag_x0s
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
            if mode["syst"]["occulter"]:
                singularity_dMag = np.inf
            else:
                try:
                    singularity_res = root_scalar(
                        self.int_time_denom_obj,
                        x0=sing_x0s[i],
                        args=args_denom,
                        bracket=[0, 50],
                    )
                except ValueError:
                    singularity_dMag = np.inf
                    dMags[i] = np.nan
                    continue
                else:
                    singularity_dMag = singularity_res.root

            if int_time == np.inf:
                dMag = singularity_dMag
            else:
                # Adjust the lower bounds until we have proper convergence
                star_vmag = TL.Vmag[sInds[i]]
                test_lb_subtractions = [2, 10]
                converged = False
                for j, lb_subtraction in enumerate(test_lb_subtractions):
                    initial_lower_bound = np.clip(
                        singularity_dMag - lb_subtraction - star_vmag,
                        5,  # Lower bound (of the lower bound) of 5
                        dMag_x0s[i] - 2,  # Upper bound of 2 under the analytic value
                    )
                    lb_adjustment = 0
                    while not converged:
                        dMag_lb = initial_lower_bound + lb_adjustment

                        if dMag_lb > singularity_dMag:
                            if j == len(test_lb_subtractions) - 1:
                                raise ValueError(
                                    (
                                        "No dMag convergence for"
                                        f" {mode['instName']}, sInds {sInds[i]}, "
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
                            method="L-BFGS-B",
                            tol=1e-10,
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
            * (C_b / (C_b * intTimes + (C_sp * intTimes) ** 2.0)).to_value(self.inv_s)
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
        abs_diff = np.abs(true_intTime.to_value(u.day) - est_intTime.to_value(u.day))
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

        # TODO: revisit this if updating occulter noise floor model
        if mode["syst"].get("occulter"):
            saturation_dMag = np.full(shape=len(sInds), fill_value=np.inf)
        else:
            saturation_dMag = np.zeros(len(sInds))
            for i, sInd in enumerate(tqdm(sInds, desc="Calculating saturation_dMag")):
                args = (
                    TL,
                    [sInd],
                    [fZ[i].value] * fZ.unit,
                    [JEZ[i].value] * JEZ.unit,
                    [WA[i].value] * WA.unit,
                    mode,
                    TK,
                )
                singularity_res = root_scalar(
                    self.int_time_denom_obj,
                    args=args,
                    method="brentq",
                    bracket=[10, 40],
                )
                singularity_dMag = singularity_res.root
                saturation_dMag[i] = singularity_dMag

        return saturation_dMag

    def int_time_denom_obj(self, dMag, *args):
        """
        Objective function for calc_dMag_per_intTime's calculation of the root
        of the denominator of calc_inTime to determine the upper bound to use
        for minimizing to find the correct dMag. Only necessary for coronagraphs.

        Args:
            dMag (~numpy.ndarray(float)):
                dMag being tested
            *args:
                all the other arguments that calc_intTime needs

        Returns:
            float:
                Denominator of integration time expression
        """
        TL, sInds, fZ, JEZ, WA, mode, TK = args
        C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, JEZ, dMag, WA, mode, TK=TK)
        denom = (
            C_p.to_value(self.inv_s) ** 2
            - (mode["SNR"] * C_sp.to_value(self.inv_s)) ** 2
        )
        return denom[0]

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

        a0 = 0
        a1 = k_SZ * ENF2 * (_Csr + _Cz + _Cez + _Cbl) + k_det * ENF2 * _Cdc
        if self.texp_flag:
            # When using texp_flag the frame time is 1/(10*C_p0) which affects
            # the clock induced charge and the read noise terms
            a0 += 10 * k_det * Npix * (ENF2 * inst["CIC"] + inst["sread"])
        else:
            a1 += k_det * (ENF2 * _Ccc + _Crn)
        if not mode["detectionMode"]:
            # Account for the direct addition of the planet signal to the
            # background noise
            a0 += ENF2

        # Calculating terms necessary for the inversion
        radDos = mode["radDos"]
        _texp = inst["texp"].to_value(u.s)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Making an approximation for the planet signal at 23 dMag
            approx_Cp0 = _Cstar * 10 ** (-0.4 * 23) * syst["core_thruput"](lam, WA)
            phConv = np.clip(
                ((approx_Cp0 + _Csr + _Cz + _Cez) / Npix * _texp),
                1,
                None,
            )
            # Form of phConv that includes a fainter planet signal to use for
            # the singularity calculation
            approx_Cp_inf = _Cstar * 10 ** (-0.4 * 30) * syst["core_thruput"](lam, WA)
            phConv_inf = np.clip(
                ((approx_Cp_inf + _Csr + _Cz + _Cez) / Npix * _texp),
                1,
                None,
            )
        # net charge transfer efficiency
        with np.errstate(invalid="ignore"):
            NCTE = 1.0 + (radDos / 4.0) * 0.51296 * (np.log10(phConv) + 0.0147233)
            NCTE_inf = 1.0 + (radDos / 4.0) * 0.51296 * (
                np.log10(phConv_inf) + 0.0147233
            )
        A = (inst["PCeff"] * NCTE * _intTime / SNR) ** 2
        B = -a0 * _intTime
        C = -a1 * _intTime
        if not mode["syst"]["occulter"]:
            # Account for speckle noise
            C -= (_Csp * _intTime) ** 2
        # First root (the positive one)
        C_p01 = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)

        core_thruput = syst["core_thruput"](lam, WA)
        dMag0 = -2.5 * np.log10(C_p01 / (_Cstar * core_thruput))

        # Calculate the dMag corresponding to the singularity (infinite intTime)
        C_p0inf = SNR * _Csp / (inst["PCeff"] * NCTE_inf)
        if not mode["syst"]["occulter"]:
            dMag_inf = -2.5 * np.log10(C_p0inf / (_Cstar * core_thruput))
        else:
            # The occulter has no speckle noise in the current formulation
            dMag_inf = np.full(len(sInds), np.inf)

        return dMag0, dMag_inf
