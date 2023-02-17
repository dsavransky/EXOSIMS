# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
from EXOSIMS.util.utils import dictToSortedStr, genHexStr
from EXOSIMS.util.keyword_fun import get_all_args
import os.path
import numbers
import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import scipy.interpolate
import scipy.optimize


class OpticalSystem(object):
    r""":ref:`OpticalSystem` Prototype

    Args:
        obscurFac (float):
            Obscuration factor (fraction of primary mirror area obscured by secondary
            and spiders). Defaults to 0.1. Must be between 0 and 1.
            See :py:attr:`~EXOSIMS.Prototypes.OpticalSystem.OpticalSystem.pupilArea`
            attribute definition.
        shapeFac (float):
            Shape Factor. Determines the ellipticity of the primary mirror.
            Defaults to np.pi/4 (circular aperture). See
            :py:attr:`~EXOSIMS.Prototypes.OpticalSystem..OpticalSystem.pupilArea`
            attribute definition.
        pupilDiam (float):
            Primary mirror major diameter (in meters).  Defaults to 4.
        intCutoff (float):
            Integration time cutoff (in days).  Determines the maximum time that is
            allowed per integration, and is used to limit integration target
            :math:`\Delta\mathrm{mag}`. Defaults to 50.
        scienceInstruments (list(dict)):
            List of dicts defining all science instruments. Minimally must contain
            one science instrument. Each dictionary must minimally contain a ``name``
            keyword, which must be unique to each instrument and must include the
            substring ``imager`` (for imaging devices) or ``spectro`` (for
            spectrometers). By default, this keyword is set to
            ``[{'name': 'imager'}]``, creating a single imaging science
            instrument. Additional parameters are filled in with default values set
            by the keywords below. For more details on science instrument definitions
            see :ref:`OpticalSystem`.
        QE  (float):
            Default quantum efficiency. Only used when not set in science instrument
            definition.  Defaults to 0.9
        optics (float):
            Total attenuation due to science instrument optics.  This is the net
            attenuation due to all optics in the science instrument path after the
            primary mirror, excluding any starlight suppression system (i.e.,
            coronagraph) optics. Only used when not set in science instrument
            definition. Defaults to 0.5
        FoV (float):
            Default instrument field of view (in arcseconds). Only used when not set
            in science instrument definition. Defaults to 10
        pixelNumber (float):
            Default number of pixels across the detector. Only used when not set
            in science instrument definition. Defaults to 1000.
        pixelSize (float):
            Default pixel pitch (nominal distance between adjacent pixel centers,
            in meters). Only used when not set in science instrument definition.
            Defaults to 1e-5
        sread (float):
            Default read noise (in electrons/pixel/read).  Only used when not set
            in science instrument definition. Defaults to 1e-6
        idark (float):
            Default dark current (in electrons/pixel/s).  Only used when not set
            in science instrument definition. Defaults to 1e-4
        CIC (float):
            Default clock-induced-charge (in electrons/pixel/read).  Only used
            when not set in science instrument definition. Defaults to 1e-3
        texp (float):
            Default single exposure time (in s).  Only used when not set
            in science instrument definition. Defaults to 100
        radDos (float):
            Default radiation dose.   Only used when not set in mode definition.
            Specific defintion depends on particular optical system. Defaults to 0.
        PCeff (float):
            Default photon counting efficiency.  Only used when not set
            in science instrument definition. Defaults to 0.8
        ENF (float):
            Default excess noise factor.  Only used when not set
            in science instrument definition. Defaults to 1.
        Rs (float):
            Default spectral resolving power.   Only used when not set
            in science instrument definition. Only applies to spetrometers.
            Defaults to 50.
        lenslSamp (float):
            Default lenslet sampling (number of pixels per lenslet rows or columns).
            Only used when not set in science instrument definition. Defaults to 2
        starlightSuppressionSystems (list(dict)):
            List of dicts defining all starlight suppression systems. Minimally must
            contain one system. Each dictionary must minimally contain a ``name``
            keyword, which must be unique to each system. By default, this keyword is
            set to ``[{'name': 'coronagraph'}]``, creating a single coronagraphic
            starlight suppression system. Additional parameters are filled in with
            default values set by the keywords below. For more details on starlight
            suppression system definitions see :ref:`OpticalSystem`.
        lam (float):
            Default central wavelength of starlight suppression system (in nm).
            Only used when not set in starlight suppression system definition.
            Defaults to 500
        BW (float):
            Default fractional bandwidth. Only used when not set in starlight
            suppression system definition. Defaults to 0.2
        occ_trans (float):
            Default coronagraphic transmission. Only used when not set in starlight
            suppression system definition. Defaults to 0.2
        core_thruput (float):
            Default core throughput. Only used when not set in starlight suppression
            system definition.  Defaults to 0.1
        core_mean_intensity (float):
            Default core mean intensity. Only used when not set in starlight
            suppression system definition. Defaults to 1e-12
        core_contrast (float):
            Default core contrast. Only used when not set in starlight suppression
            system definition. Defaults to 1e-10
        contrast_floor (float, optional):
            Default contrast floor. Only used when not set in starlight suppression
            system definition. If not None, sets absolute contrast floor.
            Defaults to None
        core_platescale (float, optional):
            Default core platescale.  Only used when not set in starlight suppression
            system definition. Defaults to None
        PSF (~numpy.ndarray(float)):
            Default point spread function suppression system. Only used when not
            set in starlight suppression system definition. Defaults to np.ones((3,3))
        ohTime (float):
            Default overhead time (in days).  Only used when not set in starlight
            suppression system definition. Time is added to every observation (on
            top of observatory settling time). Defaults to 1
        observingModes (list(dict), optional):
            List of dicts defining observing modes. These are essentially combinations
            of instruments and starlight suppression systems, identified by their
            names in keywords ``instName`` and ``systName``, respectively.  One mode
            must be identified as the default detection mode (by setting keyword
            ``detectionMode`` to True in the mode definition. If None (default)
            a single observing mode is generated combining the first instrument in
            ``scienceInstruments`` with the first starlight suppression system in
            ``starlightSuppressionSystems``, and is marked as the detection mode.
            Additional parameters are filled in with default values set by the
            keywords below.  For more details on mode definitions see
            :ref:`OpticalSystem`.
        SNR (float):
            Default target signal to noise ratio.  Only used when not set in observing
            mode definition. Defaults to 5
        timeMultiplier (float):
            Default integration time multiplier.  Only used when not set in observing
            mode definition. Every integration time calculated for an observing mode
            is scaled by this factor.  For example, if an observing mode requires two
            rolls per observation (i.e., if it covers only 180 degrees of the field),
            then this quantity should be set to 2 for that mode.  However, in some cases
            (i.e., spectroscopic followup) it may not be necessary to integrate on the
            full field, in which case this quantity could be set to 1. Defaults to 1
        IWA (float):
            Default :term:`IWA` (in arcsec).  Only used when not set in starlight
            suppression system definition. Defaults to 0.1
        OWA (float):
            Default :term:`OWA` (in arcsec). Only used when not set in starlight
            suppression system definition. Defaults to numpy.Inf
        ref_dMag (float):
            Reference star :math:`\Delta\mathrm{mag}` for reference differential
            imaging.  Defaults to 3.  Unused if ``ref_Time`` input is 0
        ref_Time (float):
            Faction of time used on reference star imaging. Must be between 0 and 1.
            Defaults to 0
        stabilityFact (float):
            Stability factor. Defaults to 1
        k_samp (float):
            Default coronagraphic intrinsice sampling. Only used if not set in
            instrument definition. Defaults to 0.25.
            TODO: move to starlight suppresion system.
        kRN (float):
            Default camera read noise before EM gain or photon counting.  Only used if
            not set in instrument definition.  Defaults to 75.0
        CTE_derate (float):
            Default charge transfer efficiency derate. Only used when not set
            in science instrument definition. Defaults to 1.0
        dark_derate (float):
            Default dark current derate.  Only used when not set
            in science instrument definition. Defaults to 1.0
        refl_derate (float):
            Default reflectivity derate.  Only used when not set
            in science instrument definition. Defaults to 1.0,
        Nlensl (float):
            Total lenslets covered by PSF core.  Only used when not set
            in science instrument definition. Only applies for spectrometers.
            Defaults to 5
        lam_d (float):
            Default instrument design wavelength (in nm).  Only used when not set
            in science instrument definition. Defaults to 500
        lam_c (float):
            Default instrument critical wavelength (in nm).  Only used when not set
            in science instrument definition. Defaults to 500
        MUF_thruput (float):
            Model uncertainty factor core throughput.  Only used when not set
            in science instrument definition. Defaults to 0.91
        cachedir (str, optional):
            Full path to cachedir.
            If None (default) use default (see :ref:`EXOSIMSCACHE`)
        ContrastScenario (str):
            Contrast scenario. Defaults to "CGDesignPerf"
        koAngles_Sun (list(float)):
            Default [Min, Max] keepout angles for Sun.  Only used when not set in
            starlight suppression system definition.  Defaults to [0,180]
        koAngles_Earth (list(float)):
            Default [Min, Max] keepout angles for Earth.  Only used when not set in
            starlight suppression system definition. Defaults to [0,180]
        koAngles_Moon (list(float)):
            Default [Min, Max] keepout angles for the moon.  Only used when not set in
            starlight suppression system definition.  Defaults to [0,180]
        koAngles_Small (list(float)):
            Default [Min, Max] keepout angles for all other bodies.  Only used when
            not set in starlight suppression system definition.
            Defaults to [0,180],
        use_char_minintTime (bool):
            DEPRECATED
        binaryleakfilepath (str, optional):
            If set, full path to binary leak definition file. Defaults to None
        texp_flag (bool):
            Toggle use of planet shot noise value for frame exposure time
            (overriides instrument texp value). Defaults to False.
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        _outspec (dict):
            :ref:`sec:outspec`
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)
        haveOcculter (bool):
            One or more starlight suppresion systems are starshade-based
        intCutoff (astropy.units.quantity.Quantity):
            Maximum allowable continuous integration time.  Time units.
        IWA (astropy.units.quantity.Quantity):
            Minimum inner working angle.
        obscurFac (float):
            Obscuration factor (fraction of primary mirror area obscured by secondary
            and spiders).
        observingModes (list):
            List of dicts defining observing modes. These are essentially combinations
            of instruments and starlight suppression systems, identified by their
            names in keywords ``instName`` and ``systName``, respectively.  One mode
            must be identified as the default detection mode (by setting keyword
            ``detectionMode`` to True in the mode definition. If None (default)
            a single observing mode is generated combining the first instrument in
            ``scienceInstruments`` with the first starlight suppression system in
            ``starlightSuppressionSystems``, and is marked as the detection mode.
            Additional parameters are filled in with default values set by the
            keywords below.  For more details on mode definitions see
            :ref:`OpticalSystem`.
        OWA (astropy.units.quantity.Quantity):
            Maximum outer working angle.
        pupilArea (astropy.units.quantity.Quantity):
            Total effective pupil area:

            .. math::

                A  = (1 - F_o)F_sD^2

            where :math:`F_o` is the obscuration factor, :math:`F_s` is the shape
            factor, and :math:`D` is the pupil diameter.
        pupilDiam (astropy.units.quantity.Quantity):
            Pupil major diameter. Length units.
        ref_dMag (float):
            Reference star :math:`\Delta\mathrm{mag}` for reference differential
            imaging. Unused if ``ref_Time`` input is 0
        ref_Time (float):
            Faction of time used on reference star imaging.
        scienceInstruments (list):
            List of dicts defining all science instruments. Minimally must contain
            one science instrument. Each dictionary must minimally contain a ``name``
            keyword, which must be unique to each instrument and must include the
            substring ``imager`` (for imaging devices) or ``spectro`` (for
            spectrometers). By default, this keyword is set to
            ``[{'name': 'imager'}]``, creating a single imaging science
            instrument. Additional parameters are filled in with default values set
            by the keywords below. For more details on science instrument definitions
            see :ref:`OpticalSystem`.
        shapeFac (float):
            Primary mirror shape factor.
        stabilityFact (float):
            Telescope stability factor.
        starlightSuppressionSystems (list):
            List of dicts defining all starlight suppression systems. Minimally must
            contain one system. Each dictionary must minimally contain a ``name``
            keyword, which must be unique to each system. By default, this keyword is
            set to ``[{'name': 'coronagraph'}]``, creating a single coronagraphic
            starlight suppression system. Additional parameters are filled in with
            default values set by the keywords below. For more details on starlight
            suppression system definitions see :ref:`OpticalSystem`.
        texp_flag (bool):
            Toggle use of planet shot noise value for frame exposure time
            (overriides instrument texp value).
        use_char_minintTime (bool):
            DEPRECATED
    """

    _modtype = "OpticalSystem"

    def __init__(
        self,
        obscurFac=0.1,
        shapeFac=np.pi / 4,
        pupilDiam=4,
        intCutoff=50,
        scienceInstruments=[{"name": "imager"}],
        QE=0.9,
        optics=0.5,
        FoV=10,
        pixelNumber=1000,
        pixelSize=1e-5,
        sread=1e-6,
        idark=1e-4,
        CIC=1e-3,
        texp=100,
        radDos=0,
        PCeff=0.8,
        ENF=1,
        Rs=50,
        lenslSamp=2,
        starlightSuppressionSystems=[{"name": "coronagraph"}],
        lam=500,
        BW=0.2,
        occ_trans=0.2,
        core_thruput=0.1,
        core_mean_intensity=1e-12,
        core_contrast=1e-10,
        contrast_floor=None,
        core_platescale=None,
        PSF=np.ones((3, 3)),
        ohTime=1,
        observingModes=None,
        SNR=5,
        timeMultiplier=1.0,
        IWA=None,
        OWA=None,
        ref_dMag=3,
        ref_Time=0,
        stabilityFact=1,
        k_samp=0.25,
        kRN=75.0,
        CTE_derate=1.0,
        dark_derate=1.0,
        refl_derate=1.0,
        Nlensl=5,
        lam_d=500,
        lam_c=500,
        MUF_thruput=0.91,
        cachedir=None,
        ContrastScenario="CGDesignPerf",
        koAngles_Sun=[0, 180],
        koAngles_Earth=[0, 180],
        koAngles_Moon=[0, 180],
        koAngles_Small=[0, 180],
        use_char_minintTime=False,
        binaryleakfilepath=None,
        texp_flag=False,
        **specs
    ):

        # start the outspec
        self._outspec = {}

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

        # load all values with defaults
        self.obscurFac = float(obscurFac)  # obscuration factor (fraction of PM area)
        self.shapeFac = float(shapeFac)  # shape factor
        self.pupilDiam = float(pupilDiam) * u.m  # entrance pupil diameter
        self.intCutoff = float(intCutoff) * u.d  # integration time cutoff
        self.ref_dMag = float(ref_dMag)  # reference star dMag for RDI
        self.ref_Time = float(ref_Time)  # fraction of time spent on ref star for RDI
        self.stabilityFact = float(stabilityFact)  # stability factor for telescope
        self.use_char_minintTime = bool(use_char_minintTime)
        self.texp_flag = bool(texp_flag)

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        specs["cachedir"] = self.cachedir

        # populate outspec with all OpticalSystem scalar attributes
        for att in self.__dict__:
            if att not in [
                "vprint",
                "_outspec",
            ]:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat

        # get all inputs that aren't attributes into the outspec as well
        kws = get_all_args(self.__class__)
        ignore_kws = [
            "self",
            "scienceInstruments",
            "starlightSuppressionSystems",
            "observingModes",
        ]
        kws = list(set(kws) - set(ignore_kws))
        for kw in kws:
            if kw not in self._outspec:
                self._outspec[kw] = locals()[kw]

        # pupil collecting area (obscured PM)
        self.pupilArea = (1 - self.obscurFac) * self.shapeFac * self.pupilDiam**2

        # loop through all science Instruments (must have one defined)
        assert isinstance(scienceInstruments, list) and (
            len(scienceInstruments) > 0
        ), "No science instrument defined."
        self.scienceInstruments = scienceInstruments
        self._outspec["scienceInstruments"] = []
        for ninst, inst in enumerate(self.scienceInstruments):
            assert isinstance(
                inst, dict
            ), "Science instruments must be defined as dicts."
            assert "name" in inst and isinstance(
                inst["name"], str
            ), "All science instruments must have key name."
            # populate with values that may be filenames (interpolants)
            inst["QE"] = inst.get("QE", QE)
            self._outspec["scienceInstruments"].append(inst.copy())

            # quantum efficiency
            if isinstance(inst["QE"], str):
                pth = os.path.normpath(os.path.expandvars(inst["QE"]))
                assert os.path.isfile(pth), "%s is not a valid file." % pth
                # Check csv vs fits
                ext = pth.split(".")[-1]
                assert ext == "fits" or ext == "csv", (
                    "%s must be a fits or csv file." % pth
                )
                dat = self.get_param_data(pth)
                lam, D = (
                    (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:, 0], dat[:, 1])
                )
                assert np.all(D >= 0) and np.all(
                    D <= 1
                ), "QE must be positive and smaller than 1."
                # parameter values outside of lam
                Dinterp1 = scipy.interpolate.interp1d(
                    lam.astype(float),
                    D.astype(float),
                    kind="cubic",
                    fill_value=0.0,
                    bounds_error=False,
                )
                inst["QE"] = (
                    lambda l: np.array(Dinterp1(l.to("nm").value), ndmin=1) / u.photon
                )
            elif isinstance(inst["QE"], numbers.Number):
                assert (
                    inst["QE"] >= 0 and inst["QE"] <= 1
                ), "QE must be positive and smaller than 1."
                inst["QE"] = (
                    lambda l, QE=float(inst["QE"]): np.array([QE] * l.size, ndmin=1)
                    / u.photon
                )
            else:
                inst["QE"] = QE
                self.vprint("Anomalous input, value set to default.")

            # load detector specifications
            inst["optics"] = float(
                inst.get("optics", optics)
            )  # attenuation due to optics
            inst["FoV"] = float(inst.get("FoV", FoV)) * u.arcsec  # field of view
            inst["pixelNumber"] = int(
                inst.get("pixelNumber", pixelNumber)
            )  # array format
            inst["pixelSize"] = (
                float(inst.get("pixelSize", pixelSize)) * u.m
            )  # pixel pitch
            inst["pixelScale"] = (
                inst.get("pixelScale", 2 * inst["FoV"].value / inst["pixelNumber"])
                * u.arcsec
            )  # pixel pitch
            inst["idark"] = float(inst.get("idark", idark)) / u.s  # dark-current rate
            inst["CIC"] = float(inst.get("CIC", CIC))  # clock-induced-charge
            inst["sread"] = float(inst.get("sread", sread))  # effective readout noise
            inst["texp"] = (
                float(inst.get("texp", texp)) * u.s
            )  # exposure time per frame
            inst["ENF"] = float(inst.get("ENF", ENF))  # excess noise factor
            inst["PCeff"] = float(
                inst.get("PCeff", PCeff)
            )  # photon counting efficiency
            inst["k_samp"] = float(
                inst.get("k_samp", k_samp)
            )  # coronagraph intrinsic sampling
            inst["kRN"] = float(inst.get("kRN", kRN))  # read noise
            inst["lam_d"] = float(inst.get("lam_d", lam_d)) * u.nm  # design wavelength
            inst["lam_c"] = (
                float(inst.get("lam_c", lam_c)) * u.nm
            )  # critical wavelength
            inst["CTE_derate"] = float(
                inst.get("CTE_derate", CTE_derate)
            )  # charge transfer efficiency derate
            inst["dark_derate"] = float(
                inst.get("dark_derate", dark_derate)
            )  # dark noise derate
            inst["refl_derate"] = float(
                inst.get("refl_derate", refl_derate)
            )  # reflectivity derate
            inst["MUF_thruput"] = float(
                inst.get("MUF_thruput", MUF_thruput)
            )  # core model uncertainty throughput

            # parameters specific to spectrograph
            if "spec" in inst["name"].lower():
                # spectral resolving power
                inst["Rs"] = float(inst.get("Rs", Rs))
                # lenslet sampling, number of pixel per lenslet rows or cols
                inst["lenslSamp"] = float(inst.get("lenslSamp", lenslSamp))
                # lenslets in core
                inst["Nlensl"] = float(inst.get("Nlensl", Nlensl))
            else:
                inst["Rs"] = 1.0
                inst["lenslSamp"] = 1.0
                inst["Nlensl"] = 5.0

            # calculate focal and f-number
            inst["focal"] = (
                inst["pixelSize"].to("m") / inst["pixelScale"].to("rad").value
            )
            inst["fnumber"] = float(inst["focal"] / self.pupilDiam)

            # populate detector specifications to outspec
            for att in inst:
                if att not in ["QE"]:
                    dat = inst[att]
                    self._outspec["scienceInstruments"][ninst][att] = (
                        dat.value if isinstance(dat, u.Quantity) else dat
                    )

        # loop through all starlight suppression systems (must have one defined)
        assert isinstance(starlightSuppressionSystems, list) and (
            len(starlightSuppressionSystems) > 0
        ), "No starlight suppression systems defined."
        self.starlightSuppressionSystems = starlightSuppressionSystems
        self.haveOcculter = False
        self._outspec["starlightSuppressionSystems"] = []
        for nsyst, syst in enumerate(self.starlightSuppressionSystems):
            assert isinstance(
                syst, dict
            ), "Starlight suppression systems must be defined as dicts."
            assert "name" in syst and isinstance(
                syst["name"], str
            ), "All starlight suppression systems must have key name."
            # populate with values that may be filenames (interpolants)
            syst["occ_trans"] = syst.get("occ_trans", occ_trans)
            syst["core_thruput"] = syst.get("core_thruput", core_thruput)
            syst["core_contrast"] = syst.get("core_contrast", core_contrast)
            syst["core_mean_intensity"] = syst.get(
                "core_mean_intensity", core_thruput * core_contrast
            )
            syst["core_area"] = syst.get(
                "core_area", 0.0
            )  # if zero, will get from lam/D
            syst["PSF"] = syst.get("PSF", PSF)
            self._outspec["starlightSuppressionSystems"].append(syst.copy())

            # attenuation due to optics specific to the coronagraph (defaults to 1)
            # e.g. polarizer, Lyot stop, extra flat mirror
            syst["optics"] = float(syst.get("optics", 1.0))

            # set an occulter, for an external or hybrid system
            syst["occulter"] = syst.get("occulter", False)
            if syst["occulter"]:
                self.haveOcculter = True

            # handle inf OWA
            if syst.get("OWA") == 0:
                syst["OWA"] = np.Inf

            # when provided, always use deltaLam instead of BW (bandwidth fraction)
            syst["lam"] = float(syst.get("lam", lam)) * u.nm  # central wavelength (nm)
            syst["deltaLam"] = (
                float(
                    syst.get(
                        "deltaLam", syst["lam"].to("nm").value * syst.get("BW", BW)
                    )
                )
                * u.nm
            )  # bandwidth (nm)
            syst["BW"] = float(syst["deltaLam"] / syst["lam"])  # bandwidth fraction
            # default lam and BW updated with values from first instrument

            if nsyst == 0:
                lam, BW = syst.get("lam").value, syst.get("BW")

            # get keepout angles for specific instrument
            syst["koAngles_Sun"] = [
                float(x) for x in syst.get("koAngles_Sun", koAngles_Sun)
            ] * u.deg
            syst["koAngles_Earth"] = [
                float(x) for x in syst.get("koAngles_Earth", koAngles_Earth)
            ] * u.deg
            syst["koAngles_Moon"] = [
                float(x) for x in syst.get("koAngles_Moon", koAngles_Moon)
            ] * u.deg
            syst["koAngles_Small"] = [
                float(x) for x in syst.get("koAngles_Small", koAngles_Small)
            ] * u.deg

            # get coronagraph input parameters
            syst = self.get_coro_param(syst, "occ_trans")
            syst = self.get_coro_param(syst, "core_thruput")
            syst = self.get_coro_param(syst, "core_contrast", fill=1.0)
            syst = self.get_coro_param(syst, "core_mean_intensity")
            syst = self.get_coro_param(syst, "core_area")
            syst["core_platescale"] = syst.get("core_platescale", core_platescale)
            syst["contrast_floor"] = syst.get("contrast_floor", contrast_floor)

            # get PSF
            if isinstance(syst["PSF"], str):
                pth = os.path.normpath(os.path.expandvars(syst["PSF"]))
                assert os.path.isfile(pth), "%s is not a valid file." % pth
                with fits.open(pth) as f:
                    hdr = f[0].header  # noqa: F841
                    dat = f[0].data
                assert len(dat.shape) == 2, "Wrong PSF data shape."
                assert np.any(dat), "PSF must be != 0"
                syst["PSF"] = lambda l, s, P=dat: P
            else:
                assert np.any(syst["PSF"]), "PSF must be != 0"
                syst["PSF"] = lambda l, s, P=np.array(syst["PSF"]).astype(float): P

            # loading system specifications
            syst["IWA"] = (
                syst.get("IWA", 0.1 if IWA is None else IWA) * u.arcsec
            )  # inner WA
            syst["OWA"] = (
                syst.get("OWA", np.Inf if OWA is None else OWA) * u.arcsec
            )  # outer WA
            syst["ohTime"] = float(syst.get("ohTime", ohTime)) * u.d  # overhead time

            # populate system specifications to outspec
            for att in syst:
                if att not in [
                    "occ_trans",
                    "core_thruput",
                    "core_contrast",
                    "core_mean_intensity",
                    "core_area",
                    "PSF",
                    "F0",
                ]:
                    dat = syst[att]
                    self._outspec["starlightSuppressionSystems"][nsyst][att] = (
                        dat.value if isinstance(dat, u.Quantity) else dat
                    )

        # loop through all observing modes
        # if no observing mode defined, create a default mode
        if observingModes is None:
            inst = self.scienceInstruments[0]
            syst = self.starlightSuppressionSystems[0]
            observingModes = [
                {
                    "detectionMode": True,
                    "instName": inst["name"],
                    "systName": syst["name"],
                }
            ]
        self.observingModes = observingModes
        self._outspec["observingModes"] = []
        for nmode, mode in enumerate(self.observingModes):
            assert isinstance(mode, dict), "Observing modes must be defined as dicts."
            assert (
                "instName" in mode and "systName" in mode
            ), "All observing modes must have key instName and systName."
            assert np.any(
                [mode["instName"] == inst["name"] for inst in self.scienceInstruments]
            ), ("The mode's instrument name " + mode["instName"] + " does not exist.")
            assert np.any(
                [
                    mode["systName"] == syst["name"]
                    for syst in self.starlightSuppressionSystems
                ]
            ), ("The mode's system name " + mode["systName"] + " does not exist.")
            self._outspec["observingModes"].append(mode.copy())
            # create temporary placeholder so we don't have to look up the original
            # mode again. alphabetically sort keys to ensure generality
            mode["hex"] = dictToSortedStr(mode)

            # loading mode specifications
            mode["SNR"] = float(mode.get("SNR", SNR))
            mode["timeMultiplier"] = float(mode.get("timeMultiplier", timeMultiplier))
            mode["detectionMode"] = mode.get("detectionMode", False)
            mode["inst"] = [
                inst
                for inst in self.scienceInstruments
                if inst["name"] == mode["instName"]
            ][0]
            mode["syst"] = [
                syst
                for syst in self.starlightSuppressionSystems
                if syst["name"] == mode["systName"]
            ][0]
            # get mode wavelength and bandwidth (get system's values by default)
            # when provided, always use deltaLam instead of BW (bandwidth fraction)
            syst_lam = mode["syst"]["lam"].to("nm").value
            syst_BW = mode["syst"]["BW"]
            mode["lam"] = float(mode.get("lam", syst_lam)) * u.nm
            mode["deltaLam"] = (
                float(mode.get("deltaLam", mode["lam"].value * mode.get("BW", syst_BW)))
                * u.nm
            )
            mode["BW"] = float(mode["deltaLam"] / mode["lam"])
            # get mode IWA and OWA: rescale if the mode wavelength is different than
            # the wavelength at which the system is defined
            mode["IWA"] = mode["syst"]["IWA"]
            mode["OWA"] = mode["syst"]["OWA"]
            if mode["lam"] != mode["syst"]["lam"]:
                mode["IWA"] = mode["IWA"] * mode["lam"] / mode["syst"]["lam"]
                mode["OWA"] = mode["OWA"] * mode["lam"] / mode["syst"]["lam"]
            # radiation dosage, goes from 0 (beginning of mission) to 1 (end of mission)
            mode["radDos"] = float(mode.get("radDos", radDos))
            mode["ContrastScenario"] = mode.get("ContrastScenario", ContrastScenario)

        # check for only one detection mode
        allModes = self.observingModes
        detModes = list(filter(lambda mode: mode["detectionMode"], allModes))
        assert len(detModes) <= 1, "More than one detection mode specified."
        # if not specified, default detection mode is first imager mode
        if len(detModes) == 0:
            imagerModes = list(
                filter(lambda mode: "imag" in mode["inst"]["name"], allModes)
            )
            if imagerModes:
                imagerModes[0]["detectionMode"] = True
            # if no imager mode, default detection mode is first observing mode
            else:
                allModes[0]["detectionMode"] = True

        # populate fundamental IWA and OWA as required
        IWAs = [x.get("IWA") for x in self.observingModes if x.get("IWA") is not None]
        if IWA is not None:
            self.IWA = float(IWA) * u.arcsec
        elif IWAs:
            self.IWA = min(IWAs)
        else:
            raise ValueError("Could not determine fundamental IWA.")

        OWAs = [x.get("OWA") for x in self.observingModes if x.get("OWA") is not None]
        if OWA is not None:
            self.OWA = float(OWA) * u.arcsec if OWA != 0 else np.inf * u.arcsec
        elif OWAs:
            self.OWA = max(OWAs)
        else:
            raise ValueError("Could not determine fundamental OWA.")

        assert self.IWA < self.OWA, "Fundamental IWA must be smaller that the OWA."

        # if binary leakage model provided, let's grab that as well
        if binaryleakfilepath is not None:
            assert os.path.exists(binaryleakfilepath), (
                "Binary leakage model data file not found at %s" % binaryleakfilepath
            )

            binaryleakdata = np.genfromtxt(binaryleakfilepath, delimiter=",")

            self.binaryleakmodel = scipy.interpolate.interp1d(
                binaryleakdata[:, 0], binaryleakdata[:, 1], bounds_error=False
            )
            self._outspec["binaryleakfilepath"] = binaryleakfilepath

        # provide every observing mode with a unique identifier
        self.genObsModeHex()

    def __str__(self):
        """String representation of the Optical System object

        When the command 'print' is used on the Optical System object, this
        method will print the attribute values contained in the object

        """

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Optical System class object attributes"

    def genObsModeHex(self):
        """Generate a unique hash for every observing mode to be used in downstream
        identification and caching.

        The hash will be based on the _outspec entries for the obsmode, its science
        instrument and its starlight suppression system.
        """

        for mode in self.observingModes:
            inst = [
                inst
                for inst in self._outspec["scienceInstruments"]
                if inst["name"] == mode["instName"]
            ][0]
            syst = [
                syst
                for syst in self._outspec["starlightSuppressionSystems"]
                if syst["name"] == mode["systName"]
            ][0]

            modestr = "{},{},{}".format(
                mode["hex"], dictToSortedStr(inst), dictToSortedStr(syst)
            )

            mode["hex"] = genHexStr(modestr)

    def get_coro_param(self, syst, param_name, fill=0.0):
        """For a given starlightSuppressionSystem, this method loads an input
        parameter from a table (fits file) or a scalar value. It then creates a
        callable lambda function, which depends on the wavelength of the system
        and the angular separation of the observed planet.

        Args:
            syst (dict):
                Dictionary containing the parameters of one starlight suppression system
            param_name (str):
                Name of the parameter that must be loaded
            fill (float):
                Fill value for working angles outside of the input array definition

        Returns:
            dict:
                Updated dictionary of starlight suppression system parameters

        .. note::

            The created lambda function handles the specified wavelength by
            rescaling the specified working angle by a factor syst['lam']/mode['lam']

        .. note::

            If the input parameter is taken from a table, the IWA and OWA of that
            system are constrained by the limits of the allowed WA on that table.

        """

        assert isinstance(param_name, str), "param_name must be a string."
        if isinstance(syst[param_name], str):
            pth = os.path.normpath(os.path.expandvars(syst[param_name]))
            assert os.path.isfile(pth), "%s is not a valid file." % pth
            # Check for fits or csv file
            ext = pth.split(".")[-1]
            assert ext == "fits" or ext == "csv", "%s must be a fits or csv file." % pth
            dat = self.get_param_data(pth, left_col_name="r_as", param_name=param_name)
            WA, D = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:, 0], dat[:, 1])
            if not self.haveOcculter:
                assert np.all(D >= 0) and np.all(D <= 1), (
                    param_name + " must be positive and smaller than 1."
                )
            # table interpolate function
            Dinterp = scipy.interpolate.interp1d(
                WA.astype(float),
                D.astype(float),
                kind="cubic",
                fill_value=fill,
                bounds_error=False,
            )
            # create a callable lambda function
            syst[param_name] = lambda l, s: np.array(
                Dinterp((s * syst["lam"] / l).to("arcsec").value), ndmin=1
            )
            # IWA and OWA are constrained by the limits of the allowed WA on that table
            syst["IWA"] = max(np.min(WA), syst.get("IWA", np.min(WA)))
            syst["OWA"] = min(np.max(WA), syst.get("OWA", np.max(WA)))

        elif isinstance(syst[param_name], numbers.Number):
            if not self.haveOcculter:
                assert syst[param_name] >= 0 and syst[param_name] <= 1, (
                    param_name + " must be positive and smaller than 1."
                )
            syst[param_name] = (
                lambda lam, s, D=float(syst[param_name]): (
                    (s * syst["lam"] / lam >= syst["IWA"])
                    & (s * syst["lam"] / lam <= syst["OWA"])
                )
                * (D - fill)
                + fill
            )

        else:
            syst[param_name] = None

        return syst

    def get_param_data(self, pth, left_col_name=None, param_name=None):
        """Gets the data from a file, used primarily to create interpolants for
        coronagraph parameters

        Args:
            pth (str):
                String to file location, will also work with any other path object
            left_col_name (str):
                String that represents the left column, as in the one that will be
                inputted when getting a value from the interpolant
            param_name (str):
                String that is the header for the parameter of interest

        Returns:
            dat (~numpy.ndarray):
                Two column array with the data for the parameter

        """
        # Check for fits or csv file
        ext = pth.split(".")[-1]
        assert ext == "fits" or ext == "csv", "%s must be a fits or csv file." % pth
        if ext == "fits":
            with fits.open(pth) as f:
                dat = f[0].data
        else:
            # Need to get all of the headers and data, then associate them in the same
            # ndarray that the fits files would generate
            table_vals = np.genfromtxt(pth, delimiter=",", skip_header=1)
            if table_vals.shape[1] == 2:
                # if there are only two columns retun the data without looking at the
                # headers
                dat = table_vals
            else:
                table_headers = np.genfromtxt(
                    pth, delimiter=",", skip_footer=len(table_vals), dtype=str
                )
                left_column_location = np.where(table_headers == left_col_name)[0][0]
                param_location = np.where(table_headers == param_name)[0][0]
                dat = np.vstack(
                    [table_vals[:, left_column_location], table_vals[:, param_location]]
                ).T
            assert len(dat.shape) == 2 and 2 in dat.shape, (
                param_name + " wrong data shape."
            )
        return dat

    def Cp_Cb_Csp(self, TL, sInds, fZ, fEZ, dMag, WA, mode, returnExtra=False, TK=None):
        """Calculates electron count rates for planet signal, background noise,
        and speckle residuals.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
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

        # coronagraph parameters
        occ_trans = syst["occ_trans"](lam, WA)
        core_thruput = syst["core_thruput"](lam, WA)
        core_contrast = syst["core_contrast"](lam, WA)
        core_area = syst["core_area"](lam, WA)

        # solid angle of photometric aperture, specified by core_area (optional)
        Omega = core_area * u.arcsec**2.0
        # if zero, get omega from (lambda/D)^2
        Omega[Omega == 0] = (
            np.pi * (np.sqrt(2.0) / 2.0 * lam / self.pupilDiam * u.rad) ** 2.0
        )
        # number of pixels per lenslet
        pixPerLens = inst["lenslSamp"] ** 2.0
        # number of pixels in the photometric aperture = Omega / theta^2
        Npix = pixPerLens * (Omega / inst["pixelScale"] ** 2.0).decompose().value

        # get stellar residual intensity in the planet PSF core
        # OPTION 1: if core_mean_intensity is missing, use the core_contrast
        if syst["core_mean_intensity"] is None:
            core_intensity = core_contrast * core_thruput
        # OPTION 2A: otherwise use core_mean_intensity and adjust for contrast_floor
        elif syst["contrast_floor"] is not None:
            core_mean_intensity = syst["core_mean_intensity"](lam, WA)
            # if a platescale was specified with the coro parameters, apply correction
            if syst["core_platescale"] is not None:
                core_mean_intensity *= (
                    (
                        inst["pixelScale"]
                        / syst["core_platescale"]
                        / (lam / self.pupilDiam)
                    )
                    .decompose()
                    .value
                )
            contrast = core_mean_intensity * Npix / core_thruput
            contrast_floor = syst["contrast_floor"]
            core_intensity = core_mean_intensity * Npix
            core_intensity[np.where(contrast < contrast_floor)] = (
                contrast_floor * core_thruput[np.where(contrast < contrast_floor)]
            )
        # OPTION 2B: otherwise use core_mean_intensity
        else:
            core_mean_intensity = syst["core_mean_intensity"](lam, WA)
            # if a platesale was specified with the coro parameters, apply correction
            if syst["core_platescale"] is not None:
                core_mean_intensity *= (
                    (
                        inst["pixelScale"]
                        / syst["core_platescale"]
                        / (lam / self.pupilDiam)
                    )
                    .decompose()
                    .value
                )
            core_intensity = core_mean_intensity * Npix

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        # get star magnitude
        mV = TL.starMag(sInds, lam)

        # ELECTRON COUNT RATES [ s^-1 ]
        # spectral flux density = F0 * A * Dlam * QE * T (attenuation due to optics)
        attenuation = inst["optics"] * syst["optics"]
        F_0 = TL.starF0(sInds, mode)

        C_F0 = F_0 * self.pupilArea * deltaLam * inst["QE"](lam) * attenuation
        # planet conversion rate (planet shot)
        C_p0 = C_F0 * 10.0 ** (-0.4 * (mV + dMag)) * core_thruput
        # starlight residual
        C_sr = C_F0 * 10.0 ** (-0.4 * mV) * core_intensity
        # zodiacal light
        C_z = C_F0 * fZ * Omega * occ_trans
        # exozodiacal light
        C_ez = C_F0 * fEZ * Omega * core_thruput
        # dark current
        C_dc = Npix * inst["idark"]
        # exposure time
        if self.texp_flag:
            texp = 1 / C_p0 / 10  # Use 1/C_p0 as frame time for photon counting
        else:
            texp = inst["texp"]
        # clock-induced-charge
        C_cc = Npix * inst["CIC"] / texp
        # readout noise
        C_rn = Npix * inst["sread"] / texp

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

            # don't double count where the bright star is the close star
            repinds = (cseps == bseps) & (cdms == bdms)
            bseps[repinds] = np.nan
            bdms[repinds] = np.nan

            crawleaks = self.binaryleakmodel(
                (
                    ((cseps * u.arcsec).to(u.rad)).value / lam * self.pupilDiam
                ).decompose()
            )
            cleaks = crawleaks * 10 ** (-0.4 * cdms)
            cleaks[np.isnan(cleaks)] = 0

            brawleaks = self.binaryleakmodel(
                (
                    ((bseps * u.arcsec).to(u.rad)).value / lam * self.pupilDiam
                ).decompose()
            )
            bleaks = brawleaks * 10 ** (-0.4 * bdms)
            bleaks[np.isnan(bleaks)] = 0

            C_bl = (cleaks + bleaks) * C_F0 * 10.0 ** (-0.4 * mV) * core_thruput
        else:
            C_bl = np.zeros(len(sInds)) / u.s

        # C_p = PLANET SIGNAL RATE
        # photon counting efficiency
        PCeff = inst["PCeff"]
        # radiation dosage
        radDos = mode["radDos"]
        # photon-converted 1 frame (minimum 1 photon)
        phConv = np.clip(
            ((C_p0 + C_sr + C_z + C_ez) / Npix * texp).decompose().value, 1, None
        )
        # net charge transfer efficiency
        NCTE = 1.0 + (radDos / 4.0) * 0.51296 * (np.log10(phConv) + 0.0147233)
        # planet signal rate
        C_p = C_p0 * PCeff * NCTE

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
                C_F0=C_F0.to("1/s"),
                C_p0=C_p0.to("1/s"),
                C_bl=C_bl.to("1/s"),
            )
            return C_p.to("1/s"), C_b.to("1/s"), C_sp.to("1/s"), C_extra
        else:
            return C_p.to("1/s"), C_b.to("1/s"), C_sp.to("1/s")

    def calc_intTime(self, TL, sInds, fZ, fEZ, dMag, WA, mode, TK=None):
        """Finds integration time for a specific target system

        This method is called in the run_sim() method of the SurveySimulation
        class object. It defines the data type expected, integration time is
        determined by specific OpticalSystem classes.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            dMag (numpy.ndarray(int)numpy.ndarray(float)):
                Differences in magnitude between planets and their host star
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            TK (:ref:`TimeKeeping`, optional):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Integration times in units of days

        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        # default intTimes are 1 day
        intTime = np.ones(len(sInds)) * u.day

        return intTime

    def calc_dMag_per_intTime(
        self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None
    ):
        """Finds achievable planet delta magnitude for one integration
        time per star in the input list at one working angle.

        Args:
            intTimes (~astropy.units.Quantity(~numpy.ndarray(float))):
                Integration times in units of day
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (~astropy.units.Quantity(~numpy.ndarray(float))):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (~astropy.units.Quantity(~numpy.ndarray(float))):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            TK (:ref:`TimeKeeping`, optional):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.

        Returns:
            numpy.ndarray(float):
                Achievable dMag for given integration time and working angle

        """

        dMag = np.ones((len(sInds),))

        return dMag

    def ddMag_dt(
        self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None
    ):
        """Finds derivative of achievable dMag with respect to integration time.

        Args:
            intTimes (~astropy.units.Quantity(~numpy.ndarray(float))):
                Integration times in units of day
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (~astropy.units.Quantity(~numpy.ndarray(float))):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (~astropy.units.Quantity(~numpy.ndarray(float))):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            TK (:ref:`TimeKeeping`, optional):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Derivative of achievable dMag with respect to integration time
                in units of 1/s

        """

        ddMagdt = np.zeros((len(sInds),)) / u.s

        return ddMagdt

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
        TL, sInds, fZ, fEZ, WA, mode, TK = args
        C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dMag, WA, mode, TK=TK)
        denom = C_p.value**2 - (mode["SNR"] * C_sp.value) ** 2
        return denom

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
        TL, sInds, fZ, fEZ, WA, mode, TK, true_intTime = args
        est_intTime = self.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode, TK)
        abs_diff = np.abs(true_intTime.to("day").value - est_intTime.to("day").value)
        return abs_diff
