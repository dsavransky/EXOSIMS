# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
from EXOSIMS.util.utils import dictToSortedStr, genHexStr
from EXOSIMS.util.keyword_fun import get_all_args
from synphot.models import Box1D
from synphot.models import Gaussian1D
from synphot import SpectralElement, SourceSpectrum, Observation
import os.path
import numbers
import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import scipy.interpolate
import scipy.optimize
import copy
import warnings
from EXOSIMS.util._numpy_compat import copy_if_needed


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
            Default instrument half-field of view (in arcseconds). Only used when not
            set in science instrument definition. Defaults to 10
        pixelNumber (float):
            Default number of pixels across the detector. Only used when not set
            in science instrument definition. Defaults to 1000.
        pixelSize (float):
            Default pixel pitch (nominal distance between adjacent pixel centers,
            in meters). Only used when not set in science instrument definition.
            Defaults to 1e-5
        pixelScale (float):
            Default pixel scale (instantaneous field of view of each pixel,
            in arcseconds). Only used when not set in science instrument definition.
            Defaults to 0.02.
        sread (float):
            Default read noise (in electrons/pixel/read).  Only used when not set
            in science instrument definition. Defaults to 1e-6
        idark (float):
            Default dark current (in electrons/pixel/s).  Only used when not set
            in science instrument definition. Defaults to 1e-4
        texp (float):
            Default single exposure time (in s).  Only used when not set
            in science instrument definition. Defaults to 100
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
        core_contrast (float):
            Default core contrast. Only used when not set in starlight suppression
            system definition. Defaults to 1e-10
        contrast_floor (float, optional):
            Default contrast floor. Only used when not set in starlight suppression
            system definition. If not None, sets absolute contrast floor.
            Defaults to None
        core_platescale (float, optional):
            Default core platescale.  Only used when not set in starlight suppression
            system definition. Defaults to None. Units determiend by
            ``input_angle_units``.
        input_angle_units (str, optional):
            Default angle units of all starlightSuppressionSystems-related inputs
            (as applicable). This includes all CSV input tables or FITS input tables
            without a UNIT keyword in the header.
            Only used when not set in starlight suppression system definition.
            None, 'unitless' or 'LAMBDA/D' are all interepreted as :math:`\\lambda/D`
            units. Otherwise must be a string that is parsable as an astropy angle unit.
            Defaults to 'arcsec'.
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
            Default :term:`IWA` (in input_angle_units).  Only used when not set in
            starlight suppression system definition. Defaults to 0.1
        OWA (float):
            Default :term:`OWA` (in input_angle_units). Only used when not set in
            starlight suppression system definition. Defaults to numpy.Inf
        stabilityFact (float):
            Stability factor. Defaults to 1
        cachedir (str, optional):
            Full path to cachedir.
            If None (default) use default (see :ref:`EXOSIMSCACHE`)
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
        binaryleakfilepath (str, optional):
            If set, full path to binary leak definition file. Defaults to None
        texp_flag (bool):
            Toggle use of planet shot noise value for frame exposure time
            (overriides instrument texp value). Defaults to False.
        bandpass_model (str):
            Default model to use for mode bandpasses. Must be one of 'gaussian' or 'box'
            (case insensitive). Only used if not set in mode definition. Defaults to
            box.
        bandpass_step (float):
            Default step size (in nm) to use when generating Box-model bandpasses. Only
            used if not set in mode definition. Defaults to 0.1.
        use_core_thruput_for_ez (bool):
            If True, compute exozodi contribution using core_thruput.
            If False (default) use occ_trans
        csv_angsep_colname (str):
            Default column name to use for the angular separation column for CSV data.
            Only used when not set in starlight suppression system definition.
            Defaults to r_as (matching the default input_angle_units). These two inputs
            should be updated together.
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        _outspec (dict):
            :ref:`sec:outspec`
        allowed_observingMode_kws (list):
            List of allowed keywords in observingMode dictionaries
        allowed_scienceInstrument_kws (list):
            List of allowed keywords in scienceInstrument dictionaries
        allowed_starlightSuppressionSystem_kws (list):
            List of allowed keywords in starlightSuppressionSystem dictionaries
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)
        default_vals (dict):
            All inputs not assigned to object attributes are considered to be default
            values to be used for filling in information in the optical system
            definition, and are copied into this dictionary for storage.
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
        use_core_thruput_for_ez (bool):
            Toggle use of core_thruput (instead of occ_trans) in computing exozodi flux.

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
        pixelScale=0.02,
        sread=1e-6,
        idark=1e-4,
        texp=100,
        Rs=50,
        lenslSamp=2,
        starlightSuppressionSystems=[{"name": "coronagraph"}],
        lam=500,
        BW=0.2,
        occ_trans=0.2,
        core_thruput=0.1,
        core_contrast=1e-10,
        contrast_floor=None,
        core_platescale=None,
        core_platescale_units=None,
        input_angle_units="arcsec",
        ohTime=1,
        observingModes=None,
        SNR=5,
        timeMultiplier=1.0,
        IWA=0.1,
        OWA=np.inf,
        stabilityFact=1,
        cachedir=None,
        koAngles_Sun=[0, 180],
        koAngles_Earth=[0, 180],
        koAngles_Moon=[0, 180],
        koAngles_Small=[0, 180],
        binaryleakfilepath=None,
        texp_flag=False,
        bandpass_model="box",
        bandpass_step=0.1,
        use_core_thruput_for_ez=False,
        csv_angsep_colname="r_as",
        **specs,
    ):

        # start the outspec
        self._outspec = {}

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

        # set attributes from inputs
        self.obscurFac = float(obscurFac)  # obscuration factor (fraction of PM area)
        self.shapeFac = float(shapeFac)  # shape factor
        self.pupilDiam = float(pupilDiam) * u.m  # entrance pupil diameter
        self.intCutoff = float(intCutoff) * u.d  # integration time cutoff
        self.stabilityFact = float(stabilityFact)  # stability factor for telescope
        self.texp_flag = bool(texp_flag)
        self.use_core_thruput_for_ez = bool(use_core_thruput_for_ez)

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        specs["cachedir"] = self.cachedir

        # if binary leakage model provided, let's grab that as well
        if binaryleakfilepath is not None:
            binaryleakfilepathnorm = os.path.normpath(
                os.path.expandvars(binaryleakfilepath)
            )

            assert os.path.exists(
                binaryleakfilepathnorm
            ), "Binary leakage model data file not found at {}".format(
                binaryleakfilepath
            )

            binaryleakdata = np.genfromtxt(binaryleakfilepathnorm, delimiter=",")

            self.binaryleakmodel = scipy.interpolate.interp1d(
                binaryleakdata[:, 0], binaryleakdata[:, 1], bounds_error=False
            )
        self._outspec["binaryleakfilepath"] = binaryleakfilepath

        # populate outspec with all attributes assigned so far
        for att in self.__dict__:
            if att not in [
                "vprint",
                "_outspec",
            ]:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat

        # consistency check IWA/OWA defaults
        if OWA == 0:
            OWA = np.inf
        assert IWA < OWA, "Input default IWA must be smaller than input default OWA."

        # get all inputs that haven't been assiged to attributes will be treated as
        # default values (and should also go into outspec)
        kws = get_all_args(self.__class__)
        ignore_kws = [
            "self",
            "scienceInstruments",
            "starlightSuppressionSystems",
            "observingModes",
            "binaryleakfilepath",
        ]
        kws = list(
            (set(kws) - set(ignore_kws) - set(self.__dict__.keys())).intersection(
                set(locals().keys())
            )
        )
        self.default_vals = {}
        for kw in kws:
            self.default_vals[kw] = locals()[kw]
            if kw not in self._outspec:
                self._outspec[kw] = locals()[kw]

        # pupil collecting area (obscured PM)
        self.pupilArea = (1 - self.obscurFac) * self.shapeFac * self.pupilDiam**2

        # load Vega's spectrum for later calculations
        self.vega_spectrum = SourceSpectrum.from_vega()

        # populate science instruments (must have one defined)
        assert isinstance(scienceInstruments, list) and (
            len(scienceInstruments) > 0
        ), "No science instrument defined."
        self.populate_scienceInstruments(scienceInstruments)

        # populate starlight suppression systems (must have one defined)
        assert isinstance(starlightSuppressionSystems, list) and (
            len(starlightSuppressionSystems) > 0
        ), "No starlight suppression systems defined."
        self.populate_starlightSuppressionSystems(starlightSuppressionSystems)

        # if no observing mode defined, create a default mode from the first instrument
        # and first starlight suppression system. then populate all observing modes
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
        else:
            assert isinstance(observingModes, list) and (
                len(observingModes) > 0
            ), "No observing modes defined."

        self.populate_observingModes(observingModes)

        # populate fundamental IWA and OWA - the extrema of both values for all modes
        IWAs = [
            x.get("IWA").to(u.arcsec).value
            for x in self.observingModes
            if x.get("IWA") is not None
        ]
        if len(IWAs) > 0:
            self.IWA = min(IWAs) * u.arcsec
        else:
            self.IWA = float(IWA) * u.arcsec

        OWAs = [
            x.get("OWA").to(u.arcsec).value
            for x in self.observingModes
            if x.get("OWA") is not None
        ]
        if len(OWAs) > 0:
            self.OWA = max(OWAs) * u.arcsec
        else:
            self.OWA = float(OWA) * u.arcsec if OWA != 0 else np.inf * u.arcsec

        assert self.IWA < self.OWA, "Fundamental IWA must be smaller that the OWA."

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

    def populate_scienceInstruments(self, scienceInstruments):
        """Helper method to parse input scienceInstrument dictionaries and assign
        default values, as needed. Also creates the allowed_scienceInstrument_kws
        attribute.

        Args:
            scienceInstruments (list):
                List of scienceInstrument dicts.

        """

        self.scienceInstruments = copy.deepcopy(scienceInstruments)
        self._outspec["scienceInstruments"] = []
        instnames = []

        for ninst, inst in enumerate(self.scienceInstruments):
            assert isinstance(
                inst, dict
            ), "Science instruments must be defined as dicts."
            assert "name" in inst and isinstance(
                inst["name"], str
            ), "All science instruments must have key 'name'."
            instnames.append(inst["name"])

            # quantum efficiency can be a single number of a filename
            inst["QE"] = inst.get("QE", self.default_vals["QE"])
            self._outspec["scienceInstruments"].append(inst.copy())
            if isinstance(inst["QE"], str):
                # Load data and create interpolant
                dat, hdr = self.get_param_data(
                    inst["QE"],
                    # left_col_name="lambda", # TODO: start enforcing these
                    # param_name="QE",
                    expected_ndim=2,
                    expected_first_dim=2,
                )
                lam, D = (dat[0].astype(float), dat[1].astype(float))
                assert np.all(D >= 0) and np.all(
                    D <= 1
                ), "All QE values must be positive and smaller than 1."
                if isinstance(hdr, fits.Header):
                    if "UNITS" in hdr:
                        lam = ((lam * u.Unit(hdr["UNITS"])).to(u.nm)).value

                # parameter values outside of lam
                Dinterp1 = scipy.interpolate.interp1d(
                    lam,
                    D,
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
                inst["QE"] = self.default_vals["QE"]
                warnings.warn(
                    (
                        "QE input is not string or number for instrument "
                        f" {inst['name']}. Value set to default."
                    )
                )

            # load all required detector specifications
            # specify dictionary of keys and units
            kws = {
                "optics": None,  # attenuation due to instrument optics
                "FoV": u.arcsec,  # angular half-field of view of instrument
                "pixelNumber": None,  # array format
                "pixelSize": u.m,  # pixel pitch
                "pixelScale": u.arcsec,  # pixel scale (angular IFOV)
                "idark": 1 / u.s,  # dark-current rate
                "sread": None,  # effective readout noise
                "texp": u.s,  # default exposure time per frame
            }

            for kw in kws:
                inst[kw] = float(inst.get(kw, self.default_vals[kw]))
                if kws[kw] is not None:
                    inst[kw] *= kws[kw]

            # start tracking allowed_scienceInstrument_kws
            self.allowed_scienceInstrument_kws = ["name", "QE"] + list(kws.keys())

            # do some basic consistency checking on pixelScale and FoV:
            predFoV = np.arctan(inst["pixelNumber"] * np.tan(inst["pixelScale"] / 2))
            # generate warning if FoV is larger than prediction (but allow for
            # approximate equality)
            if (inst["FoV"] > predFoV) and not (np.isclose(inst["FoV"], predFoV)):
                warnings.warn(
                    f'Input FoV ({inst["FoV"]}) is larger than FoV computed '
                    f"from pixelScale ({predFoV.to(u.arcsec) :.2f}) for "
                    f'instrument {inst["name"]}. This feels like a mistake.'
                )

            # parameters specific to spectrograph
            if "spec" in inst["name"].lower():
                # spectral resolving power
                inst["Rs"] = float(inst.get("Rs", self.default_vals["Rs"]))
                # lenslet sampling, number of pixel per lenslet rows or cols
                inst["lenslSamp"] = float(
                    inst.get("lenslSamp", self.default_vals["lenslSamp"])
                )
            else:
                inst["Rs"] = 1.0
                inst["lenslSamp"] = 1.0

            self.allowed_scienceInstrument_kws += ["Rs", "lenslSamp"]

            # calculate focal length and f-number as needed
            if "focal" in inst:
                inst["focal"] = float(inst["focal"]) * u.m
                inst["fnumber"] = float(inst["focal"] / self.pupilDiam)
            elif ("fnumber") in inst:
                inst["fnumber"] = float(inst["fnumber"])
                inst["focal"] = inst["fnumber"] * self.pupilDiam
            else:
                inst["focal"] = (
                    inst["pixelSize"] / 2 / np.tan(inst["pixelScale"] / 2)
                ).to(u.m)
                inst["fnumber"] = float(inst["focal"] / self.pupilDiam)

            self.allowed_scienceInstrument_kws += ["focal", "fnumber"]

            # consistency check parameters
            predFocal = (inst["pixelSize"] / 2 / np.tan(inst["pixelScale"] / 2)).to(u.m)
            if not (np.isclose(predFocal.value, inst["focal"].to(u.m).value)):
                warnings.warn(
                    f'Input focal length ({inst["focal"] :.2f}) does not '
                    f"match value from pixelScale ({predFocal :.2f}) for "
                    f'instrument {inst["name"]}. This feels like a mistkae.'
                )

            # populate updated detector specifications to outspec
            for att in inst:
                if att not in ["QE"]:
                    dat = inst[att]
                    self._outspec["scienceInstruments"][ninst][att] = (
                        dat.value if isinstance(dat, u.Quantity) else dat
                    )

        # ensure that all instrument names are unique:
        assert (
            len(instnames) == np.unique(instnames).size
        ), "Instrument names muse be unique."

        # call additional instrument setup
        self.populate_scienceInstruments_extra()

    def populate_scienceInstruments_extra(self):
        """Additional setup for science instruments.  This is intended for overloading
        in downstream implementations and is intentionally left blank in the prototype.
        """

        pass

    def populate_starlightSuppressionSystems(self, starlightSuppressionSystems):
        """Helper method to parse input starlightSuppressionSystem dictionaries and
        assign default values, as needed. Also creates the
        allowed_starlightSuppressionSystem_kws attribute.

        Args:
            starlightSuppressionSystems (list):
                List of starlightSuppressionSystem dicts.

        """

        self.starlightSuppressionSystems = copy.deepcopy(starlightSuppressionSystems)
        self.haveOcculter = False
        self._outspec["starlightSuppressionSystems"] = []
        systnames = []

        for nsyst, syst in enumerate(self.starlightSuppressionSystems):
            assert isinstance(
                syst, dict
            ), "Starlight suppression systems must be defined as dicts."
            assert "name" in syst and isinstance(
                syst["name"], str
            ), "All starlight suppression systems must have key 'name'."
            systnames.append(syst["name"])

            # determine system wavelength (lam), bandwidth (deltaLam) and bandwidth
            # fraction (BW)
            # use deltaLam if given, otherwise use BW
            syst["lam"] = float(syst.get("lam", self.default_vals["lam"])) * u.nm
            syst["deltaLam"] = (
                float(
                    syst.get(
                        "deltaLam",
                        syst["lam"].to("nm").value
                        * syst.get("BW", self.default_vals["BW"]),
                    )
                )
                * u.nm
            )
            syst["BW"] = float(syst["deltaLam"] / syst["lam"])

            # populate all required default_vals
            names = [
                "occ_trans",
                "core_thruput",
                "core_platescale",
                "input_angle_units",
                "core_platescale_units",
                "contrast_floor",
                "csv_angsep_colname",
            ]
            # fill contrast from default only if core_mean_intensity not set
            if "core_mean_intensity" not in syst:
                names.append("core_contrast")
            for n in names:
                syst[n] = syst.get(n, self.default_vals[n])

            # start tracking allowed keywords
            self.allowed_starlightSuppressionSystem_kws = [
                "name",
                "lam",
                "deltaLam",
                "BW",
                "core_mean_intensity",
                "optics",
                "occulter",
                "ohTime",
                "core_platescale",
                "IWA",
                "OWA",
                "core_area",
            ]
            self.allowed_starlightSuppressionSystem_kws += names
            if "core_contrast" not in self.allowed_starlightSuppressionSystem_kws:
                self.allowed_starlightSuppressionSystem_kws.append("core_contrast")

            # attenuation due to optics specific to the coronagraph not caputred by the
            # coronagraph throughput curves. Defaults to 1.
            syst["optics"] = float(syst.get("optics", 1.0))

            # set an occulter, for an external or hybrid system
            syst["occulter"] = syst.get("occulter", False)
            if syst["occulter"]:
                self.haveOcculter = True

            # copy system definition to outspec
            self._outspec["starlightSuppressionSystems"].append(syst.copy())

            # now we populate everything that has units

            # overhead time:
            syst["ohTime"] = (
                float(syst.get("ohTime", self.default_vals["ohTime"])) * u.d
            )

            # figure out the angle unit we're assuming for all inputs
            syst["input_angle_unit_value"] = self.get_angle_unit_from_header(None, syst)

            # if platescale was set, give it units
            if syst["core_platescale"] is not None:
                # check for units to use
                if (syst["core_platescale_units"] is None) or (
                    syst["core_platescale_units"] in ["unitless", "LAMBDA/D"]
                ):
                    platescale_unit = (syst["lam"] / self.pupilDiam).to(
                        u.arcsec, equivalencies=u.dimensionless_angles()
                    )
                else:
                    platescale_unit = 1 * u.Unit(syst["core_platescale_units"])

                syst["core_platescale"] = (
                    syst["core_platescale"] * platescale_unit
                ).to(u.arcsec)

            # if IWA/OWA are given, assign them units (otherwise they'll be set from
            # table data or defaults (whichever comes first).
            if "IWA" in syst:
                syst["IWA"] = (float(syst["IWA"]) * syst["input_angle_unit_value"]).to(
                    u.arcsec
                )
            if "OWA" in syst:
                # Zero OWA aliased to inf OWA
                if (syst["OWA"] == 0) or (syst["OWA"] == np.inf):
                    syst["OWA"] = np.inf * u.arcsec
                else:
                    syst["OWA"] = (
                        float(syst["OWA"]) * syst["input_angle_unit_value"]
                    ).to(u.arcsec)

            # get the system's keepout angles
            names = [
                "koAngles_Sun",
                "koAngles_Earth",
                "koAngles_Moon",
                "koAngles_Small",
            ]
            for n in names:
                syst[n] = [float(x) for x in syst.get(n, self.default_vals[n])] * u.deg

            self.allowed_starlightSuppressionSystem_kws += names

            # now we're going to populate everything that's callable

            # first let's handle core mean intensity
            if "core_mean_intensity" in syst:
                syst = self.get_core_mean_intensity(syst)

                # ensure that platescale has also been set
                assert syst["core_platescale"] is not None, (
                    f"In system {syst['name']}, core_mean_intensity "
                    "is set, but core_platescale is not.  This is not allowed."
                )
            else:
                syst["core_mean_intensity"] = None

            if "core_contrast" in syst:
                syst = self.get_coro_param(
                    syst,
                    "core_contrast",
                    fill=1.0,
                    expected_ndim=2,
                    expected_first_dim=2,
                    min_val=0.0,
                )
            else:
                syst["core_contrast"] = None

            # now get the throughputs
            syst = self.get_coro_param(
                syst,
                "occ_trans",
                expected_ndim=2,
                expected_first_dim=2,
                min_val=0.0,
                max_val=(np.inf if syst["occulter"] else 1.0),
            )
            syst = self.get_coro_param(
                syst,
                "core_thruput",
                expected_ndim=2,
                expected_first_dim=2,
                min_val=0.0,
                max_val=(np.inf if syst["occulter"] else 1.0),
            )

            # finally, for core_area, if none is supplied, then set to area of
            # \sqrt{2}/2 lambda/D radius aperture
            if (
                ("core_area" not in syst)
                or (syst["core_area"] is None)
                or (syst["core_area"] == 0)
            ):
                # need to put this in the proper unit
                angunit = self.get_angle_unit_from_header(None, syst)

                syst["core_area"] = (
                    (
                        (np.pi / 2)
                        * (syst["lam"] / self.pupilDiam).to(
                            u.arcsec, equivalencies=u.dimensionless_angles()
                        )
                        ** 2
                        / angunit**2
                    )
                    .decompose()
                    .value
                )
            syst = self.get_coro_param(
                syst,
                "core_area",
                expected_ndim=2,
                expected_first_dim=2,
                min_val=0.0,
            )

            # at this point, we must have set an IWA/OWA, but lets make sure
            for key in ["IWA", "OWA"]:
                assert (
                    (key in syst)
                    and isinstance(syst[key], u.Quantity)
                    and (syst[key].unit == u.arcsec)
                ), f"{key} not found or has the wrong unit in system {syst['name']}."

            # populate system specifications to outspec
            for att in syst:
                if att not in [
                    "occ_trans",
                    "core_thruput",
                    "core_contrast",
                    "core_mean_intensity",
                    "core_area",
                    "input_angle_unit_value",
                    "IWA",
                    "OWA",
                ]:
                    dat = syst[att]
                    self._outspec["starlightSuppressionSystems"][nsyst][att] = (
                        dat.value if isinstance(dat, u.Quantity) else dat
                    )

        # ensure that all starlight suppression system names are unique:
        assert (
            len(systnames) == np.unique(systnames).size
        ), "Starlight suppression system names muse be unique."

        # call additional setup
        self.populate_starlightSuppressionSystems_extra()

    def populate_starlightSuppressionSystems_extra(self):
        """Additional setup for starlight suppression systems.  This is intended for
        overloading in downstream implementations and is intentionally left blank in
        the prototype.
        """

        pass

    def populate_observingModes(self, observingModes):
        """Helper method to parse input observingMode dictionaries and assign default
        values, as needed. Also creates the allowed_observingMode_kws attribute.

        Args:
            observingModes (list):
                List of observingMode dicts.

        """

        self.observingModes = observingModes
        self._outspec["observingModes"] = []
        for nmode, mode in enumerate(self.observingModes):
            assert isinstance(mode, dict), "Observing modes must be defined as dicts."
            assert (
                "instName" in mode and "systName" in mode
            ), "All observing modes must have keys 'instName' and 'systName'."
            assert np.any(
                [mode["instName"] == inst["name"] for inst in self.scienceInstruments]
            ), f"The mode's instrument name {mode['instName']} does not exist."
            assert np.any(
                [
                    mode["systName"] == syst["name"]
                    for syst in self.starlightSuppressionSystems
                ]
            ), f"The mode's system name {mode['systName']} does not exist."
            self._outspec["observingModes"].append(mode.copy())

            # loading mode specifications
            mode["SNR"] = float(mode.get("SNR", self.default_vals["SNR"]))
            mode["timeMultiplier"] = float(
                mode.get("timeMultiplier", self.default_vals["timeMultiplier"])
            )
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

            # start tracking allowed keywords
            self.allowed_observingMode_kws = [
                "instName",
                "systName",
                "SNR",
                "timeMultiplier",
                "detectionMode",
                "lam",
                "deltaLam",
                "BW",
                "bandpass_model",
                "bandpass_step",
            ]

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

            # get mode IWA and OWA: rescale if the mode wavelength is different from
            # the wavelength at which the system is defined
            mode["IWA"] = mode["syst"]["IWA"]
            mode["OWA"] = mode["syst"]["OWA"]
            if mode["lam"] != mode["syst"]["lam"]:
                mode["IWA"] = mode["IWA"] * mode["lam"] / mode["syst"]["lam"]
                mode["OWA"] = mode["OWA"] * mode["lam"] / mode["syst"]["lam"]

            # OWA must be bounded by FOV:
            if mode["OWA"] > mode["inst"]["FoV"]:
                mode["OWA"] = mode["inst"]["FoV"]

            # generate the mode's bandpass
            # TODO: Add support for custom filter profiles
            mode["bandpass_model"] = mode.get(
                "bandpass_model", self.default_vals["bandpass_model"]
            ).lower()
            assert mode["bandpass_model"] in [
                "gaussian",
                "box",
            ], "bandpass_model must be one of ['gaussian', 'box']"
            mode["bandpass_step"] = (
                float(mode.get("bandpass_step", self.default_vals["bandpass_step"]))
                * u.nm
            )
            if mode["bandpass_model"] == "box":
                mode["bandpass"] = SpectralElement(
                    Box1D,
                    x_0=mode["lam"],
                    width=mode["deltaLam"],
                    step=mode["bandpass_step"].to(u.AA).value,
                )
            else:
                mode["bandpass"] = SpectralElement(
                    Gaussian1D,
                    mean=mode["lam"],
                    stddev=mode["deltaLam"] / np.sqrt(2 * np.pi),
                )

            # check for out of range wavelengths
            # currently capped to 10 um
            assert (
                mode["bandpass"].waveset.max() < 10 * u.um
            ), "Bandpasses beyond 10 um are not supported."

            # evaluate zero-magnitude flux for this band from vega spectrum
            # NB: This is flux, not flux density! The bandpass is already factored in.
            mode["F0"] = Observation(
                self.vega_spectrum, mode["bandpass"], force="taper"
            ).integrate()

            # populate system specifications to outspec
            for att in mode:
                if att not in [
                    "inst",
                    "syst",
                    "F0",
                    "bandpass",
                ]:
                    dat = mode[att]
                    self._outspec["observingModes"][nmode][att] = (
                        dat.value if isinstance(dat, u.Quantity) else dat
                    )

            # populate some final mode attributes (computed from the others)
            # define total mode attenution
            mode["attenuation"] = mode["inst"]["optics"] * mode["syst"]["optics"]

            # effective mode bandwidth (including any IFS spectral resolving power)
            mode["deltaLam_eff"] = (
                mode["lam"] / mode["inst"]["Rs"]
                if "spec" in mode["inst"]["name"].lower()
                else mode["deltaLam"]
            )

            # total attenuation due to non-coronagraphic optics:
            mode["losses"] = (
                self.pupilArea
                * mode["inst"]["QE"](mode["lam"])
                * mode["attenuation"]
                * mode["deltaLam_eff"]
                / mode["deltaLam"]
            )

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

        self.populate_observingModes_extra()

    def populate_observingModes_extra(self):
        """Additional setup for observing modes  This is intended for overloading in
        downstream implementations and is intentionally left blank in the prototype.
        """

        pass

    def genObsModeHex(self):
        """Generate a unique hash for every observing mode to be used in downstream
        identification and caching. Also adds an integer index to the mode corresponding
        to its order in the observingModes list.

        The hash will be based on the _outspec entries for the obsmode, its science
        instrument and its starlight suppression system.
        """

        for nmode, mode in enumerate(self.observingModes):
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
                dictToSortedStr(self._outspec["observingModes"][nmode]),
                dictToSortedStr(inst),
                dictToSortedStr(syst),
            )

            mode["hex"] = genHexStr(modestr)
            mode["index"] = nmode

    def get_core_mean_intensity(
        self,
        syst,
    ):
        """Load and process core_mean_intensity data

        Args:
            syst (dict):
                Dictionary containing the parameters of one starlight suppression system

        Returns:
            dict:
                Updated dictionary of starlight suppression system parameters

        """

        param_name = "core_mean_intensity"
        fill = 1.0
        assert param_name in syst, f"{param_name} not found in syst."
        if isinstance(syst[param_name], str):
            dat, hdr = self.get_param_data(
                syst[param_name],
                left_col_name=syst["csv_angsep_colname"],
                param_name=param_name,
                expected_ndim=2,
            )
            dat = dat.transpose()  # flip such that data is in rows
            WA, D = dat[0].astype(float), dat[1:].astype(float)

            # check values as needed
            assert np.all(
                D > 0
            ), f"{param_name} in {syst['name']} must be >0 everywhere."

            # get angle unit scale WA
            angunit = self.get_angle_unit_from_header(hdr, syst)
            WA = (WA * angunit).to(u.arcsec).value

            # get platescale from header (if this is a FITS header)
            if isinstance(hdr, fits.Header) and ("PIXSCALE" in hdr):
                # use the header unit preferentially. otherwise drop back to the
                # core_platescale_units keyword
                if "UNITS" in hdr:
                    platescale = (float(hdr["PIXSCALE"]) * angunit).to(u.arcsec)
                else:
                    if (syst["core_platescale_units"] is None) or (
                        syst["core_platescale_units"] in ["unitless", "LAMBDA/D"]
                    ):
                        platescale_unit = (syst["lam"] / self.pupilDiam).to(
                            u.arcsec, equivalencies=u.dimensionless_angles()
                        )
                    else:
                        platescale_unit = 1 * u.Unit(syst["core_platescale_units"])
                    platescale = (float(hdr["PIXSCALE"]) * platescale_unit).to(u.arcsec)

                if (syst.get("core_platescale") is not None) and (
                    syst["core_platescale"] != platescale
                ):
                    warnings.warn(
                        "platescale for core_mean_intensity in system "
                        f"{syst['name']} does not match input value.  "
                        "Overwriting with value from FITS file but you "
                        "should check your inputs."
                    )
                syst["core_platescale"] = platescale

            # handle case where only one data row is present
            if D.shape[0] == 1:
                D = np.squeeze(D)

                # table interpolate function
                Dinterp = scipy.interpolate.interp1d(
                    WA,
                    D,
                    kind="linear",
                    fill_value=fill,
                    bounds_error=False,
                )
                # create a callable lambda function. for coronagraphs, we need to scale
                # the angular separation by wavelength, but for occulters we just need
                # to ensure that we're within the wavelength range
                if syst["occulter"]:
                    minl = syst["lam"] - syst["deltaLam"] / 2
                    maxl = syst["lam"] + syst["deltaLam"] / 2
                    syst[param_name] = (
                        lambda lam, s, d=0 * u.arcsec, Dinterp=Dinterp, minl=minl, maxl=maxl, fill=fill: (  # noqa: E501
                            np.array(Dinterp(s.to("arcsec").value), ndmin=1) - fill
                        )
                        * np.array((minl < lam) & (lam < maxl), ndmin=1).astype(int)
                        + fill
                    )
                else:
                    syst[param_name] = (
                        lambda lam, s, d=0 * u.arcsec, Dinterp=Dinterp, lam0=syst[
                            "lam"
                        ]: np.array(
                            Dinterp((s * lam0 / lam).to("arcsec").value), ndmin=1
                        )
                    )

            # and now the general case of multiple rows
            else:
                # grab stellar diameters from header info
                diams = np.zeros(len(D))
                # FITS files
                if isinstance(hdr, fits.Header):
                    for j in range(len(D)):
                        k = f"DIAM{j :03d}"
                        assert k in hdr, (
                            f"Expected keyword {k} not found in header "
                            f"of file {syst[param_name]} for system "
                            f"{syst['name']}"
                        )
                        diams[j] = float(hdr[k])
                # TODO: support for CSV files
                else:
                    raise NotImplementedError(
                        "No CSV support for 2D core_mean_intensity"
                    )

                # determine units and convert as needed
                diams = (diams * angunit).to(u.arcsec).value

                Dinterp = scipy.interpolate.RegularGridInterpolator(
                    (WA, diams), D.transpose(), bounds_error=False, fill_value=1.0
                )

                # create a callable lambda function. for coronagraphs, we need to scale
                # the angular separation and stellar diameter by wavelength, but for
                # occulters we just need to ensure that we're within the wavelength
                # range
                if syst["occulter"]:
                    minl = syst["lam"] - syst["deltaLam"] / 2
                    maxl = syst["lam"] + syst["deltaLam"] / 2
                    syst[param_name] = (
                        lambda lam, s, d=0 * u.arcsec, Dinterp=Dinterp, minl=minl, maxl=maxl, fill=fill: (  # noqa: E501
                            np.array(
                                Dinterp((s.to("arcsec").value, d.to("arcsec").value)),
                                ndmin=1,
                            )
                            - fill
                        )
                        * np.array((minl < lam) & (lam < maxl), ndmin=1).astype(int)
                        + fill
                    )
                else:
                    syst[
                        param_name
                    ] = lambda lam, s, d=0 * u.arcsec, Dinterp=Dinterp, lam0=syst[
                        "lam"
                    ]: np.array(
                        Dinterp(
                            (
                                (s * lam0 / lam).to("arcsec").value,
                                (d * lam0 / lam).to("arcsec").value,
                            )
                        ),
                        ndmin=1,
                    )

            # update IWA/OWA in system as needed
            syst = self.update_syst_WAs(syst, WA, param_name)

        elif isinstance(syst[param_name], numbers.Number):
            # ensure paramter is within bounds
            D = float(syst[param_name])
            assert D > 0, f"{param_name} in {syst['name']} must be > 0."

            # ensure you have values for IWA/OWA, otherwise use defaults
            syst = self.update_syst_WAs(syst, None, None)
            IWA = syst["IWA"].to(u.arcsec).value
            OWA = syst["OWA"].to(u.arcsec).value

            # same as for interpolant: coronagraphs scale with wavelength, occulters
            # don't
            if syst["occulter"]:
                minl = syst["lam"] - syst["deltaLam"] / 2
                maxl = syst["lam"] + syst["deltaLam"] / 2

                syst[param_name] = (
                    lambda lam, s, d=0 * u.arcsec, D=D, IWA=IWA, OWA=OWA, minl=minl, maxl=maxl, fill=fill: (  # noqa: E501
                        np.array(
                            (IWA <= s.to("arcsec").value)
                            & (s.to("arcsec").value <= OWA)
                            & (minl < lam)
                            & (lam < maxl),
                            ndmin=1,
                        ).astype(float)
                        * (D - fill)
                        + fill
                    )
                )

            else:
                syst[param_name] = (
                    lambda lam, s, d=0 * u.arcsec, D=D, lam0=syst[
                        "lam"
                    ], IWA=IWA, OWA=OWA, fill=fill: (
                        np.array(
                            (IWA <= (s * lam0 / lam).to("arcsec").value)
                            & ((s * lam0 / lam).to("arcsec").value <= OWA),
                            ndmin=1,
                        ).astype(float)
                    )
                    * (D - fill)
                    + fill
                )
        elif syst[param_name] is None:
            syst[param_name] = None
        else:
            raise TypeError(
                f"{param_name} for system {syst['name']} is neither a "
                f"string nor a number. I don't know what to do with that."
            )

        return syst

    def get_angle_unit_from_header(self, hdr, syst):
        """Helper method. Extract angle unit from header, if it exists.

        Args:
            hdr (astropy.io.fits.header.Header or list):
                FITS header for data or header row from CSV
            syst (dict):
                Dictionary containing the parameters of one starlight suppression system

        Returns:
            astropy.units.Unit:
                The angle unit.
        """
        # if this is a FITS header, grab value from UNITS key if it exists
        if isinstance(hdr, fits.Header) and ("UNITS" in hdr):
            if hdr["UNITS"] in ["unitless", "LAMBDA/D"]:
                angunit = (syst["lam"] / self.pupilDiam).to(
                    u.arcsec, equivalencies=u.dimensionless_angles()
                )
            else:
                angunit = 1 * u.Unit(hdr["UNITS"])
        # otherwise, use the input_angle_units key
        else:
            # check if we've already computed this
            if "input_angle_unit_value" in syst:
                angunit = syst["input_angle_unit_value"]
            else:
                # if we're here, we have to do it from scratch
                if (syst["input_angle_units"] is None) or (
                    syst["input_angle_units"] in ["unitless", "LAMBDA/D"]
                ):
                    angunit = (syst["lam"] / self.pupilDiam).to(
                        u.arcsec, equivalencies=u.dimensionless_angles()
                    )
                else:
                    angunit = 1 * u.Unit(syst["input_angle_units"])

        # final consistency check before returning
        assert (
            angunit.unit.physical_type == "angle"
        ), f"Angle unit for system {syst['name']} is not an angle."

        return angunit

    def update_syst_WAs(self, syst, WA0, param_name):
        """Helper method. Check system IWA/OWA and update from table
        data, as needed. Alternatively, set from defaults.

        Args:
            syst (dict):
                Dictionary containing the parameters of one starlight suppression system
            WA0 (~numpy.ndarray, optional):
                Array of angles from table data. Must be in arcseconds. If None, then
                just set from defaults.
            param_name (str, optional):
                Name of parameter the table data belongs to. Must be set if WA is set.

        Returns:
            dict:
                Updated dictionary of starlight suppression system parameters

        """

        # if WA not given, then we're going to be setting defaults, as needed.
        if WA0 is None:
            if "IWA" not in syst:
                syst["IWA"] = (
                    float(self.default_vals["IWA"]) * syst["input_angle_unit_value"]
                ).to(u.arcsec)

            if "OWA" not in syst:
                syst["OWA"] = (
                    float(self.default_vals["OWA"]) * syst["input_angle_unit_value"]
                ).to(u.arcsec)

            return syst

        # otherwise, update IWA from table value
        WA = WA0 * u.arcsec
        if ("IWA" in syst) and (np.min(WA) > syst["IWA"]):
            warnings.warn(
                f"{param_name} has larger IWA than current system value "
                f"for {syst['name']}. Updating to match table, but you "
                "should check your inputs."
            )
            syst["IWA"] = np.min(WA)
        elif "IWA" not in syst:
            syst["IWA"] = np.min(WA)

        # update OWA (if not an occulter)
        if not (syst["occulter"]) and ("OWA" in syst) and (np.max(WA) < syst["OWA"]):
            warnings.warn(
                f"{param_name} has smaller OWA than current system "
                f"value for {syst['name']}. Updating to match table, but "
                "you should check your inputs."
            )
            syst["OWA"] = np.max(WA)
        elif "OWA" not in syst:
            syst["OWA"] = np.max(WA)

        return syst

    def get_coro_param(
        self,
        syst,
        param_name,
        fill=0.0,
        expected_ndim=None,
        expected_first_dim=None,
        min_val=None,
        max_val=None,
    ):
        """For a given starlightSuppressionSystem, this method loads an input
        parameter from a table (fits or csv file) or a scalar value. It then creates a
        callable lambda function, which depends on the wavelength of the system
        and the angular separation of the observed planet.

        Args:
            syst (dict):
                Dictionary containing the parameters of one starlight suppression system
            param_name (str):
                Name of the parameter that must be loaded
            fill (float):
                Fill value for working angles outside of the input array definition
            expected_ndim (int, optional):
                Expected number of dimensions.  Only checked if not None. Defaults None.
            expected_first_dim (int, optional):
                Expected size of first dimension of data.  Only checked if not None.
                Defaults None
            min_val (float, optional):
                Minimum allowed value of parameter. Defaults to None (no check).
            max_val (float, optional):
                Maximum allowed value of paramter. Defaults to None (no check).

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

        assert param_name in syst, f"{param_name} not found in system {syst['name']}."
        if isinstance(syst[param_name], str):
            dat, hdr = self.get_param_data(
                syst[param_name],
                left_col_name=syst["csv_angsep_colname"],
                param_name=param_name,
                expected_ndim=expected_ndim,
                expected_first_dim=expected_first_dim,
            )
            WA, D = dat[0].astype(float), dat[1].astype(float)

            # check values as needed
            if min_val is not None:
                assert np.all(D >= min_val), (
                    f"{param_name} in {syst['name']} may not "
                    f"have values less than {min_val}."
                )
            if max_val is not None:
                assert np.all(D <= max_val), (
                    f"{param_name} in {syst['name']} may "
                    f"not have values greater than {max_val}."
                )

            # check for units
            angunit = self.get_angle_unit_from_header(hdr, syst)
            WA = (WA * angunit).to(u.arcsec).value

            # for core_area only, also need to scale the data
            if param_name == "core_area":
                D = (D * angunit**2).to(u.arcsec**2).value

            # update IWA/OWA as needed
            syst = self.update_syst_WAs(syst, WA, param_name)

            # table interpolate function
            Dinterp = scipy.interpolate.interp1d(
                WA,
                D,
                kind="linear",
                fill_value=fill,
                bounds_error=False,
            )
            # create a callable lambda function. for coronagraphs, we need to scale the
            # angular separation by wavelength, but for occulters we just need to
            # ensure that we're within the wavelength range. for core_area, we also
            # need to scale the output by wavelengh^2.
            if syst["occulter"]:
                minl = syst["lam"] - syst["deltaLam"] / 2
                maxl = syst["lam"] + syst["deltaLam"] / 2
                if param_name == "core_area":
                    outunit = 1 * u.arcsec**2
                else:
                    outunit = 1
                syst[param_name] = (
                    lambda lam, s, Dinterp=Dinterp, minl=minl, maxl=maxl, fill=fill: (
                        (np.array(Dinterp(s.to("arcsec").value), ndmin=1) - fill)
                        * np.array((minl < lam) & (lam < maxl), ndmin=1).astype(int)
                        + fill
                    )
                    * outunit
                )
            else:
                if param_name == "core_area":
                    syst[param_name] = (
                        lambda lam, s, Dinterp=Dinterp, lam0=syst["lam"]: np.array(
                            Dinterp((s * lam0 / lam).to("arcsec").value), ndmin=1
                        )
                        * ((lam / lam0).decompose() * u.arcsec) ** 2
                    )
                else:
                    syst[param_name] = lambda lam, s, Dinterp=Dinterp, lam0=syst[
                        "lam"
                    ]: np.array(Dinterp((s * lam0 / lam).to("arcsec").value), ndmin=1)
        # now the case where we just got a scalar input
        elif isinstance(syst[param_name], numbers.Number):
            # ensure paramter is within bounds
            D = float(syst[param_name])
            if min_val is not None:
                assert D >= min_val, (
                    f"{param_name} in {syst['name']} may not "
                    f"have values less than {min_val}."
                )
            if max_val is not None:
                assert D <= max_val, (
                    f"{param_name} in {syst['name']} may "
                    f"not have values greater than {min_val}."
                )

            # for core_area only, need to make sure that the units are right
            if param_name == "core_area":
                angunit = self.get_angle_unit_from_header(None, syst)
                D = (D * angunit**2).to(u.arcsec**2).value

            # ensure you have values for IWA/OWA, otherwise use defaults
            syst = self.update_syst_WAs(syst, None, None)
            IWA = syst["IWA"].to(u.arcsec).value
            OWA = syst["OWA"].to(u.arcsec).value

            # same as for interpolant: coronagraphs scale with wavelength, occulters
            # don't
            if syst["occulter"]:
                minl = syst["lam"] - syst["deltaLam"] / 2
                maxl = syst["lam"] + syst["deltaLam"] / 2
                if param_name == "core_area":
                    outunit = 1 * u.arcsec**2
                else:
                    outunit = 1

                syst[param_name] = (
                    lambda lam, s, D=D, IWA=IWA, OWA=OWA, minl=minl, maxl=maxl, fill=fill: (  # noqa: E501
                        (
                            np.array(
                                (IWA <= s.to("arcsec").value)
                                & (s.to("arcsec").value <= OWA)
                                & (minl < lam)
                                & (lam < maxl),
                                ndmin=1,
                            ).astype(float)
                            * (D - fill)
                            + fill
                        )
                        * outunit
                    )
                )
            # coronagraph:
            else:
                if param_name == "core_area":
                    syst[param_name] = (
                        lambda lam, s, D=D, lam0=syst[
                            "lam"
                        ], IWA=IWA, OWA=OWA, fill=fill: (
                            np.array(
                                (IWA <= (s * lam0 / lam).to("arcsec").value)
                                & ((s * lam0 / lam).to("arcsec").value <= OWA),
                                ndmin=1,
                            ).astype(float)
                            * (lam / lam0 * u.arcsec) ** 2
                        )
                        * (D - fill)
                        + fill
                    )
                else:
                    syst[param_name] = (
                        lambda lam, s, D=D, lam0=syst[
                            "lam"
                        ], IWA=IWA, OWA=OWA, fill=fill: (
                            np.array(
                                (IWA <= (s * lam0 / lam).to("arcsec").value)
                                & ((s * lam0 / lam).to("arcsec").value <= OWA),
                                ndmin=1,
                            ).astype(float)
                        )
                        * (D - fill)
                        + fill
                    )
        # finally the case where the input is None
        elif syst[param_name] is None:
            syst[param_name] = None
        # anything else (not string, number, or None) throws an error
        else:
            raise TypeError(
                f"{param_name} for system {syst['name']} is neither a "
                f"string nor a number. I don't know what to do with that."
            )

        return syst

    def get_param_data(
        self,
        ipth,
        left_col_name=None,
        param_name=None,
        expected_ndim=None,
        expected_first_dim=None,
    ):
        """Gets the data from a file, used primarily to create interpolants for
        coronagraph parameters

        Args:
            ipth (str):
                String to file location, will also work with any other path object
            left_col_name (str,optional):
                For CSV files only. String representing the column containing the
                independent parameter to be extracted. This is for use in the case
                where the CSV file contains multiple columns and only two need to be
                returned. Defaults None.
            param_name (str, optional):
                For CSV files only. String representing the column containing the
                dependent parameter to be extracted. This is for use in the case where
                the CSV file contains multiple columns and only two need to be returned.
                Defaults None.
            expected_ndim (int, optional):
                Expected number of dimensions.  Only checked if not None. Defaults None.
            expected_first_dim (int, optional):
                Expected size of first dimension of data.  Only checked if not None.
                Defaults None

        Returns:
            tuple:
                dat (~numpy.ndarray):
                    Data array
                hdr (list or astropy.io.fits.header.Header):
                    Data header.  For CVS files this is a list of column header strings.

        .. note::

            CSV files *must* have a single header row

        """
        # Check that path represents a valid file
        pth = os.path.normpath(os.path.expandvars(ipth))
        assert os.path.isfile(pth), f"{ipth} is not a valid file."

        # Check for fits or csv file
        ext = pth.split(".")[-1]
        assert ext.lower() in ["fits", "csv"], f"{pth} must be a fits or csv file."
        if ext.lower() == "fits":
            with fits.open(pth) as f:
                dat = f[0].data.squeeze()
                hdr = f[0].header
        else:
            # Need to get all of the headers and data, then associate them in the same
            # ndarray that the fits files would generate
            table_vals = np.genfromtxt(pth, delimiter=",", skip_header=1)
            hdr = np.genfromtxt(
                pth, delimiter=",", skip_footer=len(table_vals), dtype=str
            )

            if left_col_name is not None:
                assert (
                    param_name is not None
                ), "If left_col_name is nont None, param_name cannot be None."

                assert (
                    left_col_name in hdr
                ), f"{left_col_name} not found in table header for file {ipth}"
                assert (
                    param_name in hdr
                ), f"{param_name} not found in table header for file {ipth}"

                left_column_location = np.where(hdr == left_col_name)[0][0]
                param_location = np.where(hdr == param_name)[0][0]
                dat = np.vstack(
                    [table_vals[:, left_column_location], table_vals[:, param_location]]
                ).T
                hdr = [left_col_name, param_name]
            else:
                dat = table_vals

        if expected_ndim is not None:
            assert len(dat.shape) == expected_ndim, (
                f"Data shape did not match expected {expected_ndim} "
                f"dimensions for file {ipth}"
            )

        if expected_first_dim is not None:
            assert expected_first_dim in dat.shape, (
                f"Expected first dimension size {expected_first_dim} not found in any "
                f"data dimension for file {ipth}."
            )

            if dat.shape[0] != expected_first_dim:
                assert len(dat.shape) == 2, (
                    f"Data in file {ipth} contains a dimension of expected size "
                    f"{expected_first_dim}, but it is not the first dimension, and the "
                    "data has dimensionality of > 2, so I do not know how to reshape "
                    "it."
                )

                dat = dat.transpose()

        return dat, hdr

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

        # grab all count rates
        C_star, C_p, C_sr, C_z, C_ez, C_dc, C_bl, Npix = self.Cp_Cb_Csp_helper(
            TL, sInds, fZ, fEZ, dMag, WA, mode
        )

        # readout noise
        inst = mode["inst"]
        C_rn = Npix * inst["sread"] / inst["texp"]

        # background signal rate
        C_b = C_sr + C_z + C_ez + C_bl + C_dc + C_rn

        # for characterization, Cb must include the planet
        # C_sp = spatial structure to the speckle including post-processing contrast
        # factor and stability factor
        if not (mode["detectionMode"]):
            C_b = C_b + C_p
            C_sp = C_sr * TL.PostProcessing.ppFact_char(WA) * self.stabilityFact
        else:
            C_sp = C_sr * TL.PostProcessing.ppFact(WA) * self.stabilityFact

        if returnExtra:
            # organize components into an optional fourth result
            C_extra = dict(
                C_sr=C_sr.to("1/s"),
                C_z=C_z.to("1/s"),
                C_ez=C_ez.to("1/s"),
                C_dc=C_dc.to("1/s"),
                C_rn=C_rn.to("1/s"),
                C_star=C_star.to("1/s"),
                C_bl=C_bl.to("1/s"),
            )
            return C_p.to("1/s"), C_b.to("1/s"), C_sp.to("1/s"), C_extra
        else:
            return C_p.to("1/s"), C_b.to("1/s"), C_sp.to("1/s")

    def Cp_Cb_Csp_helper(self, TL, sInds, fZ, fEZ, dMag, WA, mode):
        """Helper method for Cp_Cb_Csp that performs lots of common computations
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

        Returns:
            tuple:
                C_star (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Non-coronagraphic star count rate (1/s)
                C_p0 (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Planet count rate (1/s)
                C_sr (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Starlight residual count rate (1/s)
                C_z (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Local zodi count rate (1/s)
                C_ez (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Exozodi count rate (1/s)
                C_dc (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Dark current count rate (1/s)
                C_bl (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Background leak count rate (1/s)'
                Npix (float):
                    Number of pixels in photometric aperture
        """

        # get scienceInstrument and starlightSuppressionSystem and wavelength
        inst = mode["inst"]
        syst = mode["syst"]
        lam = mode["lam"]

        # coronagraph parameters
        occ_trans = syst["occ_trans"](lam, WA)
        core_thruput = syst["core_thruput"](lam, WA)
        Omega = syst["core_area"](lam, WA)

        # number of pixels per lenslet
        pixPerLens = inst["lenslSamp"] ** 2.0

        # number of detector pixels in the photometric aperture = Omega / theta^2
        Npix = pixPerLens * (Omega / inst["pixelScale"] ** 2.0).decompose().value

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
            core_platescale = syst["core_platescale"].copy()
            # furthermore, if we're a coronagraph, we have to scale by wavelength
            if not (syst["occulter"]) and (syst["lam"] != mode["lam"]):
                core_platescale *= mode["lam"] / syst["lam"]

            # core_intensity is the mean intensity times the number of map pixels
            core_intensity = core_mean_intensity * Omega / core_platescale**2

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

        # Star fluxes (ph/m^2/s)
        flux_star = TL.starFlux(sInds, mode)

        # ELECTRON COUNT RATES [ s^-1 ]
        # non-coronagraphic star counts
        C_star = flux_star * mode["losses"]
        # planet counts:
        C_p0 = (C_star * 10.0 ** (-0.4 * dMag) * core_thruput).to("1/s")
        # starlight residual
        C_sr = (C_star * core_intensity).to("1/s")
        # zodiacal light
        C_z = (mode["F0"] * mode["losses"] * fZ * Omega * occ_trans).to("1/s")
        # exozodiacal light
        if self.use_core_thruput_for_ez:
            C_ez = (mode["F0"] * mode["losses"] * fEZ * Omega * core_thruput).to("1/s")
        else:
            C_ez = (mode["F0"] * mode["losses"] * fEZ * Omega * occ_trans).to("1/s")
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

            C_bl = (cleaks + bleaks) * C_star * core_thruput
        else:
            C_bl = np.zeros(len(sInds)) / u.s

        return C_star, C_p0, C_sr, C_z, C_ez, C_dc, C_bl, Npix

    def calc_intTime(self, TL, sInds, fZ, fEZ, dMag, WA, mode, TK=None):
        """Finds integration time to reach a given dMag at a particular WA with given
        local and exozodi values for specific targets and for a specific observing mode.


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
                Integration times

        .. note::

            All infeasible integration times are returned as NaN values

        """
        # count rates
        C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dMag, WA, mode, TK=TK)

        # get SNR threshold
        SNR = mode["SNR"]

        with np.errstate(divide="ignore", invalid="ignore"):
            intTime = np.true_divide(
                SNR**2.0 * C_b, (C_p**2.0 - (SNR * C_sp) ** 2.0)
            ).to("day")

        # infinite and negative values are set to NAN
        intTime[np.isinf(intTime) | (intTime.value < 0.0)] = np.nan

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

        .. warning::

            The prototype implementation assumes the exact same integration time model
            as the other prototype methods (specifically Cp_Cb_Csp and calc_intTime).
            If either of these is overloaded, and, in particular, if C_b and/or C_sp are
            not modeled as independent of C_p, then the analytical approach used here
            will *not* work and must be replaced with numerical inversion.

        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)

        if (C_b is None) or (C_sp is None):
            _, C_b, C_sp = self.Cp_Cb_Csp(
                TL, sInds, fZ, fEZ, np.zeros(len(sInds)), WA, mode, TK=TK
            )

        C_p = mode["SNR"] * np.sqrt(C_sp**2 + C_b / intTimes)  # planet count rate
        core_thruput = mode["syst"]["core_thruput"](mode["lam"], WA)
        flux_star = TL.starFlux(sInds, mode)

        dMag = -2.5 * np.log10(C_p / (flux_star * mode["losses"] * core_thruput))

        return dMag.value

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
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)

        if (C_b is None) or (C_sp is None):
            _, C_b, C_sp = self.Cp_Cb_Csp(
                TL, sInds, fZ, fEZ, np.zeros(len(sInds)), WA, mode, TK=TK
            )

        ddMagdt = 5 / 4 / np.log(10) * C_b / (C_b * intTimes + (C_sp * intTimes) ** 2)

        return ddMagdt.to("1/s")

    def calc_saturation_dMag(self, TL, sInds, fZ, fEZ, WA, mode, TK=None):
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
            fEZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            TK (:ref:`TimeKeeping`, optional):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.

        Returns:
            ~numpy.ndarray(float):
                Saturation (maximum achievable) dMag for each target star
        """

        _, C_b, C_sp = self.Cp_Cb_Csp(
            TL, sInds, fZ, fEZ, np.zeros(len(sInds)), WA, mode, TK=TK
        )

        flux_star = TL.starFlux(sInds, mode)
        core_thruput = mode["syst"]["core_thruput"](mode["lam"], WA)

        dMagmax = -2.5 * np.log10(
            mode["SNR"] * C_sp / (flux_star * mode["losses"] * core_thruput)
        )

        return dMagmax.value
