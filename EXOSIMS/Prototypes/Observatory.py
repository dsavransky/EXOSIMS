# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.eccanom import eccanom
from EXOSIMS.util.get_dirs import get_cache_dir
from EXOSIMS.util.get_dirs import get_downloads_dir
import EXOSIMS.Prototypes.Observatory
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
import pickle
import hashlib
import os
from tqdm import tqdm
from urllib.request import urlretrieve
from inspect import getfullargspec as getargspec
from EXOSIMS.util._numpy_compat import copy_if_needed


class Observatory(object):
    """:ref:`Observatory` Prototype

    Args:
        SRP (bool):
            Toggle solar radiation pressure.  Defaults True.
        koAngles_SolarPanel (list(float)):
            [Min, Max] keepout angles (in degrees) due to solar panels.
            Defaults to [0,180].
        ko_dtStep (float):
            Step size to use when calculating keepout maps (in days). Defaults to 1.
        settlingTime (float):
            Observatory settling time after retargeting (in days). Defaults to 1. This
            time is added to every observation and counts againts the total integration
            time allocation.
        thrust (float):
            Slew thrust mangitude (in mN). Defaults to 450 mN.
        slewIsp (float):
            Slew specific impulse (in seconds). Defaults to 4160 s.
        scMass (float):
            Maneuvering spacecraft initial wet mass (in kg). Nominally this is the
            starshade, but may also be the observatory if the starshade is kept on the
            stable orbit. Defaults to 6000 kg.

            .. warning::

                If ``twotanks`` is true, this input will be ignored and attribute
                ``scMass`` will be initially set to the sum of ``slewMass`` and
                ``skMass``.

        slewMass (float):
            Initial fuel mass of slewing propulsion system (in kg). Defaults to 0. Only
            used if twotanks is True.
        skMass (float):
            Initial fuel mass of stationkeeping propulsion system (in kg).
            Defaults to 0. Only used if twotanks is True.
        twotanks (bool):
            Determines whether stationkeeping and slewing propulsion systems use
            separate tanks. If False, it is assumed that all onboard fuel is fungible.
            Defaults False.
        skEff (float):
            Stationkeeping propulsion system efficiency. Must be between 0 and 1.
            Defaults to 0.7098 (approximately 45 deg cosine losses).
        slewEff (float):
            Slewing propulsion system efficiency. Must be between 0 and 1.
            Defaults to 1.
        dryMass (float):
            Maneuvering spacecraft dry mass (in kg).  Defaults to 3400 kg.
            Must be smaller than scMass.
        coMass (float):
            Non-manuevering spacecraft (nominally the observatory) initial wet mass
            (in kg). Defaults to 5800 kg.
        occulterSep (float):
            Initial occulter separation (in km). Defaults to 55000.
        skIsp (float):
            Stationkeeping propulsion system specific impulse (in seconds).
            Defaults to 220 s.
        defburnPortion (float):
            Default burn portion for simple model slews.  Must be between 0 and 1.
            Defaults to 0.05.
        constTOF (float):
            Constant time of flight value (in days). Defaults to 14. DEPRECATED
        maxdVpct (float):
            Maximum delta V percentage allowed for any maneuver.
            Must be between 0 and 1. Defaults to 0.02.
        spkpath (str, optional):
            Path to SPK file on disk.
            If not set, defaults to de432s.bsp in :ref:`EXOSIMSDOWNLOADS`.
        checkKeepoutEnd (bool):
            Check keepout conditions at end of observation.  Defaults True.
            TODO: Move to SurveySimulation
        forceStaticEphem (bool):
            Use static ephemerides for solar system objects instead of jplephem.
            Defaults False.
        occ_dtmin (float):
            Minimum slew time (in days). Defaults to 0.055
        occ_dtmax (float):
            Maximum slew time (in days). Defaults to 61
        sk_Tmin (float):
            Minimum time after mission start to compute stationkeeping (in days).
            Defaults to 0.
        sk_Tmax (float):
            Maximum  time after mission start to compute stationkeeping (in days).
            Defaults to 365
        non_lambertian_coefficient_front (float):
            Non-Lambertion reflectivity coefficient of front face of manuevering
            spacecraft. Used for SRP calculations. Defaults to 0.038.
        non_lambertian_coefficient_back (float):
            Non-Lambertion reflectivity coefficient of back face of manuevering
            spacecraft. Used for SRP calculations. Defaults to 0.004.
        specular_reflection_factor (float):
            Specular reflectivity of maneuvering spacecraft. Used for SRP calculations.
            Defaults to 0.975.
        nreflection_coefficient (float):
            non-specular reflectivity of maneuvering spacecraft.
            Used for SRP calculations. Defaults to 0.999
        emission_coefficient_front (float):
            Emission coefficient of front face of maneuvering spacecraft.
            Used for SRP calculations. Defaults to 0.8
        emission_coefficient_back (float):
            Emission coefficient of rear face of maneuvering spacecraft.
            Used for SRP calculations. Defaults to 0.2
        allowRefueling (bool):
            Fuel tanks can be topped off when they reach empty from external fuel source
            whose capacity is defined by the ``external_fuel_mass`` input.
            Defaults False.
        external_fuel_mass (float):
            Initial mass of external fuel supply (in kg). Ignored if ``allowRefueling``
            is False. Defaults to 0.
        cachedir (str, optional):
            Full path to cachedir.
            If None (default) use default (see :ref:`EXOSIMSCACHE`)
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        _outspec (dict):
            :ref:`sec:outspec`
        ao (astropy.units.quantity.Quantity):
            Thurst acceleration (current thrust/spacecraft mass). Acceleration units.
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)
        checkKeepoutEnd (bool):
            Toggle checking of keepout at end of observations (as well as the
            beginning). TODO: need to depricate in favor of continuous visibility.
        coMass (astropy.units.quantity.Quantity):
            Non-manuevering spacecraft (nominally the observatory) wet mass. Mass units.
        constTOF (astropy.units.quantity.Quantity):
            Constant time of flight for single occulter slew. DEPRECATED
        defburnPortion (float):
            Default burn portion for simple slew model.
        dryMass (astropy.units.quantity.Quantity):
            Maneuvering spacecraft dry mass. Mass units.
        dVmax (astropy.units.quantity.Quantity):
            Maximum single-slew allowable delta V (as determined by maxdVpct input.
            Units of velocity.
        dVtot (astropy.units.quantity.Quantity):
            Total possible slew delta V as determined by ideal rocket equation applied
            to the initial fuel mass.
        emission_coefficient_back (float):
            Emission coefficient of back face of maneuvering spacecraft.
            Used for SRP calculations.
        emission_coefficient_front (float):
            Emission coefficient of front face of maneuvering spacecraft.
            Used for SRP calculations.
        flowRate (astropy.units.quantity.Quantity):
            Slew ropulsion system mass flow rate. Units: mass/time
        forceStaticEphem (bool):
            Use static ephemerides for solar system objects instead of jplephem.
        havejplephem (bool):
            jplephem module installed and SPK available.
        kernel (jplephem.spk.SPK):
            jplephem kernel used for ephemeris calculations of solar system bodies
        ko_dtStep (astropy.units.quantity.Quantity):
            Step size to use when calculating keepout maps. Time units.
        koAngles_SolarPanel (astropy.units.quantity.Quantity):
            [Min, Max] keepout angles due to solar panels.
        maxdVpct (float):
            Maximum delta V percentage allowed for any maneuver.
        maxFuelMass (astropy.units.quantity.Quantity):
            Total capacity of all fuel.  This parameter is constant and should never
            be modified externally.
        non_lambertian_coefficient_back (float):
            Non-Lambertion reflectivity coefficient of back face of manuevering
            spacecraft. Used for SRP calculations.
        non_lambertian_coefficient_front (float):
            Non-Lambertion reflectivity coefficient of front face of manuevering
            spacecraft. Used for SRP calculations.
        nreflection_coefficient (float):
            Non-specular reflectivity of maneuvering spacecraft.
            Used for SRP calculations.
        occ_dtmax (astropy.units.quantity.Quantity):
            Maximum allowable slew time
        occ_dtmin (astropy.units.quantity.Quantity):
            Minimum allowable slew time.
        occulterSep (astropy.units.quantity.Quantity):
            Current occulter separation distance.
        scMass (astropy.units.quantity.Quantity):
            Current maneuvering spacecraft mass.
        settlingTime (astropy.units.quantity.Quantity):
            Observatory settling time after every retargeting.
        sk_Tmax (astropy.units.quantity.Quantity):
            Maximum time after mission start to compute stationkeeping
        sk_Tmin (astropy.units.quantity.Quantity):
            Minimum time after mission start to compute stationkeeping
        skEff (float):
            Stationkeeping propulsion system efficience.
        skIsp (astropy.units.quantity.Quantity):
            Stationkeeping propulsion system specific impulse. Time units.
        skMass (astropy.units.quantity.Quantity):
            Stationkeeping propulsion system fuel mass.
        skMaxFuelMass (astropy.units.quantity.Quantity):
            Total capacity of stationkeeping fuel tank.  This parameter is constant
            and should never be modified externally. Set to 0 if allowRefueling is
            False.
        slewEff (float):
            Slew propulsion system efficiencey.
        slewIsp (astropy.units.quantity.Quantity):
            Slew propulsion system specific impulse. Time units.
        slewMass (astropy.units.quantity.Quantity):
            Slew propulsion system fuel mass.
        slesMaxFuelMass (astropy.units.quantity.Quantity):
            Total capacity of slewing fuel tank.  This parameter is constant
            and should never be modified externally. Set to 0 if allowRefueling is
            False.
        specular_reflection_factor (float):
            Specular reflectivity of maneuvering spacecraft. Used for SRP calculations.
        spkpath (str):
            Full path to SPK file used by jplephem.
        SRP (bool):
            Toggles whether solar radiation pressure is included in calculations.
        thrust (astropy.units.quantity.Quantity):
            Slew propulsion system thrust. Force units.
        twotanks (bool):
            Toggles whether stationkeeping and slewing fuel are bookkept separately. If
            false, all fuel is fungible.


    .. note::

        For finding positions of solar system bodies, this routine will attempt to
        use the jplephem module and a local SPK file on disk.  The module can be
        installed via pip or from source.  The default SPK file  (which the code
        attempts to automatically download) can be downloaded manually from:
        http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp
        and should be placed in the :ref:`EXOSIMSDOWNLOADS` (or another path, specified
        by the ``spkpath`` input).

    """

    _modtype = "Observatory"

    def __init__(
        self,
        SRP=True,
        koAngles_SolarPanel=[0, 180],
        ko_dtStep=1,
        settlingTime=1,
        thrust=450,
        slewIsp=4160.0,
        scMass=6000.0,
        slewMass=0.0,
        skMass=0.0,
        twotanks=False,
        skEff=0.7098,
        slewEff=1.0,
        dryMass=3400.0,
        coMass=5800.0,
        occulterSep=55000.0,
        skIsp=220.0,
        defburnPortion=0.05,
        constTOF=14,
        maxdVpct=0.02,
        spkpath=None,
        checkKeepoutEnd=True,
        forceStaticEphem=False,
        occ_dtmin=0.055,
        occ_dtmax=61.0,
        sk_Tmin=0.0,
        sk_Tmax=365.0,
        non_lambertian_coefficient_front=0.038,
        non_lambertian_coefficient_back=0.004,
        specular_reflection_factor=0.975,
        nreflection_coefficient=0.999,
        emission_coefficient_front=0.8,
        emission_coefficient_back=0.2,
        allowRefueling=False,
        external_fuel_mass=0,
        cachedir=None,
        **specs,
    ):

        # start the outspec
        self._outspec = {}

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

        # validate inputs
        assert isinstance(checkKeepoutEnd, bool), "checkKeepoutEnd must be a boolean."
        assert isinstance(forceStaticEphem, bool), "forceStaticEphem must be a boolean."

        # default Observatory values
        self.SRP = SRP
        # solar panel keepout angles
        self.koAngles_SolarPanel = [float(x) for x in koAngles_SolarPanel] * u.deg
        # time step for generating koMap of stars (day)
        self.ko_dtStep = float(ko_dtStep) * u.d
        # Observatory settling time after repoint
        self.settlingTime = float(settlingTime) * u.d
        # occulter slew thrust (mN)
        self.thrust = float(thrust) * u.mN
        # occulter slew specific impulse (s)
        self.slewIsp = float(slewIsp) * u.s
        # occulter initial (wet) mass (kg)
        self.scMass = float(scMass) * u.kg
        # slew fuel initial mass (kg)
        self.slewMass = float(slewMass) * u.kg
        # station keeping fuel initial mass (kg)
        self.skMass = float(skMass) * u.kg
        # boolean used to seperate manuevering fuel
        self.twotanks = bool(twotanks)
        # slew efficiency factor
        self.slewEff = float(slewEff)
        # station-keeping efficiency factor
        self.skEff = float(skEff)
        # occulter dry mass (kg)
        self.dryMass = float(dryMass) * u.kg
        # telescope mass (kg)
        self.coMass = float(coMass) * u.kg
        # occulter-telescope distance (km)
        self.occulterSep = float(occulterSep) * u.km
        # station-keeping Isp (s)
        self.skIsp = float(skIsp) * u.s
        # default burn portion
        self.defburnPortion = float(defburnPortion)
        # true if keepout called at obs end
        self.checkKeepoutEnd = bool(checkKeepoutEnd)
        # boolean used to force static ephemerides
        self.forceStaticEphem = bool(forceStaticEphem)
        # starshade constant slew time (days)
        self.constTOF = np.array(constTOF, ndmin=1) * u.d
        # Minimum occulter slew time (days)
        self.occ_dtmin = float(occ_dtmin) * u.d
        # Maximum occulter slew time (days)
        self.occ_dtmax = float(occ_dtmax) * u.d
        # Minimum days after missionstart to calculate stationkeeping (days)
        self.sk_Tmin = float(sk_Tmin) * u.d
        # Maximum days after missionstart to calculate stationkeeping (days)
        self.sk_Tmax = float(sk_Tmax) * u.d
        # Maximum deltaV percent
        self.maxdVpct = float(maxdVpct)
        # non-Lambertian coefficient (front)
        self.non_lambertian_coefficient_front = float(non_lambertian_coefficient_front)
        # non-Lambertian coefficient (back)
        self.non_lambertian_coefficient_back = float(non_lambertian_coefficient_back)
        # specular reflection factor
        self.specular_reflection_factor = float(specular_reflection_factor)
        # nreflection coefficient
        self.nreflection_coefficient = float(nreflection_coefficient)
        # emission coefficient (front)
        self.emission_coefficient_front = float(emission_coefficient_front)
        # emission coefficient (back)
        self.emission_coefficient_back = float(emission_coefficient_back)
        # Allow Refueling from external tank
        self.allowRefueling = bool(allowRefueling)
        # initial mass of external tank
        self.external_fuel_mass = float(external_fuel_mass) * u.kg

        # check that twotanks and dry mass add up to total mass
        if self.twotanks:
            assert (self.slewMass > 0) and (
                self.skMass > 0
            ), "Tank mass must be positive"
            self.scMass = self.slewMass + self.skMass + self.dryMass

            # record tank capacities in case you need to refuel
            self.skMaxFuelMass = self.skMass.copy()
            self.slewMaxFuelMass = self.slewMass.copy()
        else:
            self.skMaxFuelMass = 0 * u.kg
            self.slewMaxFuelMass = 0 * u.kg

        # total tank capacity:
        self.maxFuelMass = self.scMass - self.dryMass
        assert (
            self.maxFuelMass > 0 * u.kg
        ), "Initial spacecraft wet mass must be greater than dry mass."

        # Acceleration
        self.ao = self.thrust / self.scMass

        # find the cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        specs["cachedir"] = self.cachedir

        # find amount of fuel on board starshade and an upper bound for single slew dV
        self.dVtot = self.slewIsp * const.g0 * np.log(self.scMass / self.dryMass)
        self.dVmax = self.dVtot * self.maxdVpct

        # set values derived from quantities above
        # slew flow rate (kg/day)
        self.flowRate = (self.thrust / self.slewEff / const.g0 / self.slewIsp).to(
            "kg/day"
        )

        # if jplephem is available, we'll use that for propagating solar system bodies
        # otherwise, use static ephemerides
        if self.forceStaticEphem is False:
            try:
                from jplephem.spk import SPK

                self.havejplephem = True
            except ImportError:
                self.vprint(
                    "WARNING: Module jplephem not found, "
                    + "using static solar system ephemerides."
                )
                self.havejplephem = False
        else:
            self.havejplephem = False
            self.vprint("Using static solar system ephemerides.")

        # define function for calculating obliquity of the ecliptic
        # (arg Julian centuries from J2000)
        self.obe = (
            lambda TDB: 23.439279
            - 0.0130102 * TDB
            - 5.086e-8 * (TDB**2.0)
            + 5.565e-7 * (TDB**3.0)
            + 1.6e-10 * (TDB**4.0)
            + 1.21e-11 * (TDB**5.0)
        )

        # if you have jplephem, load spice file, otherwise load static ephem
        if self.havejplephem:
            if (spkpath is None) or not (os.path.exists(spkpath)):
                # if the path does not exist, load the default de432s.bsp

                filename = "de432s.bsp"
                downloadsdir = get_downloads_dir()
                spkpath = os.path.join(downloadsdir, filename)
                # attempt to fetch ephemeris from NAIF
                if not os.path.exists(spkpath) and os.access(
                    downloadsdir, os.W_OK | os.X_OK
                ):
                    spk_on_web = (
                        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
                        "spk/planets/de432s.bsp"
                    )
                    self.vprint(
                        "Fetching planetary ephemeris from %s to %s"
                        % (spk_on_web, spkpath)
                    )
                    urlretrieve(spk_on_web, spkpath)
            self.kernel = SPK.open(spkpath)
        else:
            # All ephemeride data from Vallado Appendix D.4
            # Values are: a = sma (AU), e = eccentricity, I = inclination (deg),
            #            O = long. ascending node (deg), w = long. perihelion (deg),
            #            lM = mean longitude (deg)

            # store ephemerides data in heliocentric true ecliptic frame
            a = 0.387098310
            e = [0.20563175, 0.000020406, -0.0000000284, -0.00000000017]
            I = [7.004986, -0.0059516, 0.00000081, 0.000000041]  # noqa: E741
            O = [48.330893, -0.1254229, -0.00008833, -0.000000196]  # noqa: E741
            w = [77.456119, 0.1588643, -0.00001343, 0.000000039]
            lM = [252.250906, 149472.6746358, -0.00000535, 0.000000002]
            Mercury = self.SolarEph(a, e, I, O, w, lM)

            a = 0.723329820
            e = [0.00677188, -0.000047766, 0.0000000975, 0.00000000044]
            I = [3.394662, -0.0008568, -0.00003244, 0.000000010]  # noqa: E741
            O = [76.679920, -0.2780080, -0.00014256, -0.000000198]  # noqa: E741
            w = [131.563707, 0.0048646, -0.00138232, -0.000005332]
            lM = [181.979801, 58517.8156760, 0.00000165, -0.000000002]
            Venus = self.SolarEph(a, e, I, O, w, lM)

            a = 1.000001018
            e = [0.01670862, -0.000042037, -0.0000001236, 0.00000000004]
            I = [0.0, 0.0130546, -0.00000931, -0.000000034]  # noqa: E741
            O = [174.873174, -0.2410908, 0.00004067, -0.000001327]  # noqa: E741
            w = [102.937348, 0.3225557, 0.00015026, 0.000000478]
            lM = [100.466449, 35999.3728519, -0.00000568, 0.0]
            Earth = self.SolarEph(a, e, I, O, w, lM)

            a = 1.523679342
            e = [0.09340062, 0.000090483, -0.0000000806, -0.00000000035]
            I = [1.849726, -0.0081479, -0.00002255, -0.000000027]  # noqa: E741
            O = [49.558093, -0.2949846, -0.00063993, -0.000002143]  # noqa: E741
            w = [336.060234, 0.4438898, -0.00017321, 0.000000300]
            lM = [355.433275, 19140.2993313, 0.00000261, -0.000000003]
            Mars = self.SolarEph(a, e, I, O, w, lM)

            a = [5.202603191, 0.0000001913]
            e = [0.04849485, 0.000163244, -0.0000004719, -0.00000000197]
            I = [1.303270, -0.0019872, 0.00003318, 0.000000092]  # noqa: E741
            O = [100.464441, 0.1766828, 0.00090387, -0.000007032]  # noqa: E741
            w = [14.331309, 0.2155525, 0.00072252, -0.000004590]
            lM = [34.351484, 3034.9056746, -0.00008501, 0.000000004]
            Jupiter = self.SolarEph(a, e, I, O, w, lM)

            a = [9.554909596, -0.0000021389]
            e = [0.05550862, -0.000346818, -0.0000006456, 0.00000000338]
            I = [2.488878, 0.0025515, -0.00004903, 0.000000018]  # noqa: E741
            O = [113.665524, -0.2566649, -0.00018345, 0.000000357]  # noqa: E741
            w = [93.056787, 0.5665496, 0.00052809, 0.000004882]
            lM = [50.077471, 1222.1137943, 0.00021004, -0.000000019]
            Saturn = self.SolarEph(a, e, I, O, w, lM)

            a = [19.218446062, -0.0000000372, 0.00000000098]
            e = [0.04629590, -0.000027337, 0.0000000790, 0.00000000025]
            I = [0.773196, -0.0016869, 0.00000349, 0.000000016]  # noqa: E741
            O = [74.005947, 0.0741461, 0.00040540, 0.000000104]  # noqa: E741
            w = [173.005159, 0.0893206, -0.00009470, 0.000000413]
            lM = [314.055005, 428.4669983, -0.00000486, 0.000000006]
            Uranus = self.SolarEph(a, e, I, O, w, lM)

            a = [30.110386869, -0.0000001663, 0.00000000069]
            e = [0.00898809, 0.000006408, -0.0000000008]
            I = [1.769952, 0.0002257, 0.00000023, -0.000000000]  # noqa: E741
            O = [131.784057, -0.0061651, -0.00000219, -0.000000078]  # noqa: E741
            w = [48.123691, 0.0291587, 0.00007051, 0.0]
            lM = [304.348665, 218.4862002, 0.00000059, -0.000000002]
            Neptune = self.SolarEph(a, e, I, O, w, lM)

            a = [39.48168677, -0.00076912]
            e = [0.24880766, 0.00006465]
            I = [17.14175, 0.003075]  # noqa: E741
            O = [110.30347, -0.01036944]  # noqa: E741
            w = [224.06676, -0.03673611]
            lM = [238.92881, 145.2078]
            Pluto = self.SolarEph(a, e, I, O, w, lM)

            # store all as dictionary:
            self.planets = {
                "Mercury": Mercury,
                "Venus": Venus,
                "Earth": Earth,
                "Mars": Mars,
                "Jupiter": Jupiter,
                "Saturn": Saturn,
                "Uranus": Uranus,
                "Neptune": Neptune,
                "Pluto": Pluto,
            }

        self.spkpath = spkpath

        # populate outspec
        inputatts = getargspec(EXOSIMS.Prototypes.Observatory.Observatory.__init__)[0]
        if "self" in inputatts:
            inputatts.remove("self")

        for att in inputatts:
            dat = self.__dict__[att]
            self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat

    def __del__(self):
        """destructor method.  only here to clean up SPK kernel if it exists."""
        if "kernel" in self.__dict__:
            if self.kernel:
                self.kernel.close()

    def __str__(self):
        """String representation of the Observatory object

        When the command 'print' is used on the Observatory object, this method
        will print the attribute values contained in the object

        """

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Observatory class object attributes"

    def equat2eclip(self, r_equat, currentTime, rotsign=1):
        """Rotates heliocentric coordinates from equatorial to ecliptic frame.

        Args:
            r_equat (~astropy.units.Quantity(~numpy.ndarray(float))):
                Positions vector in heliocentric equatorial frame in units of AU. nx3
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD
            rotsign (int):
                Optional flag, default 1, set -1 to reverse the rotation

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Positions vector in heliocentric ecliptic frame in units of AU. nx3
        """

        # check size of arrays
        assert currentTime.size == 1 or currentTime.size == len(
            r_equat
        ), "If multiple times and positions, currentTime and r_equat sizes must match"
        # find Julian centuries from J2000
        TDB = self.cent(currentTime)
        # find obliquity of the ecliptic
        obe = rotsign * np.array(np.radians(self.obe(TDB)), ndmin=1)
        # positions vector in heliocentric ecliptic frame
        if currentTime.size == 1:
            r_eclip = (
                np.array(
                    [
                        np.dot(self.rot(obe[0], 1), r_equat[x, :].to("AU").value)
                        for x in range(len(r_equat))
                    ]
                )
                * u.AU
            )
        else:
            r_eclip = (
                np.array(
                    [
                        np.dot(self.rot(obe[x], 1), r_equat[x, :].to("AU").value)
                        for x in range(len(r_equat))
                    ]
                )
                * u.AU
            )

        return r_eclip

    def eclip2equat(self, r_eclip, currentTime):
        """Rotates heliocentric coordinates from ecliptic to equatorial frame.

        Args:
            r_eclip (~astropy.units.Quantity(~numpy.ndarray(float))):
                Positions vector in heliocentric ecliptic frame in units of AU
            currentTime (astropy.time.Time):
                Current absolute mission time in MJD

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Positions vector in heliocentric equatorial frame in units of AU. nx3

        """

        r_equat = self.equat2eclip(r_eclip, currentTime, rotsign=-1)

        return r_equat

    def orbit(self, currentTime, eclip=False):
        """Finds observatory orbit positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).

        This method returns the telescope geosynchronous circular orbit position vector.

        Args:
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD
            eclip (bool):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to
                False, corresponding to heliocentric equatorial frame.

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Observatory orbit positions vector in heliocentric equatorial (default)
                or ecliptic frame in units of AU. nx3

        Note:
            Use eclip=True to get ecliptic coordinates.

        """

        mjdtime = np.array(currentTime.mjd, ndmin=1)  # modified julian day time
        t = mjdtime % 1  # gives percent of day
        f = 2.0 * np.pi  # orbital frequency (2*pi/sideral day)
        r = (42164.0 * u.km).to("AU").value  # orbital height (convert from km to AU)
        I = np.radians(28.5)  # noqa: E741  # orbital inclination in degrees
        O = np.radians(228.0)  # noqa: E741  # right ascension of the ascending node

        # observatory positions vector wrt Earth in orbital plane
        r_orb = r * np.array([np.cos(f * t), np.sin(f * t), np.zeros(t.size)])
        # observatory positions vector wrt Earth in equatorial frame
        r_obs_earth = np.dot(np.dot(self.rot(-O, 3), self.rot(-I, 1)), r_orb).T * u.AU
        # Earth positions vector in heliocentric equatorial frame
        r_earth = self.solarSystem_body_position(currentTime, "Earth")
        # observatory positions vector in heliocentric equatorial frame
        r_obs = (r_obs_earth + r_earth).to("AU")

        assert np.all(
            np.isfinite(r_obs)
        ), "Observatory positions vector r_obs has infinite value."

        if eclip:
            # observatory positions vector in heliocentric ecliptic frame
            r_obs = self.equat2eclip(r_obs, currentTime)

        return r_obs

    def keepout(self, TL, sInds, currentTime, koangles, returnExtra=False):
        """Finds keepout Boolean values for stars of interest.

        This method returns the keepout Boolean values for stars of interest, where
        True is an observable star.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD
                MAY ONLY BE ONE VALUE OR DUPLICATES OF THE SAME VALUE
            koangles (~astropy.units.Quantity(~numpy.ndarray(float))):
                s x 4 x 2 array where s is the number of starlight suppression systems
                as defined in the Optical System. Each of the remaining 4 x 2 arrays
                are system specific koAngles for the Sun, Moon, Earth, and small bodies
                (4), each with a minimum and maximum value (2) in units of deg.
            returnExtra (bool):
                Optional flag, default False, set True to return additional information.

        Returns:
            tuple or ~numpy.ndarray(bool):
                kogood (~numpy.ndarray(bool)):
                    kogood s x n x m array of boolean values. True is a target
                    unobstructed and observable, and False is a target unobservable due
                    to obstructions in the keepout zone.
                r_body (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Only returned if returnExtra is True
                    11 x m x 3 array where m is len(currentTime) of heliocentric
                    equatorial Cartesian elements of the Sun, Moon, Earth and
                    Mercury->Pluto
                r_targ (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Only returned if returnExtra is True
                    m x n x 3 array where m is len(currentTime) or 1 if staticStars is
                    true in TargetList of heliocentric equatorial Cartesian coords of
                    target and n is the len(sInds)
                culprit (numpy.ndarray(int)):
                    Only returned if returnExtra is True
                    s x n x m x 12 array of boolean integer values identifying which
                    body is responsible for keepout (when equal to 1).  m is number of
                    targets and n is len(currentTime). Last dimension is ordered same
                    as r_body, with an extra line for solar panels being the culprit
                koangleArray (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Only returned if returnExtra is True
                    s x 11 x 2 element array of minimum and maximum keepouts used for
                    each body. Same ordering as r_body.

        """

        # if multiple time values, check they are different otherwise reduce to scalar
        if currentTime.size > 1:
            if np.all(currentTime == currentTime[0]):
                currentTime = currentTime[0]

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        # get all array sizes
        nStars = sInds.size
        nTimes = currentTime.size
        nSystems = koangles.shape[0]
        nBodies = 11

        # observatory positions vector in heliocentric equatorial frame
        r_obs = self.orbit(currentTime)  # (m x 3)
        # traget star positions vector in heliocentric equatorial frame
        r_targ = TL.starprop(sInds, currentTime)
        # r_targ = r_targ.reshape(nTimes,nStars,3) # (m x n x 3).
        if TL.staticStars and (nStars == 1):
            # When the stars are not moving the position vectors are always the same
            # so they are tiled because r_targ returns only a 1xnStars array
            r_targ = np.tile(r_targ, (nTimes, 1))
        else:
            r_targ = r_targ.reshape(nTimes, nStars, 3)  # (m x n x 3).
        # body positions vector in heliocentric equatorial frame
        r_body = (
            np.array(
                [
                    self.solarSystem_body_position(currentTime, "Sun").to("AU").value,
                    self.solarSystem_body_position(currentTime, "Moon").to("AU").value,
                    self.solarSystem_body_position(currentTime, "Earth").to("AU").value,
                    self.solarSystem_body_position(currentTime, "Mercury")
                    .to("AU")
                    .value,
                    self.solarSystem_body_position(currentTime, "Venus").to("AU").value,
                    self.solarSystem_body_position(currentTime, "Mars").to("AU").value,
                    self.solarSystem_body_position(currentTime, "Jupiter")
                    .to("AU")
                    .value,
                    self.solarSystem_body_position(currentTime, "Saturn")
                    .to("AU")
                    .value,
                    self.solarSystem_body_position(currentTime, "Uranus")
                    .to("AU")
                    .value,
                    self.solarSystem_body_position(currentTime, "Neptune")
                    .to("AU")
                    .value,
                    self.solarSystem_body_position(currentTime, "Pluto").to("AU").value,
                ]
            )
            * u.AU
        )
        # position vectors wrt spacecraft
        r_targ = (r_targ - r_obs.reshape(nTimes, 1, 3)).to("pc")  # (m  x n x 3)
        r_body = (r_body - r_obs.reshape(1, nTimes, 3)).to("AU")  # (11 x m x 3)
        # unit vectors wrt spacecraft
        u_targ = (r_targ / np.linalg.norm(r_targ, axis=-1, keepdims=True)).value
        u_body = (r_body / np.linalg.norm(r_body, axis=-1, keepdims=True)).value

        # create array of koangles for all bodies, using minimum and maximum keepout
        # angles of each starlight suppression system in the telescope for
        # bright objects (Sun, Moon, Earth, other small bodies)
        koangleArray = np.zeros([nSystems, nBodies, 2])
        koangleArray[:, 0:3, :] = koangles[:, 0:3, :]
        koangleArray[:, 3:, :] = koangles[:, 3, :].reshape(
            nSystems, 1, 2
        )  # small bodies have same values
        koangleArray = koangleArray * u.deg

        # find angles and make angle comparisons to build kogood array:
        # if bright objects have an angle with the target vector less than koangle
        # (e.g. pi/4) they are in the field of view and the target star may not be
        # observed, thus ko associated with this target becomes False.
        kogood = np.ones([nSystems, nStars, nTimes], dtype=bool)  # (s x n x m)
        culprit = np.zeros([nSystems, nStars, nTimes, nBodies + 1])  # (s x n x m x 12)
        # running loop for nSystems, nStars, and nTimes (three loops total)
        for s in tqdm(
            np.arange(nSystems), desc="Starlight Suppression System", position=0
        ):
            for n in tqdm(
                np.arange(nStars), desc="Star Keepout", position=1, leave=False
            ):
                for m in np.arange(nTimes):
                    # unit vectors for the 11 bodies and the nth target at the mth time
                    u_b = u_body[:, m, :]
                    u_t = u_targ[m, n, :]
                    # relative angle between the target and bright body look vectors
                    angles = np.arccos(np.clip(np.dot(u_b, u_t), -1, 1)) * u.rad
                    # create array of "culprits" that prevent a target from being
                    # observed
                    culprit[s, n, m, :-1] = (angles < koangleArray[s, :, 0]) | (
                        angles > koangleArray[s, :, 1]
                    )
                    # adding solar panel restrictions as a final culprit
                    culprit[s, n, m, -1] = (angles[0] < self.koAngles_SolarPanel[0]) | (
                        angles[0] > self.koAngles_SolarPanel[1]
                    )
                    # if any bright body obstructs, kogood becomes False
                    if np.any(culprit[s, n, m, :]):
                        kogood[s, n, m] = False

        # checking that all ko elements are boolean
        trues = [isinstance(element, np.bool_) for element in kogood.flatten()]
        assert all(trues), "An element of kogood is not Boolean"

        if returnExtra:
            return kogood, r_body, r_targ, culprit, koangleArray
        else:
            return kogood

    def generate_koMap(self, TL, missionStart, missionFinishAbs, koangles):
        """Creates keepout map for all targets throughout mission lifetime.

        This method returns a binary map showing when all stars in the given
        target list are in or out of the keepout zone (i.e. when they are not
        observable) from mission start to mission finish.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            missionStart (~astropy.time.Time):
                Absolute start of mission time in MJD
            missionFinishAbs (~astropy.time.Time):
                Absolute end of mission time in MJD
            koangles (~astropy.units.Quantity(~numpy.ndarray(float))):
                s x 4 x 2 array where s is the number of starlight suppression systems
                as defined in the Optical System. Each of the remaining 4 x 2 arrays
                are system specific koAngles for the Sun, Moon, Earth, and small bodies
                (4), each with a minimum and maximum value (2) in units of deg.

        Returns:
            tuple:
                koMap (~numpy.ndarray(bool)):
                    True is a target unobstructed and observable, and False is a
                    target unobservable due to obstructions in the keepout zone.
                koTimes (~astropy.time.Time):
                    Absolute MJD mission times from start to end in steps of 1 d

        """
        # generating hash name
        filename = "koMap_"
        atts = ["koAngles_SolarPanel", "ko_dtStep"]
        extstr = ""
        for att in sorted(atts, key=str.lower):
            if not callable(getattr(self, att)):
                extstr += "%s: " % att + str(getattr(self, att)) + " "
        extstr += "%s: " % "missionStart" + str(missionStart) + " "
        extstr += "%s: " % "missionFinishAbs" + str(missionFinishAbs) + " "
        extstr += "%s: " % "koangles" + str(koangles) + " "
        extstr += "%s: " % "Name" + str(getattr(TL, "Name")) + " "
        # TODO: is this needed?
        # extstr += '%s: ' % 'Name' + TL.StarCatalog.__class__.__name__ + ' '
        extstr += "%s: " % "nStars" + str(getattr(TL, "nStars")) + " "
        ext = hashlib.md5(extstr.encode("utf-8")).hexdigest()
        filename += ext
        koPath = os.path.join(self.cachedir, filename + ".komap")

        # global times when keepout is checked for all stars
        koTimes = np.arange(
            missionStart.value, missionFinishAbs.value, self.ko_dtStep.value
        )
        koTimes = Time(
            koTimes, format="mjd", scale="tai"
        )  # scale must be tai to account for leap seconds

        if os.path.exists(koPath):
            # keepout map already exists for parameters
            self.vprint("Loading cached keepout map file from %s" % koPath)
            try:
                with open(koPath, "rb") as ff:
                    A = pickle.load(ff)
            except UnicodeDecodeError:
                with open(koPath, "rb") as ff:
                    A = pickle.load(ff, encoding="latin1")
            self.vprint("Keepout Map loaded from cache.")
            koMap = A["koMap"]
        else:
            self.vprint('Cached keepout map file not found at "%s".' % koPath)
            # looping over all stars to generate map of when all stars are observable
            self.vprint("Starting keepout calculations for %s stars." % TL.nStars)
            koMap = self.keepout(TL, np.arange(TL.nStars), koTimes, koangles, False)
            A = {"koMap": koMap}
            with open(koPath, "wb") as f:
                pickle.dump(A, f)
            self.vprint("Keepout Map calculations finished")
            self.vprint("Keepout Map array stored in %r" % koPath)

        return koMap, koTimes

    def calculate_observableTimes(self, TL, sInds, currentTime, koMaps, koTimes, mode):
        """Returns the next window of time during which targets are observable

        This method returns a nx2 ndarray of times for every star given in the
        target list. The two entries for every star are the next times (after
        current time) when the star exits and enters keepout (i.e. the start
        and end times of the next window of observability).

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (numpy.ndarray(int)):
                Integer indices of the stars of interest
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD
            koMaps (dict):
                Keepout values for n stars throughout time range of length m,
                key names being the system names specified in mode.
                True is a target unobstructed and observable, and False is a
                target unobservable due to obstructions in the keepout zone.
            koTimes (~astropy.time.Time):
                Absolute MJD mission times from start to end in steps of 1 d
            mode (dict):
                Selected observing mode

        Returns:
            ~astropy.time.Time:
                Start and end times of next observability time window in
                absolute time MJD. n is length of sInds
        """
        # creating time arrays to use in the keepout method (# stars == # times)
        # minimum slew time for occulter to align with new star
        if mode["syst"]["occulter"]:
            nextObTimes = np.ones(len(sInds)) * currentTime.value + self.occ_dtmin.value
        else:
            nextObTimes = np.ones(len(sInds)) * currentTime.value
        nextObTimes = Time(
            nextObTimes, format="mjd", scale="tai"
        )  # converting to astropy MJD time array

        # find appropriate koMap
        systName = mode["syst"]["name"]
        koMap = koMaps[systName]

        # finding observable times
        observableTimes = self.find_nextObsWindow(
            TL, sInds, nextObTimes, koMap, koTimes
        ).value
        observableTimesNorm = (
            observableTimes - nextObTimes.value
        )  # days since currentTime

        # in case of an occulter, correct for short windows
        if mode["syst"]["occulter"]:
            # find length of observable range in days
            observable_range = np.diff(observableTimesNorm, axis=0)[0]
            # re-do calculations for observable windows that are less than
            # dt_min days long
            reDo = np.where(observable_range < self.occ_dtmin.value)[0]
            if reDo.size:
                correctedObTimes = (
                    nextObTimes[reDo].value + observableTimesNorm[1, reDo]
                )
                correctedObTimes = Time(correctedObTimes, format="mjd", scale="tai")
                observableTimes[:, reDo] = self.find_nextObsWindow(
                    TL, reDo, correctedObTimes, koMap, koTimes
                ).value

        return Time(observableTimes, format="mjd", scale="tai")

    def find_nextObsWindow(self, TL, sInds, currentTimes, koMap, koTimes):
        """Method used by calculate_observableTimes for finding next observable windows

        This method returns a nx2 ndarray of times for every star given in the
        target list. The two entries for every star are the next times (after
        current time) when the star exits and enters keepout (i.e. the start
        and end times of the next window of observability).

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            currentTimes (~astropy.time.Time):
                Current absolute mission time in MJD same length as sInds
            koMap (~numpy.ndarray(int)):
                Keepout values for n stars throughout time range of length m (mxn)
            koTimes (astropy.time.Time):
                Absolute MJD mission times from start to end in steps of 1 d
            mode (dict):
                Selected observing mode

        Returns:
            astropy.time.Time(~numpy.ndarray):
                Start and end times of next observability time window in MJD
        """
        # create arrays
        nLoops = len(sInds)
        nextExitTime = np.zeros(nLoops)
        nextEntryTime = np.zeros(nLoops)

        # getting saved time closest to currentTime
        xx = [abs(koTimes - currentTimes[t]).value for t in range(nLoops)]
        xxMin = np.min(xx, axis=1)
        T = np.array([np.where(xx[x] == xxMin[x])[0][0] for x in range(nLoops)])

        # checking to see if stars are in keepout at currentTime
        kogoodStart = [bool(koMap[x, S]) for x, S in zip(sInds, T)]
        kobadStart = [bool(not koMap[x, S]) for x, S in zip(sInds, T)]
        nextExitTime[kogoodStart] = currentTimes[kogoodStart].value

        # finding next entry into keepout for currently observable stars
        for n, S in zip(sInds[kogoodStart], T[kogoodStart]):
            idxG_E = np.where(~koMap[n, S:])

            # enters KO after missionEnd
            if not idxG_E[0].tolist():
                nEnd = np.where(sInds == n)
                nextEntryTime[nEnd] = koTimes[-1].value
            else:
                nextEntry = idxG_E[0][0] + S
                # enters KO after missionEnd (missed these)
                if nextEntry > len(koTimes):
                    nEnd = np.where(sInds == n)
                    nextEntryTime[nEnd] = koTimes[-1].value
                # enters KO before missionEnd
                else:
                    nGood = np.where(sInds == n)
                    nextEntryTime[nGood] = koTimes[nextEntry].value

        # finding next exit and entry of keepout for unobservable stars (in keepout)
        for n, S in zip(sInds[kobadStart], T[kobadStart]):
            idx_X = np.where(koMap[n, S:])

            # exit KO after missionEnd (enter after as well)
            if not idx_X[0].tolist():
                nEnd = np.where(sInds == n)
                nextExitTime[nEnd] = koTimes[-1].value
                nextEntryTime[nEnd] = koTimes[-1].value
            else:
                nextExit = idx_X[0][0] + S
                # exit KO after missionEnd (missed these)
                if nextExit > len(koTimes):
                    nEnd = np.where(sInds == n)
                    nextExitTime[nEnd] = koTimes[-1].value
                    nextEntryTime[nEnd] = koTimes[-1].value
                # exit KO before missionEnd
                else:
                    nBad = np.where(sInds == n)
                    nextExitTime[nBad] = koTimes[nextExit].value

                    idx_E = np.where(~koMap[n, nextExit:])
                    # enters KO again after missionEnd
                    if not idx_E[0].tolist():
                        nEnd = np.where(sInds == n)
                        nextEntryTime[nEnd] = koTimes[-1].value
                    else:
                        nextEntry = idx_E[0][0] + nextExit
                        # enters KO again after missionEnd (missed these)
                        if nextEntry > len(koTimes):
                            nEnd = np.where(sInds == n)
                            nextEntryTime[nEnd] = koTimes[-1].value
                        # enters KO before missionEnd
                        else:
                            nextEntryTime[nBad] = koTimes[nextEntry].value

        observableTimes = np.vstack([nextExitTime, nextEntryTime]) * u.d

        return observableTimes

    def star_angularSep(self, TL, old_sInd, sInds, currentTime):
        """Finds angular separation from old star to given list of stars

        This method returns the angular separation from the last observed
        star to all others on the given list at the currentTime.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            old_sInd (int):
                Integer index of the last star of interest
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD

        Returns:
            float:
                Angular separation between two target stars
        """
        if old_sInd is None:
            sd = np.zeros(len(sInds)) * u.rad
        else:
            # position vector of previous target star
            r_old = TL.starprop(old_sInd, currentTime)[0]
            u_old = r_old.to("AU").value / np.linalg.norm(r_old.to("AU").value)
            # position vector of new target stars
            r_new = TL.starprop(sInds, currentTime)
            u_new = (
                r_new.to("AU").value.T / np.linalg.norm(r_new.to("AU").value, axis=1)
            ).T
            # angle between old and new stars
            sd = np.arccos(np.clip(np.dot(u_old, u_new.T), -1, 1)) * u.rad

            # A-frame
            a1 = u_old / np.linalg.norm(u_old)  # normalized old look vector
            a2 = np.array([a1[1], -a1[0], 0])  # normal to a1
            a3 = np.cross(a1, a2)  # last part of the A basis vectors

            # finding sign of angle
            # The star angular separation can be negative
            u2_Az = np.dot(a3, u_new.T)
            sgn = np.sign(u2_Az)
            sgn[np.where(sgn == 0)] = 1
            sd = sgn * sd

        return sd

    def solarSystem_body_position(self, currentTime, bodyname, eclip=False):
        """Finds solar system body positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).

        This passes all arguments to one of spk_body or keplerplanet, depending
        on the value of self.havejplephem.

        Args:
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD
            bodyname (str):
                Solar system object name
            eclip (bool):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to
                False, corresponding to heliocentric equatorial frame.

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Solar system body positions in heliocentric equatorial (default)
                or ecliptic frame in units of AU. nx3

        Note:
            Use eclip=True to get ecliptic coordinates.

        """

        # heliocentric
        if bodyname == "Sun":
            return np.zeros((currentTime.size, 3)) * u.AU

        # choose JPL or static ephemerides
        if self.havejplephem:
            r_body = self.spk_body(currentTime, bodyname, eclip=eclip).to("AU")
        else:
            r_body = self.keplerplanet(currentTime, bodyname, eclip=eclip).to("AU")

        return r_body

    def spk_body(self, currentTime, bodyname, eclip=False):
        """Finds solar system body positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).

        This method uses spice kernel from NAIF to find heliocentric
        equatorial position vectors for solar system objects.

        Args:
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD
            bodyname (str):
                Solar system object name
            eclip (bool):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to
                False, corresponding to heliocentric equatorial frame.

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Solar system body positions in heliocentric equatorial (default)
                or ecliptic frame in units of AU. nx3

        Note: Use eclip=True to get ecliptic coordinates.

        """

        # dictionary of solar system bodies available in spice kernel (in km)
        bodies = {
            "Mercury": 199,
            "Venus": 299,
            "Earth": 399,
            "Mars": 4,
            "Jupiter": 5,
            "Saturn": 6,
            "Uranus": 7,
            "Neptune": 8,
            "Pluto": 9,
            "Sun": 10,
            "Moon": 301,
        }
        assert bodyname in bodies, "%s is not a recognized body name." % (bodyname)

        # julian day time
        jdtime = np.array(currentTime.jd, ndmin=1)
        # body positions vector in heliocentric equatorial frame
        if bodies[bodyname] == 199:
            r_body = (
                self.kernel[0, 1].compute(jdtime)
                + self.kernel[1, 199].compute(jdtime)
                - self.kernel[0, 10].compute(jdtime)
            )
        elif bodies[bodyname] == 299:
            r_body = (
                self.kernel[0, 2].compute(jdtime)
                + self.kernel[2, 299].compute(jdtime)
                - self.kernel[0, 10].compute(jdtime)
            )
        elif bodies[bodyname] == 399:
            r_body = (
                self.kernel[0, 3].compute(jdtime)
                + self.kernel[3, 399].compute(jdtime)
                - self.kernel[0, 10].compute(jdtime)
            )
        elif bodies[bodyname] == 301:
            r_body = (
                self.kernel[0, 3].compute(jdtime)
                + self.kernel[3, 301].compute(jdtime)
                - self.kernel[0, 10].compute(jdtime)
            )
        else:
            r_body = self.kernel[0, bodies[bodyname]].compute(jdtime) - self.kernel[
                0, 10
            ].compute(jdtime)
        # reshape and convert units
        r_body = (r_body * u.km).T.to("AU")

        if eclip:
            # body positions vector in heliocentric ecliptic frame
            r_body = self.equat2eclip(r_body, currentTime)

        return r_body

    def keplerplanet(self, currentTime, bodyname, eclip=False):
        """Finds solar system body positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).

        This method uses algorithms 2 and 10 from Vallado 2013 to find
        heliocentric equatorial position vectors for solar system objects.

        Args:
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD
            bodyname (str):
                Solar system object name
            eclip (bool):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to
                False, corresponding to heliocentric equatorial frame.

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Solar system body positions in heliocentric equatorial (default)
                or ecliptic frame in units of AU

        Note:
            Use eclip=True to get ecliptic coordinates.

        """

        # Moon positions based on Earth positions
        if bodyname == "Moon":
            r_Earth = self.keplerplanet(currentTime, "Earth")
            return r_Earth + self.moon_earth(currentTime)

        assert bodyname in self.planets, "%s is not a recognized body name." % (
            bodyname
        )

        # find Julian centuries from J2000
        TDB = self.cent(currentTime)
        # update ephemerides data (convert sma from km to AU)
        planet = self.planets[bodyname]
        a = (self.propeph(planet.a, TDB) * u.km).to("AU").value
        e = self.propeph(planet.e, TDB)
        I = np.radians(self.propeph(planet.I, TDB))  # noqa: E741
        O = np.radians(self.propeph(planet.O, TDB))  # noqa: E741
        w = np.radians(self.propeph(planet.w, TDB))
        lM = np.radians(self.propeph(planet.lM, TDB))
        # find mean anomaly and argument of perigee
        M = (lM - w) % (2.0 * np.pi)
        wp = (w - O) % (2.0 * np.pi)
        # find eccentric anomaly
        E = eccanom(M, e)[0]
        # find true anomaly
        nu = np.arctan2(np.sin(E) * np.sqrt(1.0 - e**2.0), np.cos(E) - e)
        # find semiparameter
        p = a * (1.0 - e**2.0)
        # body positions vector in orbital plane
        rx = p * np.cos(nu) / (1.0 + e * np.cos(nu))
        ry = p * np.sin(nu) / (1.0 + e * np.cos(nu))
        rz = np.zeros(currentTime.size)
        r_orb = np.array([rx, ry, rz])
        # body positions vector in heliocentric ecliptic plane
        r_body = (
            np.array(
                [
                    np.dot(
                        np.dot(self.rot(-O[x], 3), self.rot(-I[x], 1)),
                        np.dot(self.rot(-wp[x], 3), r_orb[:, x]),
                    )
                    for x in range(currentTime.size)
                ]
            )
            * u.AU
        )

        if not eclip:
            # body positions vector in heliocentric equatorial frame
            r_body = self.eclip2equat(r_body, currentTime)

        return r_body

    def moon_earth(self, currentTime):
        """Finds geocentric equatorial positions vector for Earth's moon

        This method uses Algorithm 31 from Vallado 2013 to find the geocentric
        equatorial positions vector for Earth's moon.

        Args:
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Geocentric equatorial position vector in units of AU

        """

        TDB = np.array(self.cent(currentTime), ndmin=1)
        la = np.radians(
            218.32
            + 481267.8813 * TDB
            + 6.29 * np.sin(np.radians(134.9 + 477198.85 * TDB))
            - 1.27 * np.sin(np.radians(259.2 - 413335.38 * TDB))
            + 0.66 * np.sin(np.radians(235.7 + 890534.23 * TDB))
            + 0.21 * np.sin(np.radians(269.9 + 954397.70 * TDB))
            - 0.19 * np.sin(np.radians(357.5 + 35999.05 * TDB))
            - 0.11 * np.sin(np.radians(186.6 + 966404.05 * TDB))
        )
        phi = np.radians(
            5.13 * np.sin(np.radians(93.3 + 483202.03 * TDB))
            + 0.28 * np.sin(np.radians(228.2 + 960400.87 * TDB))
            - 0.28 * np.sin(np.radians(318.3 + 6003.18 * TDB))
            - 0.17 * np.sin(np.radians(217.6 - 407332.20 * TDB))
        )
        P = np.radians(
            0.9508
            + 0.0518 * np.cos(np.radians(134.9 + 477198.85 * TDB))
            + 0.0095 * np.cos(np.radians(259.2 - 413335.38 * TDB))
            + 0.0078 * np.cos(np.radians(235.7 + 890534.23 * TDB))
            + 0.0028 * np.cos(np.radians(269.9 + 954397.70 * TDB))
        )
        e = np.radians(
            23.439291 - 0.0130042 * TDB - 1.64e-7 * TDB**2 + 5.04e-7 * TDB**3
        )
        r = 1.0 / np.sin(P) * 6378.137  # km
        r_moon = r * np.array(
            [
                np.cos(phi) * np.cos(la),
                np.cos(e) * np.cos(phi) * np.sin(la) - np.sin(e) * np.sin(phi),
                np.sin(e) * np.cos(phi) * np.sin(la) + np.cos(e) * np.sin(phi),
            ]
        )

        # set format and units
        r_moon = (r_moon * u.km).T.to("AU")

        return r_moon

    def cent(self, currentTime):
        """Finds time in Julian centuries since J2000 epoch

        This quantity is needed for many algorithms from Vallado 2013.

        Args:
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD

        Returns:
            ~numpy.ndarray(float):
                time in Julian centuries since the J2000 epoch

        """

        j2000 = Time(2000.0, format="jyear", scale="tai")
        TDB = (currentTime.jd - j2000.jd) / 36525.0

        return TDB

    def propeph(self, x, TDB):
        """Propagates an ephemeris from Vallado 2013 to current time.

        Args:
            x (list):
                ephemeride list (maximum of 4 elements)
            TDB (float):
                time in Julian centuries since the J2000 epoch

        Returns:
            numpy.darray(float):
                ephemerides value at current time

        """

        if isinstance(x, list):
            if len(x) < 4:
                q = 4 - len(x)
                i = 0
                while i < q:
                    x.append(0.0)
                    i += 1
        elif isinstance(x, float) or isinstance(x, int):
            x = [float(x)]
            i = 0
            while i < 3:
                x.append(0.0)
                i += 1

        # propagated ephem
        y = x[0] + x[1] * TDB + x[2] * (TDB**2) + x[3] * (TDB**3)
        # cast to array
        y = np.array(y, ndmin=1, copy=copy_if_needed)

        return y

    def rot(self, th, axis):
        """Finds the rotation matrix of angle th about the axis value

        Args:
            th (float):
                Rotation angle in radians
            axis (int):
                Integer value denoting rotation axis (1,2, or 3)

        Returns:
            ~numpy.ndarray(float):
                Rotation matrix

        """

        if axis == 1:
            rot_th = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(th), np.sin(th)],
                    [0.0, -np.sin(th), np.cos(th)],
                ]
            )
        elif axis == 2:
            rot_th = np.array(
                [
                    [np.cos(th), 0.0, -np.sin(th)],
                    [0.0, 1.0, 0.0],
                    [np.sin(th), 0.0, np.cos(th)],
                ]
            )
        elif axis == 3:
            rot_th = np.array(
                [
                    [np.cos(th), np.sin(th), 0.0],
                    [-np.sin(th), np.cos(th), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

        return rot_th

    def distForces(self, TL, sInd, currentTime):
        """Finds lateral and axial disturbance forces on an occulter

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInd (int):
                Integer index of the star of interest
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD

        Returns:
            tuple:
                :obj:`~astropy.units.Quantity`:
                    dF_lateral: Lateral disturbance force in units of N
                :obj:`~astropy.units.Quantity`:
                    dF_axial: Axial disturbance force in units of N

        """

        # get spacecraft position vector
        r_obs = self.orbit(currentTime)[0]
        # sun -> earth position vector
        r_Es = self.solarSystem_body_position(currentTime, "Earth")[0]
        # Telescope -> target vector and unit vector
        r_targ = TL.starprop(sInd, currentTime)[0] - r_obs
        u_targ = r_targ.to("AU").value / np.linalg.norm(r_targ.to("AU").value)
        # sun -> occulter vector
        r_Os = r_obs.to("AU") + self.occulterSep.to("AU") * u_targ
        # Earth-Moon barycenter -> spacecraft vectors
        r_TE = r_obs - r_Es
        r_OE = r_Os - r_Es
        # force on occulter
        Mfactor = -self.scMass * const.M_sun * const.G
        F_sO = r_Os / (np.linalg.norm(r_Os.to("AU").value) * r_Os.unit) ** 3.0 * Mfactor
        F_EO = (
            r_OE
            / (np.linalg.norm(r_OE.to("AU").value) * r_OE.unit) ** 3.0
            * Mfactor
            / 328900.56
        )
        F_O = F_sO + F_EO
        # force on telescope
        Mfactor = -self.coMass * const.M_sun * const.G
        F_sT = (
            r_obs / (np.linalg.norm(r_obs.to("AU").value) * r_obs.unit) ** 3.0 * Mfactor
        )
        F_ET = (
            r_TE
            / (np.linalg.norm(r_TE.to("AU").value) * r_TE.unit) ** 3.0
            * Mfactor
            / 328900.56
        )
        F_T = F_sT + F_ET
        # differential forces
        dF = F_O - F_T * self.scMass / self.coMass
        dF_axial = (dF.dot(u_targ)).to("N")
        dF_lateral = (dF - dF_axial * u_targ).to("N")
        dF_lateral = np.linalg.norm(dF_lateral.to("N").value) * dF_lateral.unit
        dF_axial = np.abs(dF_axial)

        return dF_lateral, dF_axial

    def mass_dec(self, dF_lateral, t_int):
        """Returns mass_used and deltaV

        The values returned by this method are used to decrement spacecraft
        mass for station-keeping.

        Args:
            dF_lateral (astropy.units.Quantity):
                Lateral disturbance force in units of N
            t_int (astropy.units.Quantity):
                Integration time in units of day

        Returns:
            tuple:
                intMdot (astropy.units.Quantity):
                    Mass flow rate in units of kg/s
                mass_used (astropy.units.Quantity):
                    Mass used in station-keeping units of kg
                deltaV (astropy.units.Quantity):
                    Change in velocity required for station-keeping in units of km/s

        """

        intMdot = (dF_lateral / self.skEff / const.g0 / self.skIsp).to("kg/s")
        mass_used = (intMdot * t_int).to("kg")
        deltaV = (dF_lateral / self.scMass * t_int).to("km/s")

        return intMdot, mass_used, deltaV

    def mass_dec_sk(self, TL, sInd, currentTime, t_int):
        """Returns mass_used, deltaV and disturbance forces

        This method calculates all values needed to decrement spacecraft mass
        for station-keeping.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInd (int):
                Integer index of the star of interest
            currentTime (astropy.time.Time):
                Current absolute mission time in MJD
            t_int (astropy.units.Quantity):
                Integration time in units of day

        Returns:
            tuple:
                dF_lateral (astropy.units.Quantity):
                    Lateral disturbance force in units of N
                dF_axial (astropy.units.Quantity):
                    Axial disturbance force in units of N
                intMdot (astropy.units.Quantity):
                    Mass flow rate in units of kg/s
                mass_used (astropy.units.Quantity):
                    Mass used in station-keeping units of kg
                deltaV (astropy.units.Quantity):
                    Change in velocity required for station-keeping in units of km/s

        """

        dF_lateral, dF_axial = self.distForces(TL, sInd, currentTime)
        intMdot, mass_used, deltaV = self.mass_dec(dF_lateral, t_int)

        return dF_lateral, dF_axial, intMdot, mass_used, deltaV

    def calculate_dV(self, TL, old_sInd, sInds, sd, slewTimes, tmpCurrentTimeAbs):
        """Finds the change in velocity needed to transfer to a new star line of sight

        This method sums the total delta-V needed to transfer from one star
        line of sight to another. It determines the change in velocity to move from
        one station-keeping orbit to a transfer orbit at the current time, then from
        the transfer orbit to the next station-keeping orbit at currentTime + dt.
        Station-keeping orbits are modeled as discrete boundary value problems.
        This method can handle multiple indeces for the next target stars and calculates
        the dVs of each trajectory from the same starting star.

        The prototype implementation does not perform any real calculations and returns
        all zero values.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            old_sInd (int):
                Index of the current star
            sInds (~numpy.ndarray(int)):
                Integer index of the next star(s) of interest
            sd (~astropy.units.Quantity(~numpy.ndarray(float))):
                Angular separation between stars in rad
            slewTimes (~astropy.time.Time(~numpy.ndarray)):
                Slew times.
            tmpCurrentTimeAbs (~astropy.time.Time):
                Current absolute mission time in MJD

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Delta-V values in units of length/time
        """

        dV = np.zeros(len(sInds))

        return dV * u.m / u.s

    def calculate_slewTimes(self, TL, old_sInd, sInds, sd, obsTimes, currentTime):
        """Finds slew times and separation angles between target stars

        This method determines the slew times of an occulter spacecraft needed
        to transfer from one star's line of sight to all others in a given
        target list.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            old_sInd (int):
                Integer index of the most recently observed star
            sInds (~numpy.ndarray(int)):
                Integer indices of the star of interest
            sd (~astropy.units.Quantity):
                Angular separation between stars in rad
            obsTimes (~astropy.time.Time(~numpy.ndarray)):
                Observation times for targets.
            currentTime (~astropy.time.Time(~numpy.ndarray)):
                Current absolute mission time in MJD

        Returns:
            ~astropy.units.Quantity:
                Time to transfer to new star line of sight in units of days
        """

        self.ao = self.thrust / self.scMass
        slewTime_fac = (
            (
                2.0
                * self.occulterSep
                / np.abs(self.ao)
                / (self.defburnPortion / 2.0 - self.defburnPortion**2.0 / 4.0)
            )
            .decompose()
            .to("d2")
        )

        if old_sInd is None:
            slewTimes = np.zeros(TL.nStars) * u.d
        else:
            # calculate slew time
            slewTimes = np.sqrt(
                slewTime_fac * np.sin(abs(sd) / 2.0)
            )  # an issue exists if sd is negative

            # The following are debugging
            assert (
                np.where(np.isnan(slewTimes))[0].shape[0] == 0
            ), "At least one slewTime is nan"

        return slewTimes

    def log_occulterResults(self, DRM, slewTimes, sInd, sd, dV):
        """Updates the given DRM to include occulter values and results

        Args:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
            slewTimes (astropy.units.Quantity):
                Time to transfer to new star line of sight in units of days
            sInd (int):
                Integer index of the star of interest
            sd (astropy.units.Quantity):
                Angular separation between stars in rad
            dV (astropy.units.Quantity):
                Delta-V used to transfer to new star line of sight in units of m/s

        Returns:
            dict:
                Design Reference Mission dictionary, contains the results of one
                complete observation (detection and characterization)

        """

        DRM["slew_time"] = slewTimes.to("day")
        DRM["slew_angle"] = sd.to("deg")

        slew_mass_used = slewTimes * self.defburnPortion * self.flowRate
        DRM["slew_dV"] = (slewTimes * self.ao * self.defburnPortion).to("m/s")
        DRM["slew_mass_used"] = slew_mass_used.to("kg")
        self.scMass = self.scMass - slew_mass_used
        DRM["scMass"] = self.scMass.to("kg")
        if self.twotanks:
            self.slewMass = self.slewMass - slew_mass_used
            DRM["slewMass"] = self.slewMass.to("kg")
        return DRM

    def refuel_tank(self, TK, tank=None):
        """Attempt to refuel a fuel tank and report status

        Args:
            TK (:ref:`TimeKeeping`):
                TimeKeeping object. Not used in prototype but an input for any
                implementations that wish to do time-aware operations.
            tank (str, optional):
                Either 'sk' or 'slew' when ``twotanks`` is True. Otherwise, None.
                Defaults None

        Returns:
            bool:
                True represents successful refeuling. False means refueling is not
                possible for selected tank.
        """

        if not (self.allowRefueling):
            return False

        if self.external_fuel_mass <= 0 * u.kg:
            return False

        if tank is not None:
            assert tank.lower() in ["sk", "slew"], "Tank must be 'sk' or 'slew'."
            assert self.twotanks, "You may only specify a tank when twotanks is True."

            if tank == "sk":
                tank_mass = self.skMass
                tank_capacity = self.skMaxFuelMass
                tank_name = "stationkeeping"
            else:
                tank_mass = self.slewMass
                tank_capacity = self.slewMaxFuelMass
                tank_name = "slew"
        else:
            tank_mass = self.scMass
            tank_capacity = self.maxFuelMass + self.dryMass
            tank_name = ""

        # Add as much fuel as can fit in the tank (plus any currently carried negative
        # value, or whatever remains in the external tank)
        topoff = (
            np.min(
                [
                    self.external_fuel_mass.to(u.kg).value,
                    (tank_capacity - tank_mass).to(u.kg).value,
                ]
            )
            * u.kg
        )
        assert topoff >= 0 * u.kg, "Topoff calculation produced negative result."

        self.external_fuel_mass -= topoff
        tank_mass += topoff
        if tank is not None:
            self.scMass += topoff
        self.vprint("{} {} fuel added".format(topoff, tank_name))
        self.vprint("{} remaining in external tank.".format(self.external_fuel_mass))

        return True

    class SolarEph:
        """Solar system ephemerides class

        This class takes the constants in Appendix D.4 of Vallado as inputs
        and stores them for use in defining solar system ephemerides at a
        given time.

        Args:
            a (list):
                semimajor axis list (in AU)
            e (list):
                eccentricity list
            I (list):
                inclination list
            O (list):
                right ascension of the ascending node list
            w (list):
                longitude of periapsis list
            lM (list):
                mean longitude list

        Each of these lists has a maximum of 4 elements. The values in
        these lists are used to propagate the solar system planetary
        ephemerides for a specific solar system planet.

        Attributes:
            a (list):
                list of semimajor axis (in AU)
            e (list):
                list of eccentricity
            I (list):
                list of inclination
            O (list):
                list of right ascension of the ascending node
            w (list):
                list of longitude of periapsis
            lM (list):
                list of mean longitude values

        Each of these lists has a maximum of 4 elements. The values in
        these lists are used to propagate the solar system planetary
        ephemerides for a specific solar system planet.

        """

        def __init__(self, a, e, I, O, w, lM):  # noqa: E741

            # store list of semimajor axis values (convert from AU to km)
            self.a = (a * u.AU).to("km").value
            if not isinstance(self.a, float):
                self.a = self.a.tolist()
            # store list of dimensionless eccentricity values
            self.e = e
            # store list of inclination values (degrees)
            self.I = I  # noqa: E741
            # store list of right ascension of ascending node values (degrees)
            self.O = O  # noqa: E741
            # store list of longitude of periapsis values (degrees)
            self.w = w
            # store list of mean longitude values (degrees)
            self.lM = lM

        def __str__(self):
            """String representation of the SolarEph object

            When the command 'print' is used on the SolarEph object, this
            method will print the attribute values contained in the object

            """

            for att in self.__dict__:
                print("%s: %r" % (att, getattr(self, att)))

            return "SolarEph class object attributes"
