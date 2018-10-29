.. _userparams:

Input Parameters
########################

These are the standard input parameters typically included in a simulation JSON script, split by
the module that typically processes them.  Note, however, that *ALL* parameters are passed through
all modules, and can be used multiple times in instantiation.

MissionSim
------------

-  **verbose** (boolean) Boolean used to create the vprint function,
   equivalent to the python print function with an extra verbose toggle
   parameter (True by default). The vprint function can be accessed by
   all modules from EXOSIMS.util.vprint.

-  **seed** (integer) Number used to seed the NumPy generator. Generated
   randomly by default.

-  **logfile** (string) Path to the log file. If None, logging is turned
   off. If supplied but empty string (''), a temporary file is generated.

-  **loglevel** (string) The level of log, defaults to ’INFO’. Valid
   levels are: CRITICAL, ERROR, WARNING, INFO, DEBUG (case sensitive).


PlanetPopulation
--------------------

-  **arange** (float) 1\ :math:`\times`\ 2 list of semi-major axis range
   in units of :math:`AU`.

-  **erange** (float) 1\ :math:`\times`\ 2 list of eccentricity range.

-  **Irange** (float) 1\ :math:`\times`\ 2 list of inclination range in
   units of :math:`deg`.

-  **Orange** (float) 1\ :math:`\times`\ 2 list of ascension of the
   ascending node range in units of :math:`deg`.

-  **wrange** (float) 1\ :math:`\times`\ 2 list of argument of perigee
   range in units of :math:`deg`.

-  **prange** (float) 1\ :math:`\times`\ 2 list of planetary geometric
   albedo range.

-  **Rprange** (float) 1\ :math:`\times`\ 2 list of planetary radius
   range in Earth radii.

-  **Mprange** (float) 1\ :math:`\times`\ 2 list of planetary mass range
   in Earth masses.

-  **scaleOrbits** (boolean) True means planetary orbits are scaled by
   the square root of stellar luminosity.

-  **constrainOrbits** (boolean) True means planetary orbits are
   constrained to never leave the semi-major axis range (arange).

-  **eta** (float) The average occurrence rate of planets per star for
   the entire population.

OpticalSystem
---------------

-  **obscurFac** (float) Obscuration factor due to secondary mirror and
   spiders.

-  **shapeFac** (float) Telescope aperture shape factor.

-  **pupilDiam** (float) Entrance pupil diameter in units of :math:`m`.

-  **IWA** (float) Fundamental Inner Working Angle in units of
   :math:`arcsec`. No planets can ever be observed at smaller
   separations.

-  **OWA** (float) Fundamental Outer Working Angle in units of
   :math:`arcsec`. Set to :math:`Inf` for no OWA. JSON values of 0
   will be interpreted as :math:`Inf`.

-  **intCutoff** (float) Maximum allowed integration time in units of
   :math:`day`.

-  **dMag0** (float) Favorable planet delta magnitude value used to
   calculate the minimum integration times for inclusion in target list.

-  **WA0** (float) Instrument working angle value used to calculate the
   minimum integration times for inclusion in target list, in units of
   :math:`arcsec`.

-  **scienceInstruments** (list of dicts) Contains specific attributes
   of all science instruments.

-  **starlight-**

-  **SuppressionSystems** (list of dicts) Contains specific attributes
   of all starlight suppression systems.

-  **observingModes** (list of dicts) Contains specific attributes of
   all observing modes.

ZodiacalLight
---------------

-  **magZ** (float) 1 zodi brightness magnitude (per arcsec2).

-  **magEZ** (float) 1 exo-zodi brightness magnitude (per arcsec2).

-  **varEZ** (float) exo-zodiacal light variation (variance of
   log-normal distribution).

PostProcessing
-----------------

-  **FAP** (float) False Alarm Probability.

-  **MDP** (float) Missed Detection Probability.

-  **ppFact** (float, callable) Post-processing contrast factor, between
   0 and 1.

-  **FAdMag0** (float, callable) Minimum delta magnitude that can be
   obtained by a false alarm.

Completeness
---------------

-  **dMagLim** (float) Limiting planet-to-star delta magnitude for
   completeness.

-  **minComp** (float) Minimum completeness value for inclusion in
   target list.

TargetList
-------------

-  **staticStars** (boolean) Boolean used to force static target
   positions set at mission start time.

-  **keepStarCatalog** (boolean) Boolean representing whether to delete
   the star catalog after assembling the target list. If true, object
   reference will be available from TargetList object.

-  **fillPhotometry** (boolean) If True, attempt to fill in missing 
   photometric values based on target spectral type.

Observatory
--------------

-  **koAngleMin** (float) Telescope minimum keepout angle in units of
   :math:`deg`.

-  **koAngleMinMoon** (float) Telescope minimum keepout angle in units
   of :math:`deg`, for the Moon only.

-  **koAngleMinEarth** (float) Telescope minimum keepout angle in units
   of :math:`deg`, for the Earth only.

-  **koAngleMax** (float) Telescope maximum keepout angle (for occulter)
   in units of :math:`deg`.

-  **koAngleSmall** (float) Telescope keepout angle for smaller (angular
   size) bodies in units of :math:`deg`.

-  **checkKeepoutEnd** (boolean) Boolean signifying if the keepout
   method must be called at the end of each observation.

-  **settlingTime** (float) Amount of time needed for observatory to
   settle after a repointing in units of :math:`day`.

-  **thrust** (float) Occulter slew thrust in units of :math:`mN`.

-  **slewIsp** (float) Occulter slew specific impulse in units of
   :math:`s`.

-  **scMass** (float) Occulter (maneuvering spacecraft) initial wet mass
   in units of :math:`kg`.

-  **dryMass** (float) Occulter (maneuvering spacecraft) dry mass in
   units of :math:`kg`.

-  **coMass** (float) Telescope (or non-maneuvering spacecraft) mass in
   units of :math:`kg`.

-  **occulterSep** (float) Occulter-telescope distance in units of
   :math:`km`.

-  **skIsp** (float) Specific impulse for station keeping in units of
   :math:`s`.

-  **defburnPortion** (float) Default burn portion for slewing.

-  **checkKeepoutEnd** (boolean) Boolean signifying if the keepout
   method must be called at the end of each observation.

-  **forceStaticEphem** (boolean) Force use of static solar system
   ephemeris if set to True, even if jplephem module is present.

-  **spkpath** (string) Full path to SPK kernel file.

TimeKeeping
--------------

-  **missionLife** (float) The total mission lifetime in units of
   :math:`year`. When the mission time is equal or greater to this
   value, the mission simulation stops.

-  **missionPortion** (float) The portion of the mission dedicated to
   exoplanet science, given as a value between 0 and 1. The mission
   simulation stops when the total integration time plus observation
   overhead time is equal to the missionLife :math:`\times`
   missionPortion.

-  **extendedLife** (float) Extended mission time in units of
   :math:`year`. Extended life typically differs from the primary
   mission in some way—most typically only revisits are allowed

-  **missionStart** (float) Mission start time in :math:`MJD`.

-  **OBduration** (float) Default allocated duration of observing
   blocks, in units of :math:`day`. If no OBduration was specified, a
   new observing block is created for each new observation in the
   SurveySimulation module.

-  **waitTime** (float) Default allocated duration to wait in units of
   :math:`day`, when the Survey Simulation does not find any observable
   target.

-  **waitMultiple** (float) Multiplier applied to the wait time in case
   of repeated empty lists of observable targets, which makes the wait
   time grow exponentially.

SurveySimulation
-----------------

-  **nt\_flux** (integer) Observation time sampling, to determine the
   integration time interval.

-  **nVisitsMax** (integer) Maximum number of observations (in detection
   mode) per star.

-  **charMargin** (float) Integration time margin for characterization.

-  **seed** (integer) Random seed used to make all random number
   generation reproducible.

-  **WAint** (float) Working angle used for integration time calculation
   in units of :math:`arcsec`.

-  **dMagint** (float) Delta magnitude used for integration time
   calculation.

-  **cachedir** (string) Path to desired cache directory (default is ``$HOME/.EXOSIMS/cache``).


