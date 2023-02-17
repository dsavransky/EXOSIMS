.. _opticalsystem:

OpticalSystem
================

Optical system modules describe the science instrument and starlight suppression system, and provide methods for integration time calculation.

Optical System Definition
----------------------------

An optical system is defined by three objects:

* Science Instrument(s)
* Starlight Suppression System(s)
* Observing Mode(s)

Each of these is encoded as a list of dictionaries with a minimum size of 1.  A science instrument is a description of a detector and any associated optics not belonging to the starlight suppression system.  A science instrument must be classified as an imager (no spectral resolution) or a spectrometer (finite spectral resolution). A starlight suppression system is a description of all of the optics responsible for producing regions of high contrast.  It must be classified as a coronagraph (internal) or occulter (starshade; external). Finally, an observing mode is the combination of a starlight suppression system and a science instrument, along with a target :term:`SNR` for all integrations performed with that observing mode.  

Many quantities defining the optical system must be parametrizable by wavelength or angular separation (or both).  In cases where only a single value exists at your current design stage, you must still structure these aspects of the system as callable (just returning the same value regardless of input).  As an example, quantum efficiency is a function of wavelength for nearly all physical devices.  If you have not yet selected a specific device (or do not happen to have the QE curve for the device you are modeling) your QE parameter should be a callable method or lambda function that always returns the same constant QE you wish to use in your current model.

In general, each dictionary describing each of these objects can have essentially any keywords. This description allows for optical system definitions to be highly flexible and extensible, but can also lead to inescapable complexity.  To attempt to make the code more parsable, a few conventions are maintained, as outlined below.


Science Instruments
""""""""""""""""""""

Each ``scienceInstrument`` dictionary must contain a unique ``name`` keyword.  This string must include a substring of the form ``imager`` or ``spectro``. For example, an optical system might contain science instruments called ``imager-EMCCD`` or ``spectro-CCD``, describing a photon counting electron multiplying CCD imager and a mid-resolution imaging spectrometer.  In cases where the same physical detector hardware is expected to be used in different modes (i.e., a single chip serving as an imager and polarizer, and integral field unit by the introduction of additional removable optics), you must still set up separate science instruments for each operating mode.

Common science instrument attributes include:

* name (string):
    Instrument name (e.g. imager-EMCCD, spectro-CCD). Must contain the type of
    instrument (imager or spectro). Every instrument should have a unique name.
* QE (callable):
    Detector quantum efficiency, parametrized by wavelength.
* optics (float):
    Attenuation due to optics specific to the science instrument
* FoV (Quantity):
    Field of view in angle units
* pixelNumber (int):
    Detector array format, number of pixels per detector lines/columns
* pixelScale (Quantity):
    Detector pixel scale in units of angle per pixel
* pixelSize (Quantity):
    Pixel pitch in units of length
* focal (Quantity):
    Focal length in units of length
* fnumber (float):
    Detector f-number
* sread (float):
    Detector effective read noise per frame per pixel
* idark (Quantity):
    Detector dark-current per pixel in units of 1/Time
* CIC (float):
    Clock-induced-charge per frame per pixel
* texp (Quantity):
    Exposure time per frame in units of Time
* radDos (float):
    Radiation dosage. Use of this quantity is highly specific to your particular optical system model.
* PCeff (float):
    Photon counting efficiency
* ENF (float):
    (Specific to EM-CCDs) Excess noise factor
* Rs (float):
    (Specific to spectrometers) Spectral resolving power
* lenslSamp (float):
    (Specific to spectrometers) Lenslet sampling, number of pixel per
    lenslet rows or cols


Starlight Suppression System
""""""""""""""""""""""""""""""

Each ``starlighSuppressionSystem`` dictionary must contain a unique name identifying the starlight suppression system (coronagraph or occulter).  As with the science instruments, if you are modeling a reconfigurable coronagraph (i.e., multiple filter wheels with multiple masks) you must define a separate system for each unique configuration you wish to model. Occulters operating at multiple distances must also be set up this way.

Common starlight suppression system attributes include:

* name (string):
    System name (e.g. HLC-565, SPC-660), should also contain the
    central wavelength the system is optimized for. Every system must have
    a unique name.
* optics (float):
    Attenuation due to optics specific to the coronagraph,
    e.g. polarizer, Lyot stop, extra fold mirrors, etc.
* lam (Quantity):
    Central wavelength in units of length
* deltaLam (Quantity):
    Bandwidth in units of length
* BW (float):
    Bandwidth fraction
* IWA (Quantity):
    Inner working angle in units of arcsec
* OWA (Quantity):
    Outer working angle in units of arcsec
* occ_trans (callable):
    Intensity transmission of extended background sources such as zodiacal light, parametrized by angular separation.
    Includes the pupil mask, occulter, Lyot stop and polarizer.
* core_thruput (callable):
    System throughput in the FWHM region of the planet PSF core, parametrized by angular separation.
* core_contrast (callable):
    System contrast = mean_intensity / PSF_peak, parametrized by angular separation.
* contrast_floor (float):
    An absolute limit on achievable core_contrast.
* core_mean_intensity (callable):
    Mean starlight residual normalized intensity per pixel, required to calculate
    the total core intensity as core_mean_intensity * Npix. If not specified,
    then the total core intensity is equal to core_contrast * core_thruput. Parametrized by angular separation.
* core_area (callable):
    Area of the FWHM region of the planet PSF, in units of arcsec^2, parametrized by angular separation.
* core_platescale (float):
    Platescale used for a specific set of coronagraph parameters, in units
    of lambda/D per pixel
* PSF (callable):
    Point spread function - 2D ndarray of values, normalized to 1 at
    the core. Note: normalization means that all throughput effects
    must be contained in the throughput attribute. Parametrized by angular separation.
* ohTime (Quantity):
    Overhead time for all integrations. 
* occulter (boolean):
    True if the system has an occulter (external or hybrid system) otherwise False (internal system)
* occulterDiameter (Quantity):
    Occulter diameter in units of m. Measured petal tip-to-tip.
* occulterDistance (Quantity):
    Telescope-occulter separation in units of km.

Observing Mode
"""""""""""""""""

An observing mode is the combination of a science instrument with a starlight suppression system along with rules for determining integration times. The observing mode can also specify additional parameters overwriting the values in the two sub-systems. One observing mode in the optical system must be tagged as the default detection mode (by setting boolean keyword ``detectionMode`` to True).  This is the mode used for all blind searches or initial target observations.

Common observing mode attributes include:

* instName (string):
    Instrument name. Must match with the name of a defined Science Instrument.
* systName (string):
    System name. Must match with the name of a defined Starlight Suppression System.
* inst (dict):
    Selected instrument of the observing mode.
* syst (dict):
    Selected system of the observing mode.
* detectionMode (boolean):
    True if this observing mode is the detection mode, otherwise False. Only one detection mode can be specified.
* SNR (float):
    Signal-to-noise ratio threshold
* timeMultiplier (float):
    Integration time multiplier applied for this mode.  For example, if this mode requires two full rolls for every observation, the timeMultiplier should be set to 2.
* lam (Quantity):
    Central wavelength in units of length
* deltaLam (Quantity):
    Bandwidth in units of length
* BW (float):
    Bandwidth fraction


Optical System Methods
-------------------------

Various different optical system models will have a variety of methods, but all optical systems are expected to provide the following:

Cp_Cb_Csp
"""""""""""

This method computes the count rates (electrons per unit time) for the planet (:math:`C_p`), the background (:math:`C_b`), and the residual speckle (:math:`C_{sp}`).  The last of these typically determines the systematic noise floor of the system.  In a simple optical system model, the foreground and background rates are likely entirely independent of one another (i.e.,  :math:`C_b` and :math:`C_{sp}` have no dependence on :math:`C_p`), but this is not actually a requirement.  More complicated descriptions, including those of electron-multiplying CCDs run in photon counting mode, will have clock-induced-charge coupling the foreground and background counts. See:  :py:meth:`~EXOSIMS.Prototypes.OpticalSystem.OpticalSystem.Cp_Cb_Csp`.

calc_intTime
"""""""""""""""""

Calculate the integration time required to reach the selected observing mode's target SNR on one or more targets for a planet of given :math:`\Delta\mathrm{mag}` at a given angular separation. If the SNR is unreachable by the selected observing mode, return NaN. See::py:meth:`~EXOSIMS.Prototypes.OpticalSystem.OpticalSystem.calc_intTime`.

calc_dMag_per_intTime
"""""""""""""""""""""""

Calculate the maximum :math:`\Delta\mathrm{mag}` planet observable at the observing mode's target SNR with the given integration time, at the given angular separation.  This should be a strict inverse of ``calc_intTime``.  See: :py:meth:`~EXOSIMS.Prototypes.OpticalSystem.OpticalSystem.calc_dMag_per_intTime`.


ddMag_dt
"""""""""""""

Calculate:

    .. math::
        
        \frac{\mathrm{d}}{\mathrm{d}t} \Delta\mathrm{mag}

This is used for integration time allocation optimization. See: :py:meth:`~EXOSIMS.Prototypes.OpticalSystem.OpticalSystem.ddMag_dt`.
