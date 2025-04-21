# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
# ]
# ///
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.optimize import curve_fit
from synphot import SourceSpectrum, SpectralElement
from synphot.models import BlackBodyNorm1D, Empirical1D
from synphot.units import VEGAMAG

import EXOSIMS.MissionSim as MissionSim

specs = {
    "modules": {
        "PlanetPopulation": "",
        "StarCatalog": "",
        "OpticalSystem": "",
        "ZodiacalLight": "",
        "BackgroundSources": "",
        "PlanetPhysicalModel": "",
        "Observatory": "",
        "TimeKeeping": " ",
        "PostProcessing": " ",
        "Completeness": "",
        "TargetList": " ",
        "SimulatedUniverse": "",
        "SurveySimulation": " ",
        "SurveyEnsemble": " ",
    },
    "observingModes": [
        {
            "instName": "imager",
            "systName": "coronagraph",
            "detectionMode": True,
            "SNR": 5,
            "lam": 700,
        }
    ],
}
sim = MissionSim.MissionSim(**specs)


def get_solar_spectrum(TL, mode=None):
    """
    Retrieve the solar spectrum using a TargetList object

    Args:
        mode (dict or None): Optional observing mode dictionary to specify
                                a particular bandpass. Defaults to None.

    Returns:
        SourceSpectrum: The solar spectrum (G2V star).
    """
    # Define the spectral type for the Sun (G2V)
    solar_spectral_type = "G2V"

    # Check if the solar spectral type is in the catalog
    if solar_spectral_type in TL.spectral_catalog_index:
        # Use the existing template loader method
        solar_spectrum = TL.get_template_spectrum(solar_spectral_type)
    else:
        raise ValueError(f"Spectral type {solar_spectral_type} not found in catalog.")

    # Optionally normalize or modify spectrum if an observing mode is provided
    # if mode is not None:
    solar_spectrum = solar_spectrum.normalize(
        4.83 * VEGAMAG, TL.standard_bands["V"], vegaspec=TL.OpticalSystem.vega_spectrum
    )

    return solar_spectrum


star_spectrum = get_solar_spectrum(sim.TargetList)

# Leinert zodi wavelength in um
zodi_lam = (
    np.array(
        [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 2.2, 3.5, 4.8, 12, 25, 60, 100, 140]
    )
    * u.um
)
# Leinert zodi brightness in W/(m**2 sr um)
zodi_brightness = (
    np.array(
        [
            2.5e-8,
            5.3e-7,
            2.2e-6,
            2.6e-6,
            2.0e-6,
            1.3e-6,
            1.2e-6,
            8.1e-7,
            1.7e-7,
            5.2e-8,
            1.2e-7,
            7.5e-7,
            3.2e-7,
            1.8e-8,
            3.2e-9,
            6.9e-10,
        ]
    )
    * u.W
    / (u.m**2 * u.sr * u.um)
)

zodi_photon_flux = zodi_brightness.to(
    u.ph / (u.s * u.m**2 * u.arcsec**2 * u.nm),
    equivalencies=u.spectral_density(zodi_lam),
)

# Define the wavelength range for the model
wavelengths = (
    np.logspace(
        np.log10(zodi_lam[0].to(u.um).value),
        np.log10(zodi_lam[-1].to(u.um).value),
        1000,
    )
    * u.micron
)


def model_flux(wavelengths_um, fstar, fdust):
    # Create star and dust spectra
    star_spectrum = get_solar_spectrum(sim.TargetList)
    dust_spectrum = SourceSpectrum(BlackBodyNorm1D, temperature=261.5 * u.K)

    # Define the transmission function: 1 up to 10 microns, 0 afterwards
    transmission_wavelengths = [0.1, 10, 10.001, 200] * u.um
    transmission_values = [1, 1, 0, 0]

    # Create a SpectralElement from this filter
    custom_filter = SpectralElement(
        Empirical1D, points=transmission_wavelengths, lookup_table=transmission_values
    )
    star_spectrum *= custom_filter

    # Create composite spectrum
    combined_spectrum = star_spectrum * fstar + dust_spectrum * fdust

    # Get wavelength-dependent flux
    combined_flux = combined_spectrum(wavelengths_um).to(
        u.ph / (u.nm * u.s * u.m**2),
        equivalencies=u.spectral_density(wavelengths_um),
    )
    return combined_flux.value


# Initial guesses for fstar and fdust
initial_guess = [1e-3, 1e4]

# Run curve fit
popt, pcov = curve_fit(
    lambda lam_um, scat_fac, dust_fac: model_flux(
        lam_um * u.um,
        scat_fac,
        dust_fac,
    ),
    zodi_lam.value,
    zodi_photon_flux.value,
    p0=initial_guess,
)

# Get optimized parameters
fstar_fit, fdust_fit = popt
print(f"Fitted starlight scattering factor: {fstar_fit}")
print(f"Fitted dust thermal emission factor: {fdust_fit}")

# Get fitted flux
fitted_flux = model_flux(wavelengths, fstar_fit, fdust_fit)

# Create final star and dust spectra for plotting
final_star_spectrum = get_solar_spectrum(sim.TargetList) * fstar_fit
transmission_wavelengths = [0.1, 10, 10.001, 200] * u.um
transmission_values = [1, 1, 0, 0]
custom_filter = SpectralElement(
    Empirical1D, points=transmission_wavelengths, lookup_table=transmission_values
)
final_star_spectrum *= custom_filter
final_star_flux = final_star_spectrum(wavelengths).to(
    u.ph / (u.nm * u.s * u.m**2), equivalencies=u.spectral_density(wavelengths)
)
final_dust_spectrum = (
    SourceSpectrum(BlackBodyNorm1D, temperature=261.5 * u.K) * fdust_fit
)
final_dust_flux = final_dust_spectrum(wavelengths).to(
    u.ph / (u.nm * u.s * u.m**2), equivalencies=u.spectral_density(wavelengths)
)

# Plot
plt.figure(figsize=(8, 6))
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.25, 1, 3))

plt.scatter(zodi_lam, zodi_photon_flux.value, color="red", label="Leinert Data")
plt.plot(wavelengths.to(u.um), fitted_flux, label="Fitted Model", color=colors[0])
plt.plot(
    wavelengths.to(u.um),
    final_star_flux,
    label="Reflected light",
    linestyle=":",
    color=colors[1],
)
plt.plot(
    wavelengths.to(u.um),
    final_dust_flux,
    label="Thermal emission",
    linestyle="--",
    color=colors[2],
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Wavelength (μm)")
plt.ylabel("Specific Intensity (ph s⁻¹ m⁻² arcsec⁻² nm⁻¹)")
plt.xlim(0.1, 200)
plt.ylim(2e-4, 2)
plt.legend()
plt.grid(True)
plt.title("Fit to Leinert Data")
plt.text(
    0.65,
    0.24,
    f"$f_\\text{{star}}$={fstar_fit:.2e}\n"
    f"$f_\\text{{thermal}}$={fdust_fit:.2e}\n"
    f"$T_\\text{{dust}}$={261.5:.2f} K",
    fontsize=12,
    transform=plt.gca().transAxes,
)
plt.savefig("fit_leinert_fixed_temp.png")
plt.show()
