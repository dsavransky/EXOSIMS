from EXOSIMS.Prototypes.StarCatalog import StarCatalog
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


class FakeCatalog_UniformAngles(StarCatalog):
    """Fake Catalog of stars separated uniformly by angle

    Generate a fake catalog of stars that are uniformly separated.

    Args:
        ntargs (int):
            Sqrt of number of target stars to generate. ntargs by ntargs grid in
            ra and dec.
        star_dist (float):
            Star distance from observer
        specs:
            Additional parameters passed to StarCatalog parent class

    """

    def __init__(self, ntargs=20, star_dist=5, **specs):

        StarCatalog.__init__(self, **specs)

        # ntargs must be an integer >= 1
        self.ntargs = max(int(ntargs**2), 1)

        # putting it all together
        raRng = np.linspace(0, 360, ntargs)
        decRng = np.linspace(-90, 90, ntargs)
        dists = star_dist * np.ones(ntargs**2) * u.pc

        ra, dec = np.meshgrid(raRng, decRng) * u.deg

        # reference star should be first on the list
        coords = SkyCoord(ra.flatten(), dec.flatten(), dists, frame="icrs")

        # list of astropy attributes
        self.coords = coords  # ICRS coordinates
        self.ntargs = int(len(self.coords.ra))
        self.dist = star_dist * np.ones(self.ntargs) * u.pc  # distance
        self.parx = self.dist.to("mas", equivalencies=u.parallax())  # parallax
        self.pmra = np.zeros(self.ntargs) * u.mas / u.yr  # proper motion in RA
        self.pmdec = np.zeros(self.ntargs) * u.mas / u.yr  # proper motion in DEC
        self.rv = np.zeros(self.ntargs) * u.km / u.s  # radial velocity

        # list of non-astropy attributes to pass target list filters
        self.Name = np.array([str(x) for x in range(self.ntargs)])  # star names
        self.Spec = np.array(["G"] * self.ntargs)  # spectral types
        self.Umag = np.zeros(self.ntargs)  # U magnitude
        self.Bmag = np.zeros(self.ntargs)  # B magnitude
        self.Vmag = 5 * np.ones(self.ntargs)  # V magnitude
        self.Rmag = np.zeros(self.ntargs)  # R magnitude
        self.Imag = np.zeros(self.ntargs)  # I magnitude
        self.Jmag = np.zeros(self.ntargs)  # J magnitude
        self.Hmag = np.zeros(self.ntargs)  # H magnitude
        self.Kmag = np.zeros(self.ntargs)  # K magnitude
        self.BV = np.zeros(self.ntargs)  # B-V Johnson magnitude
        self.MV = self.Vmag - 5 * (np.log10(star_dist) - 1)  # absolute V magnitude
        self.BC = -0.10 * np.ones(self.ntargs)  # bolometric correction

        BM = self.MV + self.BC
        BMsun = 4.74
        self.L = 10 ** (0.4 * (BMsun - BM))  # stellar luminosity in solar units
        self.Binary_Cut = np.zeros(
            self.ntargs, dtype=bool
        )  # binary closer than 10 arcsec

        # populate outspecs
        self._outspec["ntargs"] = self.ntargs
