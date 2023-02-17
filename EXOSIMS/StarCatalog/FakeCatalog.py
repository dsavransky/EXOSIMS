from EXOSIMS.Prototypes.StarCatalog import StarCatalog
import numpy as np
import astropy.units as u
import random as py_random
from astropy.coordinates import SkyCoord


class FakeCatalog(StarCatalog):
    """Fake Catalog class
    This class generates an artificial target list of stars with a logistic
    distribution.

    Args:
        ntargs (int):
            Number of targets
        star_dist (float):
            Distance of the stars from observer
        ra0 (float):
            Reference right ascension
        dec0 (float):
            Reference declination
    """

    def __init__(self, ntargs=1000, star_dist=5, ra0=0, dec0=0, **specs):

        StarCatalog.__init__(self, **specs)

        self.seed = int(specs.get("seed", py_random.randint(1, int(1e9))))
        np.random.seed(self.seed)

        # ntargs must be an integer >= 1
        self.ntargs = max(int(ntargs), 1)
        self.ra0 = ra0 * u.rad
        self.dec0 = dec0 * u.rad

        # list of astropy attributes
        self.coords = self.inverse_method(self.ntargs, star_dist)  # ICRS coordinates
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

    def inverse_method(self, N, d):
        """Obtain coordinates for the targets from the inverse of a logistic function

        Args:
            N (int):
                Number of targets
            d (float):
                Star distance

        Returns:
            SkyCoord module:
                The coordinates for the targets

        """

        # getting sizes of the two angular sep distributions
        nP = int(np.floor(N / 2.0))  # half of stars in positive branch
        nN = nP + 1 if N % 2 else nP  # checks to see if N is odd

        # creating output of logistic function (positive and negative branch)
        tP = np.linspace(0.5, 0.99, nP)
        tN = np.linspace(0.5, 0.01, nN)[1:]  # not using the same reference star twice

        # getting inverse of logistic function as distribution of separations
        fP = np.log(tP / (1 - tP))
        fP = fP / np.abs(fP[-1])

        fN = np.log(tN / (1 - tN))
        fN = fN / np.abs(fN[-1])

        # getting angular distributions of stars for two branches
        raP, decP, distsP = self.get_angularDistributions(fP, d, pos=True)
        raN, decN, distsN = self.get_angularDistributions(fN, d, pos=False)

        # putting it all together
        ra = np.hstack([raP, raN]) * u.rad
        dec = np.hstack([decP, decN]) * u.rad
        dists = np.hstack([distsP, distsN]) * u.pc

        ra += self.ra0
        dec += self.dec0

        # reference star should be first on the list
        coords = SkyCoord(ra, dec, dists)

        return coords

    def get_angularDistributions(self, f, d, pos=True):
        """Get the distribution of target positions

        Args:
            f (array):
                Distribution function evaluated
            d (float):
                Star distance
            pos (boolean):
                North or south

        Returns:
            tuple:
                array:
                    Right ascension values
                array:
                    Declination values
                array:
                    Distances of the star
        """

        n = int(len(f))

        #        flips = np.arange(1,n,2) if f[0] == 0 else np.arange(0,n,2)
        flips = np.arange(0, n, 2)

        # angular separations from reference star
        psi = np.pi * f
        cosPsi = np.cos(psi)

        # calculating phi angle (i.e. DEC)
        sinPhi = np.abs(cosPsi) + (1 - np.abs(cosPsi)) * np.random.rand(n)
        phi = np.arcsin(sinPhi)  # only returns angles from 0 to pi/2

        # calculating phi angle (i.e. RA)
        cosTheta = cosPsi / sinPhi
        theta = np.arccos(cosTheta)

        # moving stars to southern hemisphere
        phi[flips] = np.pi - phi[flips]
        if pos:
            theta = 2 * np.pi - theta

        # final transforms
        dec = np.pi / 2.0 - phi
        dists = d * np.ones(n)

        return theta, dec, dists
