from EXOSIMS.PlanetPhysicalModel.FortneyMarleyCahoyMix1 import FortneyMarleyCahoyMix1
from EXOSIMS.util.get_dirs import get_downloads_dir
import astropy.units as u
import numpy as np
import os
import h5py
from scipy.stats import norm
from urllib.request import urlretrieve


class Forecaster(FortneyMarleyCahoyMix1):
    """Planet M-R relation model based on the FORECASTER software, Chen & Kippling 2016.

    This module requires to download the fitting_parameters.h5 file from the
    FORECASTER GitHub repository at https://github.com/chenjj2/forecaster
    and add it to the PlanetPhysicalModel directory.

    Args:
        **specs:
            user specified values

    """

    def __init__(self, n_pop=4, **specs):

        FortneyMarleyCahoyMix1.__init__(self, **specs)

        # number of category
        self.n_pop = int(n_pop)

        # read forecaster parameter file
        downloadsdir = get_downloads_dir()
        filename = "fitting_parameters.h5"
        parampath = os.path.join(downloadsdir, filename)
        if not os.path.exists(parampath) and os.access(downloadsdir, os.W_OK | os.X_OK):
            fitting_url = (
                "https://raw.github.com/dsavransky/forecaster/master/"
                "fitting_parameters.h5"
            )
            self.vprint(
                "Fetching Forecaster fitting parameters from %s to %s"
                % (fitting_url, parampath)
            )
            try:
                urlretrieve(fitting_url, parampath)
            except:  # noqa: E722
                self.vprint(
                    "Error: Remote fetch failed. You can try to fetch manually."
                )

        assert os.path.exists(
            parampath
        ), "fitting_parameters.h5 must exist in /.EXOSIMS/downloads"

        h5 = h5py.File(parampath, "r")
        self.all_hyper = h5["hyper_posterior"][:]
        h5.close()

    def calc_radius_from_mass(self, Mp):
        """Forecast the Radius distribution given the mass distribution.

        Args:
            Mp (astropy Quantity array):
                Planet mass in units of Earth mass

        Returns:
            Rp (astropy Quantity array):
                Planet radius in units of Earth radius

        """

        mass = Mp.to("earthMass").value
        assert (
            np.min(mass) > 3e-4 and np.max(mass) < 3e5
        ), "Mass range out of model expectation. Returning None."

        sample_size = len(mass)
        logm = np.log10(mass)
        prob = np.random.random(sample_size)
        logr = np.ones_like(logm)
        hyper_ind = np.random.randint(
            low=0, high=np.shape(self.all_hyper)[0], size=sample_size
        )
        hyper = self.all_hyper[hyper_ind, :]

        for i in range(sample_size):
            logr[i] = self.piece_linear(hyper[i], logm[i], prob[i])

        Rp = 10.0**logr * u.earthRad

        return Rp

    ####################################################################################
    # The following functions where adapted from the func.py file from the FORECASTER
    # GitHub repository at https://github.com/chenjj2/forecaster (Chen & Kippling 2016)
    ####################################################################################

    def indicate(self, M, trans, i):
        """
        indicate which M belongs to population i given transition parameter
        """
        ts = np.insert(np.insert(trans, self.n_pop - 1, np.inf), 0, -np.inf)
        ind = (M >= ts[i]) & (M < ts[i + 1])
        return ind

    def split_hyper_linear(self, hyper):
        """
        split hyper and derive c
        """
        c0, slope, sigma, trans = (
            hyper[0],
            hyper[1 : 1 + self.n_pop],
            hyper[1 + self.n_pop : 1 + 2 * self.n_pop],
            hyper[1 + 2 * self.n_pop :],
        )
        c = np.zeros_like(slope)
        c[0] = c0
        for i in range(1, self.n_pop):
            c[i] = c[i - 1] + trans[i - 1] * (slope[i - 1] - slope[i])
        return c, slope, sigma, trans

    def piece_linear(self, hyper, M, prob_R):
        """
        model: straight line
        """
        c, slope, sigma, trans = self.split_hyper_linear(hyper)
        R = np.zeros_like(M)
        for i in range(4):
            ind = self.indicate(M, trans, i)
            mu = c[i] + M[ind] * slope[i]
            R[ind] = norm.ppf(prob_R[ind], mu, sigma[i])
        return R
