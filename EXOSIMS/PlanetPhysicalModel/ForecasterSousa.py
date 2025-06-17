from EXOSIMS.PlanetPhysicalModel.FortneyMarleyCahoyMix1 import FortneyMarleyCahoyMix1
import astropy.units as u
import numpy as np


class ForecasterSousa(FortneyMarleyCahoyMix1):
    """Planet M-R relation model based on the best-fit model to a planet sample
    with homogenous stellar parameters from SWEET-cat as described in
    Sousa et al. (2024)
    """

    def __init__(self, **specs):

        FortneyMarleyCahoyMix1.__init__(self, **specs)

        M = np.array([0.59, 0])
        B = np.array([-0.15, 1.15])
        T = np.array(
            [
                159,
            ]
        )

        Rj = u.R_jupiter.to(u.R_earth)

        self.M = M
        self.B = B
        self.T = T
        self.Rj = Rj

        self.Mj = (u.M_jupiter).to(u.M_earth)
        Tinv = self.calc_radius_from_mass(self.T * u.M_earth).value

        self.Tinv = Tinv

    def calc_radius_from_mass(self, Mp):
        """Calculate planet radius from mass

        Args:
            Mp (~astropy.units.Quantity(~numpy.ndarray(float))):
                Planet mass in units of Earth mass

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Planet radius in units of Earth radius


        """

        m = np.array(Mp.to(u.M_earth).value, ndmin=1)
        R = np.zeros(m.shape)
        # import pdb; pdb.set_trace()
        inds = np.digitize(m, np.hstack((0, self.T, np.inf)))
        for i in range(1, inds.max() + 1):
            R[inds == i] = 10.0 ** (
                np.log10(m[inds == i]) * self.M[i - 1] + self.B[i - 1]
            )
            # R[inds == i] = 10.0 ** (np.log10(m[inds == i]) * self.M[1] + self.B[1])

        return R * u.R_earth

    def calc_mass_from_radius(self, Rp):
        """Calculate planet mass from radius

        Args:
            Rp (~astropy.units.Quantity(~numpy.ndarray(float))):
                Planet radius in units of Earth radius

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Planet mass in units of Earth mass

        Note:
            The fit is non-invertible for Jupiter radii, so all those get 1
            Jupiter mass.


        """

        R = np.array(Rp.to(u.R_earth).value, ndmin=1)
        m = np.zeros(R.shape)
        m[np.isnan(R)] = np.nan

        inds = np.digitize(R, np.hstack((0, self.Tinv, np.inf)))
        for i in range(1, len(self.B) + 1):
            if i == 2:
                m[inds == i] = self.Mj
            else:
                m[inds == i] = 10.0 ** (
                    (np.log10(R[inds == i]) - self.B[i - 1]) / self.M[i - 1]
                )
        return m * u.M_earth
