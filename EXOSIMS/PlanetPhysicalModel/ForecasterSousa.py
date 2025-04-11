from EXOSIMS.PlanetPhysicalModel.FortneyMarleyCahoyMix1 import FortneyMarleyCahoyMix1
import astropy.units as u
import numpy as np


class ForecasterSousa(FortneyMarleyCahoyMix1):
    """Planet M-R relation model based on modification of the FORECASTER
    best-fit model (Chen & Kippling 2016) as described in Savransky et al. (2019)

    This modification forces all planets below hydrogen burning to have a maximum
    radius of 1 R_jupiter and also adds the Saturn density as an explicit point
    in the model

    """

    def __init__(self, **specs):

        FortneyMarleyCahoyMix1.__init__(self, **specs)

        S = np.array([0.59,0])
        C = np.array([-0.15,1.15])
        T = np.array(
            [
                1.94,
                305.1168,
            ]
        )

        Rj = u.R_jupiter.to(u.R_earth)

        self.S = S
        self.C = C
        self.T = T
        self.Rj = Rj

        self.Mj = (u.M_jupiter).to(u.M_earth)
        Tinv = self.calc_radius_from_mass(self.T * u.M_earth).value
        #Rerr = np.abs(self.calc_radius_from_mass(1 * u.M_jupiter)[0].value - self.Rj)
        #Tinv[2] = self.Rj - 2 * Rerr
        #Tinv[3] = self.Rj + 2 * Rerr

        self.Tinv = Tinv
        import pdb; pdb.set_trace()

    def calc_radius_from_mass(self, Mp):
        """Calculate planet radius from mass

        Args:
            Mp (astropy Quantity array):
                Planet mass in units of Earth mass

        Returns:
            astropy Quantity array:
                Planet radius in units of Earth radius

        """

        m = np.array(Mp.to(u.M_earth).value, ndmin=1)
        R = np.zeros(m.shape)

        inds = np.digitize(m, np.hstack((0, self.T, np.inf)))
        for j in range(1, inds.max() + 1):
            if m[inds == j] < 159:
                R[inds == j] = 10.0 ** (
                np.log10(m[inds == j]) * self.S[0] + self.C[0]
            )
            else: 
                R[inds == j] = 10.0 ** (
                np.log10(m[inds == j]) * self.S[1] + self.C[1]
            )

        return R * u.R_earth

    def calc_mass_from_radius(self, Rp):
        """Calculate planet mass from radius

        Args:
            Rp (astropy Quantity array):
                Planet radius in units of Earth radius

        Returns:
            astropy Quantity array:
                Planet mass in units of Earth mass

        Note:
            The fit is non-invertible for Jupiter radii, so all those get 1
            Jupiter mass.


        """

        R = np.array(Rp.to(u.R_earth).value, ndmin=1)
        m = np.zeros(R.shape)
        m[np.isnan(R)] = np.nan

        inds = np.digitize(R, np.hstack((0, self.Tinv, np.inf)))
        for j in range(1, len(self.C) + 1):
            if R[inds == j] < 14.125:
                m[inds == j] = 10.0 ** ((np.log10(R[inds == j]) + self.C[0]) / self.S[0])
            else:
                m[inds == j] =  self.Mj
        return m * u.M_earth
