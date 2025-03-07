from EXOSIMS.PlanetPhysicalModel.FortneyMarleyCahoyMix1 import FortneyMarleyCahoyMix1
import astropy.units as u
import numpy as np


class ForecasterMod(FortneyMarleyCahoyMix1):
    """Planet M-R relation model based on modification of the FORECASTER
    best-fit model (Chen & Kippling 2016) as described in Savransky et al. (2019)

    This modification forces all planets below hydrogen burning to have a maximum
    radius of 1 R_jupiter and also adds the Saturn density as an explicit point
    in the model

    """

    def __init__(self, **specs):

        FortneyMarleyCahoyMix1.__init__(self, **specs)

        S = np.array([0.2790, 0, 0, 0, 0.881])
        C = np.array([np.log10(1.008), 0, 0, 0, 0])
        T = np.array(
            [
                2.04,
                95.16,
                (u.M_jupiter).to(u.M_earth),
                ((0.0800 * u.M_sun).to(u.M_earth)).value,
            ]
        )

        Rj = u.R_jupiter.to(u.R_earth)
        Rs = 8.522  # saturn radius
        S[1] = (np.log10(Rs) - (C[0] + np.log10(T[0]) * S[0])) / (
            np.log10(T[1]) - np.log10(T[0])
        )
        C[1] = np.log10(Rs) - np.log10(T[1]) * S[1]
        S[2] = (np.log10(Rj) - np.log10(Rs)) / (np.log10(T[2]) - np.log10(T[1]))
        C[2] = np.log10(Rj) - np.log10(T[2]) * S[2]
        C[3] = np.log10(Rj)
        C[4] = np.log10(Rj) - np.log10(T[3]) * S[4]

        self.S = S
        self.C = C
        self.T = T
        self.Rj = Rj

        self.Mj = (u.M_jupiter).to(u.M_earth)
        Tinv = self.calc_radius_from_mass(self.T * u.M_earth).value
        Rerr = np.abs(self.calc_radius_from_mass(1 * u.M_jupiter)[0].value - self.Rj)
        Tinv[2] = self.Rj - 2 * Rerr
        Tinv[3] = self.Rj + 2 * Rerr

        self.Tinv = Tinv

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
            R[inds == j] = 10.0 ** (
                self.C[j - 1] + np.log10(m[inds == j]) * self.S[j - 1]
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
            if j == 4:
                m[inds == j] = self.Mj
            else:
                m[inds == j] = 10.0 ** (
                    (np.log10(R[inds == j]) - self.C[j - 1]) / self.S[j - 1]
                )

        return m * u.M_earth
