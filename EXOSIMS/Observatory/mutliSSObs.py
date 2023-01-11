from EXOSIMS.Observatory.SotoStarshade import SotoStarshade
import numpy as np
import astropy.units as u
import astropy.constants as const


class multiSS_observatory(SotoStarshade):
    def __init__(
        self,
        scMass=[6000.0, 6000.0],
        dryMass=[3400.0, 3400.0],
        counter=0,
        counter_1=0,
        **specs
    ):
        SotoStarshade.__init__(self, **specs)

        # occulters' initial wet mass (kg)
        self.scMass = scMass * u.kg
        self.counter = counter
        self.counter_1 = counter_1
        # occulters' dry mass(kg)
        self.dryMass = np.array(dryMass) * u.kg

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
        if self.counter == 0:

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

            # force on current occulter
            Mfactor = -self.scMass[0] * const.M_sun * const.G
            F_sO = (
                r_Os
                / (np.linalg.norm(r_Os.to("AU").value) * r_Os.unit) ** 3.0
                * Mfactor
            )
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
                r_obs
                / (np.linalg.norm(r_obs.to("AU").value) * r_obs.unit) ** 3.0
                * Mfactor
            )
            F_ET = (
                r_TE
                / (np.linalg.norm(r_TE.to("AU").value) * r_TE.unit) ** 3.0
                * Mfactor
                / 328900.56
            )
            F_T = F_sT + F_ET
            # differential forces
            dF = F_O - F_T * self.scMass[0] / self.coMass
            dF_axial = (dF.dot(u_targ)).to("N")
            dF_lateral = (dF - dF_axial * u_targ).to("N")
            dF_lateral = np.linalg.norm(dF_lateral.to("N").value) * dF_lateral.unit
            dF_axial = np.abs(dF_axial)
            self.counter = self.counter + 1
            return dF_lateral, dF_axial

        else:
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

            # force on current occulter
            Mfactor = -self.scMass[1] * const.M_sun * const.G
            F_sO = (
                r_Os
                / (np.linalg.norm(r_Os.to("AU").value) * r_Os.unit) ** 3.0
                * Mfactor
            )
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
                r_obs
                / (np.linalg.norm(r_obs.to("AU").value) * r_obs.unit) ** 3.0
                * Mfactor
            )
            F_ET = (
                r_TE
                / (np.linalg.norm(r_TE.to("AU").value) * r_TE.unit) ** 3.0
                * Mfactor
                / 328900.56
            )
            F_T = F_sT + F_ET
            # differential forces
            dF = F_O - F_T * self.scMass[1] / self.coMass
            dF_axial = (dF.dot(u_targ)).to("N")
            dF_lateral = (dF - dF_axial * u_targ).to("N")
            dF_lateral = np.linalg.norm(dF_lateral.to("N").value) * dF_lateral.unit
            dF_axial = np.abs(dF_axial)
            self.counter = 0
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
        if self.counter_1 == 0:

            intMdot = (dF_lateral / self.skEff / const.g0 / self.skIsp).to("kg/s")
            mass_used = (intMdot * t_int).to("kg")
            deltaV = (dF_lateral / self.scMass[0] * t_int).to("km/s")
            self.counter_1 = self.counter_1 + 1
            return intMdot, mass_used, deltaV

        else:
            intMdot = (dF_lateral / self.skEff / const.g0 / self.skIsp).to("kg/s")
            mass_used = (intMdot * t_int).to("kg")
            deltaV = (dF_lateral / self.scMass[1] * t_int).to("km/s")
            self.counter_1 = 0
            return intMdot, mass_used, deltaV
