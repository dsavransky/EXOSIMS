from EXOSIMS.Observatory.SotoStarshade import SotoStarshade
import numpy as np
import astropy.units as u
import astropy.constants as const


class TwoStarShades_mission(SotoStarshade):
    def __init__(
        self,
        scMass=[6000.0, 6000.0],
        dryMass=[3400.0, 3400.0],
        counter_1=0,
        counter_2=0,
        counter=0,
        **specs
    ):
        self.counter = counter
        self.counter_1 = counter_1
        self.counter_2 = counter_2
        SotoStarshade.__init__(self, **specs)

        # occulters' initial wet mass (kg)
        self.scMass = np.array([scMass]) * u.kg

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

    def star_angularSep(self, TL, old_sInd, sInds, currentTime):
        """Finds angular separation from old star to given list of stars

        This method returns the angular separation from the last observed
        star to all others on the given list at the currentTime.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            old_sInd (int):
                Integer index of the last star of interest
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            currentTime (~astropy.time.Time):
                Current absolute mission time in MJD

        Returns:
            float:
                Angular separation between two target stars
        """
        if old_sInd is None:
            sd = np.zeros(len(sInds)) * u.rad
        else:
            # position vector of previous target star
            r_old = TL.starprop(old_sInd, currentTime)[0]
            u_old = r_old.to("AU").value / np.linalg.norm(r_old.to("AU").value)
            # position vector of new target stars
            r_new = TL.starprop(sInds, currentTime)
            u_new = (
                r_new.to("AU").value.T / np.linalg.norm(r_new.to("AU").value, axis=1)
            ).T
            # angle between old and new stars
            sd = np.arccos(np.clip(np.dot(u_old, u_new.T), -1, 1)) * u.rad

            # A-frame
            a1 = u_old / np.linalg.norm(u_old)  # normalized old look vector
            a2 = np.array([a1[1], -a1[0], 0])  # normal to a1
            a3 = np.cross(a1, a2)  # last part of the A basis vectors

            # finding sign of angle
            # The star angular separation can be negative
            u2_Az = np.dot(a3, u_new.T)
            sgn = np.sign(u2_Az)
            sgn[np.where(sgn == 0)] = 1
            sd = sgn * sd
        return sd

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

    def calculate_slewTimes(self, TL, old_sInd, sInds, sd, obsTimes, currentTime):
        """Finds slew times and separation angles between target stars

        This method determines the slew times of an occulter spacecraft needed
        to transfer from one star's line of sight to all others in a given
        target list.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            old_sInd (int):
                Integer index of the most recently observed star
            sInds (~numpy.ndarray(int)):
                Integer indices of the star of interest
            sd (~astropy.units.Quantity):
                Angular separation between stars in rad
            obsTimes (~astropy.time.Time(~numpy.ndarray)):
                Observation times for targets.
            currentTime (astropy Time):
                Current absolute mission time in MJD

        Returns:
            ~astropy.units.Quantity:
                Time to transfer to new star line of sight in units of days
        """
        if self.counter_2 == 0:

            self.ao[0] = self.thrust / self.scMass[0]
            slewTime_fac = (
                (
                    2.0
                    * self.occulterSep
                    / np.abs(self.ao[0])
                    / (self.defburnPortion / 2.0 - self.defburnPortion**2.0 / 4.0)
                )
                .decompose()
                .to("d2")
            )

            if old_sInd is None:
                slewTimes = np.zeros(TL.nStars) * u.d
            else:
                # calculate slew time
                slewTimes = np.sqrt(
                    slewTime_fac * np.sin(abs(sd) / 2.0)
                )  # an issue exists if sd is negative

                # The following are debugging
                assert (
                    np.where(np.isnan(slewTimes))[0].shape[0] == 0
                ), "At least one slewTime is nan"
            self.counter_2 = self.counter_2 + 1
            return slewTimes

        else:
            self.ao[1] = self.thrust / self.scMass[1]
            slewTime_fac = (
                (
                    2.0
                    * self.occulterSep
                    / np.abs(self.ao[1])
                    / (self.defburnPortion / 2.0 - self.defburnPortion**2.0 / 4.0)
                )
                .decompose()
                .to("d2")
            )

            if old_sInd is None:
                slewTimes = np.zeros(TL.nStars) * u.d
            else:
                # calculate slew time
                slewTimes = np.sqrt(
                    slewTime_fac * np.sin(abs(sd) / 2.0)
                )  # an issue exists if sd is negative

                # The following are debugging
                assert (
                    np.where(np.isnan(slewTimes))[0].shape[0] == 0
                ), "At least one slewTime is nan"
            self.counter_2 = 0
            return slewTimes

    def log_occulterResults(self, DRM, slewTimes, sInd, sd, dV):
        """Updates the given DRM to include occulter values and results

        Args:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
            slewTimes (astropy.units.Quantity):
                Time to transfer to new star line of sight in units of days
            sInd (int):
                Integer index of the star of interest
            sd (astropy.units.Quantity):
                Angular separation between stars in rad
            dV (astropy.units.Quantity):
                Delta-V used to transfer to new star line of sight in units of m/s

        Returns:
            dict:
                Design Reference Mission dictionary, contains the results of one
                complete observation (detection and characterization)

        """

        DRM["slew_time"] = slewTimes.to("day")
        DRM["slew_angle"] = sd.to("deg")

        slew_mass_used = slewTimes * self.defburnPortion * self.flowRate
        DRM["slew_dV"] = (slewTimes * self.ao * self.defburnPortion).to("m/s")
        DRM["slew_mass_used"] = slew_mass_used.to("kg")
        self.scMass = self.scMass - slew_mass_used
        DRM["scMass"] = self.scMass.to("kg")
        if self.twotanks:
            self.slewMass = self.slewMass - slew_mass_used
            DRM["slewMass"] = self.slewMass.to("kg")
        return DRM

    def refuel_tank(self, TK, tank=None):
        """Attempt to refuel a fuel tank and report status

        Args:
            TK (:ref:`TimeKeeping`):
                TimeKeeping object. Not used in prototype but an input for any
                implementations that wish to do time-aware operations.
            tank (str, optional):
                Either 'sk' or 'slew' when ``twotanks`` is True. Otherwise, None.
                Defaults None

        Returns:
            bool:
                True represents successful refeuling. False means refueling is not
                possible for selected tank.
        """

        if not (self.allowRefueling):
            return False

        if self.external_fuel_mass <= 0 * u.kg:
            return False

        if tank is not None:
            assert tank.lower() in ["sk", "slew"], "Tank must be 'sk' or 'slew'."
            assert self.twotanks, "You may only specify a tank when twotanks is True."

            if tank == "sk":
                tank_mass = self.skMass
                tank_capacity = self.skMaxFuelMass
                tank_name = "stationkeeping"
            else:
                tank_mass = self.slewMass
                tank_capacity = self.slewMaxFuelMass
                tank_name = "slew"
        else:
            tank_mass = self.scMass - self.dryMass
            tank_capacity = self.maxFuelMass
            tank_name = ""

        # Add as much fuel as can fit in the tank (plus any currently carried negative
        # value, or whatever remains in the external tank)
        topoff = (
            np.min(
                [
                    self.external_fuel_mass.to(u.kg).value,
                    (tank_capacity - tank_mass).to(u.kg).value,
                ]
            )
            * u.kg
        )
        assert topoff >= 0 * u.kg, "Topoff calculation produced negative result."

        self.external_fuel_mass -= topoff
        tank_mass += topoff
        if tank is not None:
            self.scMass += topoff
        self.vprint("{} {} fuel added".format(topoff, tank_name))
        self.vprint("{} remaining in external tank.".format(self.external_fuel_mass))

        return True
