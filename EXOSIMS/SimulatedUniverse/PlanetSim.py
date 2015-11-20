#!/usr/bin/env python
from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
#from EXOSIMS.PlanetPopulation.PlanetFunc import PlanetFunc as pf
import EXOSIMS.util.statsFun as statsFun
import astropy.units as u
import astropy.constants as const
import numpy as np
from random import randint


class PlanetSim(SimulatedUniverse):
        
        def planet_to_star(self):
            """Assigns planets to stars and returns index of star in target star
            list
            
            Output: planSys - numpy array containing indices of the target star 
            each planet (element in the list) belongs to
            This defines the data type expected, specific SimulatedUniverse class
            objects will populate these indices more realistically"""
        
            # assign one planet to each star in the target list
            planSys = np.array([],dtype=int)
            for i in range(len(self.TargetList.Name)):
                    nump = randint(0,8)
                    planSys = np.hstack((planSys, np.array([i]*nump,dtype=int)))
            return planSys
        
        
        def planet_masses(self):
            """Assigns each planet mass in kg
            
            This method has access to the following:
            self.TargetList - target list object
            self.PlanetPhysicalModel - planet physical model object
            self.OpticalSystem - optical system object
            self.PlanetPopulation - planet population object
            self.ZodiacalLight - zodiacal light object
            self.Completeness - completeness object
            self.planInds - indices mapping planet to target star
            self.nPlans - number of planets
            self.sysInds - indices of target stars with planets
            
            Output: M - numpy array containing masses of each planet (astropy 
            unit object in kg)
            
            This defines the data type expected, specific SimulatedUniverse class
            objects will populate these values more realistically"""
            
            mass_pdf = statsFun.simpSample(self.PlanetPopulation.mass, self.nPlans, .1, 100)
            
            # assign all planets the mass of earth in kg
            M = ([const.M_jup.value]*mass_pdf)*u.kg
            
            return M
        

        def planet_a(self):
            """Assigns each planet semi major axis in km
            
            Output: a - numpy array containing semi major axis of each planet (astropy 
            unit object in km)"""
            
            

            semi_axis_pdf = statsFun.simpSample(self.PlanetPopulation.semi_axis, self.nPlans, .1, 100)
            a = np.array([const.R_earth.value]*semi_axis_pdf)*u.km
            return a
        
        
        def planet_e(self):
            """Assigns each planet eccentricity
            
            Output: e - numpy array containing eccentricity of each planet"""
            
            eccentricity_pdf = statsFun.simpSample(self.PlanetPopulation.eccentricity, self.nPlans, 0, 1)
            e = np.array(eccentricity_pdf)
            return e
        
   
        def planet_radii(self):
            """Assigns each planet a radius in km
            
            This method has access to the following:
            self.TargetList - target list object
            self.PlanetPhysicalModel - planet physical model object
            self.OpticalSystem - optical system object
            self.PlanetPopulation - planet population object
            self.rules - rules object
            self.ZodiacalLight - zodiacal light object
            self.Completeness - completeness object
            self.planInds - indices mapping planet to target star
            self.nPlans - number of planets
            self.sysInds - indices of target stars with planets
            self.Mp - planet masses
            
            Output: R - numpy array containing radius of each planet (astropy 
            unit object in km)
            
            This defines the data type expected, specific SimulatedUniverse class
            objects will populate these values more realistically"""
            
            radii_pdf = statsFun.simpSample(self.PlanetPopulation.radius, self.nPlans, 1, 22.6)
            # assign all planets the radius of earth in km
            R = np.array([const.R_earth.value]*radii_pdf)*u.km
            
            return R
        
        
        