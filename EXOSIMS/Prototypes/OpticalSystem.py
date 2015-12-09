# -*- coding: utf-8 -*-
from astropy import units as u
import numpy as np
import os.path
import astropy.io.fits as fits
import numbers
from operator import itemgetter

class OpticalSystem(object):
    """Optical System class template
    
    This class contains all variables and methods necessary to perform
    Optical System Definition Module calculations in exoplanet mission 
    simulation.
    
    Args:
        \*\*specs:
            User specified values.
            
    Attributes:
        lam (Quantity):
            detection wavelength (default units of nm)
        shapeFac (float):
            shape factor (also known as fill factor) where 
            :math:`shapeFac * diameter^2 = Area`
        pupilArea (Quantity):
            entrance pupil area (default units of m\ :sup:`2`)
        pupilDiam (Quantity):
            entrance pupil diameter (default units of m)
        SNchar (float):
            Signal to Noise Ratio for characterization
        haveOcculter (bool):
            boolean signifying if the system has an occulter
        IWA (Quantity):
            Fundamental Inner Working Angle (default units of arcsecond)
        OWA (Quantity):
            Fundamental Outer Working Angle (default units of arcsecond)
        dMagLim (float):
            Fundamental limiting delta magnitude. 
        attenuation (float):
            non-coronagraph attenuation (throughput)
        deltaLam (Quantity):
            bandpass (default units of nm)
        telescopeKeepout (float):
            telescope keepout angle in degrees
        specLam (Quantity):
            spectral wavelength of interest (default units of nm)
            Note:  probably belongs in postprocessing or rules.
        intCutoff (Quantity):
            integration cutoff (default units of day)
        scienceInstruments (list of dicts):
            All science instrument attributes (variable)
        starlightSuppressionSystems (list of dicts):
            All starlight suppression system attributes (variable)

    Common Starlight Suppression System Attributes:

        PSF (callable(lam, alpha)):
            point spread function - 2D ndarray of values, normalized to 1 at the core
            (must be callable - can be lambda, function,
            scipy.interpolate.interp2d object, etc.) with inputs wavelength (astropy
            Quantity) and angular separation (astropy Quantity). Note normalization 
            means that all throughput effects must be contained in the throughput 
            attribute.
        PSFSampling (Quantity):
            Sampling of PSF in arcsec/pixel (default units of arcsec)
        throughput (callable(lam, alpha)):
            optical system throughput (must be callable - can be lambda, function,
            scipy.interpolate.interp2d object, etc.) with inputs wavelength (astropy
            Quantity) and angular separation (astropy Quantity).
        contrast (callable(lam, alpha)):
            optical system contrast curve (must be callable - can be lambda, function,
            scipy.interpolate.interp2d object, etc.) with inputs wavelength (astropy
            Quantity) and angular separation (astropy Quantity).

    Common Science Instrument Attributes:
        QE (callable(lam)):
            detector quantum efficiency (must be callable - can be lambda, function,
            scipy.interpolate.interp2d object, etc.) with inputs wavelength (astropy
            Quantity).  
     
    """

    _modtype = 'OpticalSystem'
    
    def __init__(self, lam=500.,deltaLam=100.,shapeFac=np.pi/4.,\
            pupilDiam=4.,SNchar=11., telescopeKeepout=45.,\
            IWA=None,OWA=None,dMagLim=None, attenuation=0.57,\
            specLam=760.,intCutoff=50.,\
            starlightSuppressionSystems=None, scienceInstruments=None, **specs):
        
        #must have a starlight suppression system and science instrument defined
        if not starlightSuppressionSystems:
            raise ValueError("No starlight suppression systems defined.")

        #load all values with defaults
        self.lam = float(lam)*u.nm              # detection wavelength (nm) 
        self.deltaLam = float(deltaLam)*u.nm              # detection wavelength (nm) 
        self.shapeFac = float(shapeFac)         # shapeFac*diameter**2 = Area
        self.pupilDiam = float(pupilDiam)*u.m   # entrance pupil diameter
        self.pupilArea = self.shapeFac * (self.pupilDiam)**2. # entrance pupil area
        self.SNchar = float(SNchar)             # Signal to Noise Ratio for characterization
        self.telescopeKeepout = float(telescopeKeepout)*u.deg # keepout angle in degrees
        self.attenuation = float(attenuation)   #non-coronagraph attenuation factor
        self.specLam = float(specLam)*u.nm  # spectral wavelength of interest
        self.intCutoff = float(intCutoff)*u.day #integration time cutoff 
                        

        # now loop through starlight suppression systems and verify what's there.
        # the only things that are really required is the system number, the type (external/internal) 
        # and that the whole thing is a dict
        sysIWAs = []
        sysOWAs = []
        sysdMagLims = []
        self.haveOcculter = False 
        for sys in starlightSuppressionSystems:
            assert isinstance(sys,dict), "Starlight suppression systems must be defined as dicts."
            assert sys.has_key('starlightSuppressionSystemNumber'),\
                    "All starlight suppression systems must have key starlightSuppressionSystemNumber."
            assert sys.has_key('type') and isinstance(sys['type'],basestring),\
                    "All starlight suppression systems must have key type."
            if (sys['type'].lower() == 'external') or  (sys['type'].lower() == 'hybrid'):
                self.haveOcculter = True

            #check for throughput
            if sys.has_key('throughput'):
                if isinstance(sys['throughput'],basestring):
                    assert os.path.isfile(sys['throughput']),\
                            "%s is not a valid file."%sys['throughput']
                    tmp = fits.open(sys['throughput'])
                    #basic validation here for size and IWA/OWA
                    #sys['throughput'] = lambda or interp
                elif isinstance(sys['throughput'],numbers.Number):
                    sys['throughput'] = lambda lam, alpha: float(sys['throughput'])

            #check for contrast
            if sys.has_key('contrast'):
                if isinstance(sys['contrast'],basestring):
                    assert os.path.isfile(sys['contrast']),\
                            "%s is not a valid file."%sys['contrast']
                    tmp = fits.open(sys['contrast'])
                    #basic validation here for size and IWA/OWA
                    #sys['contrast'] = lambda or interp
                elif isinstance(sys['contrast'],numbers.Number):
                    sys['contrast'] = lambda lam, alpha: float(sys['contrast'])

            #check for PSF
            if sys.has_key('PSF'):
                if isinstance(sys['PSF'],basestring):
                    assert os.path.isfile(sys['PSF']),\
                            "%s is not a valid file."%sys['PSF']
                    tmp = fits.open(sys['PSF'])
                    #basic validation here for size and IWA/OWA
                    #sys['PSF'] = lambda or interp
                else:
                    #placeholder PSF
                    sys['PSF'] = lambda lam, alpha: np.ones((3,3))

            # sampling of PSF in arcsec/pixel (otherwise should grab from PSF file header)
            if sys.has_key('PSFSampling'):
                sys['PSFSampling'] = float(sys['PSFSampling'])*u.arcsec

            # optical system overhead time
            if sys.has_key('opticaloh'):
                sys['opticaloh'] = float(sys['opticaloh'])*u.day
            else:
                sys['opticaloh'] = 1.*u.day

            #time multipliers
            if not sys.has_key('detectionTimeMultipler'):
                sys['detectionTimeMultiplier'] = 1.
            if not sys.has_key('characterizationTimeMultiplier'):
                sys['characterizationTimeMultiplier'] = 1.
    
            #check for system's IWA and OWA
            if sys.has_key('IWA'):
                sysIWAs.append(sys['IWA'])
            if sys.has_key('OWA'):
                if sys['OWA'] == 0: 
                    sys['OWA'] = np.Inf
                sysOWAs.append(sys['OWA'])
              
        self.starlightSuppressionSystems = sorted(starlightSuppressionSystems, key=itemgetter('starlightSuppressionSystemNumber')) 

        #populate IWA, OWA, and dMagLim as required
        if IWA is not None:
            self.IWA = float(IWA)*u.arcsec
        else:
            if len(sysIWAs) > 0:
                self.IWA = float(min(sysIWAs))*u.arcsec
            else:
                raise ValueError("Could not determine fundamental IWA.")

        if OWA is not None:
            if OWA == 0:
                self.OWA = np.inf*u.arcsec
            else:
                self.OWA = float(OWA)*u.arcsec
        else:
            if len(sysOWAs) > 0:
                self.OWA = float(max(sysOWAs))*u.arcsec
            else:
                raise ValueError("Could not determine fundamental OWA.")

        if dMagLim is not None:
            self.dMagLim = float(dMagLim)
        else:
            if len(sysdMagLims) > 0:
                self.dMagLim = float(max(sysdMagLims))
            else:
                raise ValueError("Could not determine fundamental dMagLim.")

        assert self.IWA < self.OWA, "Fundamnetal IWA must be smaller that the OWA."


        #now go through all science Instruments
        for sys in scienceInstruments:
            assert isinstance(sys,dict), "Science instruments must be defined as dicts."
            assert sys.has_key('scienceInstrumentNumber'),\
                    "All science instruments must have key scienceInstrumentNumber."
            assert sys.has_key('type') and isinstance(sys['type'],basestring),\
                    "All science instruments must have key type."


            if sys.has_key('pixelArea'): 
                sys['pixelArea'] = float(sys['pixelArea'])*(u.m)**2 #pixel are in m^2
            if sys.has_key('focalLength'):
                sys['focalLength'] = float(sys['focalLength'])*u.m #focal length in m
            if sys.has_key('darkRate'):
                sys['darkRate'] = float(sys['darkRate'])*(1/u.s) # detector dark-current rate per pixel
            if sys.has_key('texp'):
                sys['texp'] = float(sys['texp'])*u.s  # exposure time per read (s)
            if sys.has_key('readNoise'):
                sys['readNoise'] = float(sys['readNoise'])  # detector read noise (e/s)
            if sys.has_key('Rspec'):
                sys['Rspec'] = float(sys['Rspec']) 
            if sys.has_key('Npix'):
                sys['Npix'] = float(sys['Npix']) 
            if sys.has_key('Ndark'):
                sys['Ndark'] = int(sys['Ndark']) 
            if sys.has_key('ENF'):
                sys['ENF'] = float(sys['ENF'])
            if sys.has_key('CIC'):
                sys['CIC'] = float(sys['CIC'])

            if sys.has_key('QE'):
                if isinstance(sys['QE'],basestring):
                    assert os.path.isfile(sys['QE']),\
                            "%s is not a valid file."%sys['QE']
                    tmp = fits.open(sys['QE'])
                    #basic validation here for size and wavelength
                    #sys['QE'] = lambda or interp
                elif isinstance(sys['QE'],numbers.Number):
                    sys['QE'] = lambda lam, alpha: float(sys['QE'])
        
        self.scienceInstruments = sorted(scienceInstruments, key=itemgetter('scienceInstrumentNumber')) 


        #finally:  if we only have one science instrument/suppression system, bring their attributes to top
        if len(self.starlightSuppressionSystems) == 1:
            for key in self.starlightSuppressionSystems[0].keys():
                if key not in self.__dict__.keys():
                    setattr(self, key, self.starlightSuppressionSystems[0][key])
        if len(self.scienceInstruments) == 1:
            for key in self.scienceInstruments[0].keys():
                if key not in self.__dict__.keys():
                    setattr(self, key, self.scienceInstruments[0][key])
       

        # set values derived from quantities above
        #if np.isinf(self.OWA.value):
        #    self.OWA = np.pi/2
                
    def __str__(self):
        """String representation of the Optical System object
        
        When the command 'print' is used on the Optical System object, this 
        method will print the attribute values contained in the object"""
        
        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Optical System class object attributes'
        
    def calc_maxintTime(self, targlist):
        """Finds maximum integration time for target systems 
        
        This method is called in the __init__ method of the TargetList class
        object.
        
        This method defines the data type expected, maximum integration time is
        determined by specific OpticalSystem classes.
        
        Args:
            targlist (TargetList):
                TargetList class object which, in addition to TargetList class
                object attributes, has available:
                    targlist.OpticalSystem:
                        OpticalSystem class object
                    targlist.PlanetPopulation:
                        PlanetPopulation class object
                    targlist.ZodiacalLight:
                        ZodiacalLight class object
                    targlist.Completeness:
                        Completeness object
        
        Returns:
            maxintTime (Quantity):
                1D numpy ndarray of maximum integration times (default units of
                day)
        
        """
        
        # maximum integration time is given as 1 day for each system in target list
        maxintTime = np.array([1.]*len(targlist.Name))*u.day
        
        return maxintTime
        
    def calc_intTime(self, targlist, universe, s_ind, planInds):
        """Finds integration time for a specific target system 
        
        This method is called by a method in the SurveySimulation object class.
        
        This method defines the data type expected, integration time is 
        determined by specific OpticalSystem classes.
        
        Args:
            targlist (TargetList):
                TargetList class object
            universe (SimulatedUniverse):
                SimulatedUniverse class object
            s_ind (int):
                target star index
            planInds (ndarray):
                planet index linking back to target star
        
        Returns:
            intTime (Quantity):
                1D numpy ndarray of integration times (default units of day)
      
      """
        
        # integration time given as 1 day
        intTime = np.array([1.]*len(planInds))*u.day
        
        return intTime
