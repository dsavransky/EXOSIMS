import sys, logging, json, os.path
import tempfile
from EXOSIMS.util.get_module import get_module
import random as py_random
import numpy as np
import copy, re, inspect, subprocess

class MissionSim(object):
    """Mission Simulation (backbone) class
    
    This class is responsible for instantiating all objects required 
    to carry out a mission simulation.
    
    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        PlanetPopulation (PlanetPopulation):
            PlanetPopulation class object
        PlanetPhysicalModel (PlanetPhysicalModel):
            PlanetPhysicalModel class object
        OpticalSystem (OpticalSystem):
            OpticalSystem class object
        ZodiacalLight (ZodiacalLight):
            ZodiacalLight class object
        BackgroundSources (BackgroundSources):
            Background Source class object
        PostProcessing (PostProcessing):
            PostProcessing class object
        Completeness (Completeness):
            Completeness class object
        TargetList (TargetList):
            TargetList class object
        SimulatedUniverse (SimulatedUniverse):
            SimulatedUniverse class object
        Observatory (Observatory):
            Observatory class object
        TimeKeeping (TimeKeeping):
            TimeKeeping class object
        SurveySimulation (SurveySimulation):
            SurveySimulation class object
        SurveyEnsemble (SurveyEnsemble):
            SurveyEnsemble class object
    
    """

    _modtype = 'MissionSim'
    _outspec = {}

    def __init__(self, scriptfile=None, **specs):
        """Initializes all modules from a given script file or specs dictionary.
        
        Args: 
            scriptfile (string):
                Path to JSON script file. If not set, assumes that 
                dictionary has been passed through specs.
            specs (dictionary):
                Dictionary containing additional user specification values and 
                desired module names.
        
        """
        
        # extend given specs with (JSON) script file
        if scriptfile is not None:
            assert os.path.isfile(scriptfile), "%s is not a file."%scriptfile
            try:
                script = open(scriptfile).read()
                specs_from_file = json.loads(script)
                specs_from_file.update(specs)
            except ValueError as err:
                print "Error: %s: Input file `%s' improperly formatted."%(self._modtype,
                        scriptfile)
                print "Error: JSON error was: ", err
                # re-raise here to suppress the rest of the backtrace.
                # it is only confusing details about the bowels of json.loads()
                raise ValueError(err)
            except:
                print "Error: %s: %s", (self._modtype, sys.exc_info()[0])
                raise
        else:
            specs_from_file = {}
        specs.update(specs_from_file)
        
        if 'modules' not in specs.keys():
            raise ValueError("No modules field found in script.")
        
        # set up numpy random number seed at top
        self.seed = specs.get('seed', py_random.randint(1, 1e9))
        specs['seed'] = self.seed
        print 'MissionSim seed is: ', self.seed
        
        # start logging, with log file and logging level (default: INFO)
        self.logfile = specs.get('logfile', None)
        self.loglevel = specs.get('loglevel', 'INFO').upper()
        specs['logger'] = self.get_logger(self.logfile, self.loglevel)
        specs['logger'].info('Start Logging: loglevel = %s'%specs['logger'].level \
                + ' (%s)'%self.loglevel)
        
        # populate outspec
        for att in self.__dict__.keys():
            self._outspec[att] = self.__dict__[att]
        
        # initialize top level, import modules
        self.SurveyEnsemble = get_module(specs['modules']['SurveyEnsemble'],
                'SurveyEnsemble')(**specs)
        self.SurveySimulation = get_module(specs['modules']['SurveySimulation'],
                'SurveySimulation')(**specs)
        
        # collect sub-initializations
        SS = self.SurveySimulation
        self.StarCatalog = SS.StarCatalog
        self.PlanetPopulation = SS.PlanetPopulation
        self.PlanetPhysicalModel = SS.PlanetPhysicalModel
        self.OpticalSystem = SS.OpticalSystem
        self.ZodiacalLight = SS.ZodiacalLight
        self.BackgroundSources = SS.BackgroundSources
        self.PostProcessing = SS.PostProcessing
        self.Completeness = SS.Completeness
        self.TargetList = SS.TargetList
        self.SimulatedUniverse = SS.SimulatedUniverse
        self.Observatory = SS.Observatory
        self.TimeKeeping = SS.TimeKeeping
        
        # create a dictionary of all modules, except StarCatalog
        self.modules = SS.modules
        self.modules['SurveySimulation'] = SS
        self.modules['SurveyEnsemble'] = self.SurveyEnsemble

    def get_logger(self, logfile, loglevel):
        r"""Set up logging object so other modules can use logging.info(),
        logging.warning, etc.
        
        Args:
            logfile (string):
                Path to the log file. If None, logging is turned off. 
                If supplied but empty string (''), a temporary file is generated.
            loglevel (string): 
                The level of log, defaults to 'INFO'. Valid levels are: CRITICAL, 
                ERROR, WARNING, INFO, DEBUG (case sensitive).
                
        Returns:
            logger (logging object):
                Mission Simulation logger.
        
        """
        
        # this leaves the default logger in place, so logger.warn will appear on stderr
        if logfile is None:
            logger = logging.getLogger(__name__)
            return logger
        
        # if empty string, a temporary file is generated
        if logfile == '':
            (dummy, logfile) = tempfile.mkstemp(prefix='EXOSIMS.', suffix='.log',
                    dir='/tmp', text=True)
        else:
            # ensure we can write it
            try:
                f = open(logfile, 'w')
                f.close()
            except (IOError, OSError) as e:
                print '%s: Failed to open logfile "%s"'%(__file__, logfile)
                return None
        print "Logging to '%s' at level '%s'"%(logfile, loglevel.upper())
        
        # convert string to a logging.* level
        numeric_level = getattr(logging, loglevel.upper())
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s'%loglevel.upper())
        
        # set up the top-level logger
        logger = logging.getLogger(__name__.split('.')[0])
        logger.setLevel(numeric_level)
        # do not propagate EXOSIMS messages to higher loggers in this case
        logger.propagate = False
        # create a handler that outputs to the named file
        handler = logging.FileHandler(logfile, mode='w')
        handler.setLevel(numeric_level)
        # logging format
        formatter = logging.Formatter('%(levelname)s: %(filename)s(%(lineno)s): '\
                +'%(funcName)s: %(message)s')
        handler.setFormatter(formatter)
        # add the handler to the logger
        logger.addHandler(handler)
        
        return logger

    def run_sim(self):
        """Convenience method that simply calls the SurveySimulation run_sim method.
        
        """
        
        res = self.SurveySimulation.run_sim()
        
        return res

    def reset_sim(self, genNewPlanets=True, rewindPlanets=True):
        """Convenience method that simply calls the SurveySimulation reset_sim method.
        
        """
        
        res = self.SurveySimulation.reset_sim(genNewPlanets=genNewPlanets,
                rewindPlanets=rewindPlanets)
        
        return res

    def run_ensemble(self, nb_run_sim, run_one=None, genNewPlanets=True, 
            rewindPlanets=True,kwargs={}):
        """Convenience method that simply calls the SurveyEnsemble run_ensemble method.
        
        """
        
        res = self.SurveyEnsemble.run_ensemble(self, nb_run_sim, run_one=run_one,
                genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets,kwargs=kwargs)
        
        return res

    def genOutSpec(self, tofile=None):
        """Join all _outspec dicts from all modules into one output dict
        and optionally write out to JSON file on disk.
        
        Args:
           tofile (string):
                Name of the file containing all output specifications (outspecs).
                Default to None.
                
        Returns:
            out (dictionary):
                Dictionary containing additional user specification values and 
                desired module names.
        
        """
        
        out = self.SurveySimulation.genOutSpec(tofile=tofile)
        
        return out
