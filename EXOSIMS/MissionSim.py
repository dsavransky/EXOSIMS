import sys, json, inspect
import logging
import tempfile
from EXOSIMS.util.get_module import get_module
import random as py_random
import numpy as np
import os.path


class MissionSim(object):
    """Mission Simulation (backbone) class
    
    This class is responsible for instantiating all objects required 
    to carry out a mission simulation.

    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        SimulatedUniverse (SimulatedUniverse):
            SimulatedUniverse class object
        Observatory (Observatory):
            Observatory class object
        TimeKeeping (TimeKeeping):
            TimeKeeping class object
        TargetList (TargetList):
            TargetList class object
        PlanetPhysicalModel (PlanetPhysicalModel):
            PlanetPhysicalModel class object
        OpticalSystem (OpticalSystem):
            OpticalSystem class object
        PlanetPopulation (PlanetPopulation):
            PlanetPopulation class object
        ZodiacalLight (ZodiacalLight):
            ZodiacalLight class object
        PostProcessing (PostProcessing):
            PostProcessing class object
        Completeness (Completeness):
            Completeness class object
        BackgroundSources (BackgroundSources):
            Background Source class object
        SurveyEnsemble (SurveyEnsemble):
            Survey Ensemble class object
    """


    def __init__(self,scriptfile=None,**specs):
        """Initializes all modules

        Input: 
            scriptfile:
                JSON script file.  If not set, assumes that 
                dictionary has been passed through specs 
            specs: 
                Dictionary containing user specification values and 
                desired module names
        """

        if scriptfile is not None:
            assert os.path.isfile(scriptfile), "%s is not a file."%scriptfile

            try:
                script = open(scriptfile).read()
                specs_from_file = json.loads(script)
            except ValueError:
                print "Error: Input file `%s' improperly formatted."%scriptfile
                raise
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise
        else:
            specs_from_file = {}

        # extend given specs with file specs
        specs.update(specs_from_file)

        if 'modules' not in specs.keys():
            raise ValueError("No modules field found in script.")

        # set up log file, if it was desired
        self.start_logging(specs)
        # in this module, use the logger like this:
        # logger = logging.getLogger(__name__)
        # logger.info('__init__ started logging.')

        # set up numpy random number seed at top
        self.random_seed_initialize(specs)

        #preserve star catalog name
        self.StarCatalog = specs['modules']['StarCatalog']

        #initialize top level
        self.modules = {}
        # import simulated universe class
        SimUni = get_module(specs['modules']['SimulatedUniverse'], 'SimulatedUniverse')
        self.modules['SimulatedUniverse'] = SimUni(**specs)

        # import observatory class
        Obs = get_module(specs['modules']['Observatory'], 'Observatory')
        self.modules['Observatory'] = Obs(**specs)

        # import timekeeping class
        TK = get_module(specs['modules']['TimeKeeping'], 'TimeKeeping')
        self.modules['TimeKeeping'] = TK(**specs)

        #collect sub-initializations
        self.modules['OpticalSystem'] = self.modules['SimulatedUniverse'].OpticalSystem # optical system object
        self.modules['PlanetPopulation'] = self.modules['SimulatedUniverse'].PlanetPopulation # planet population object
        self.modules['ZodiacalLight'] = self.modules['SimulatedUniverse'].ZodiacalLight # zodiacal light object
        self.modules['BackgroundSources'] = self.modules['SimulatedUniverse'].BackgroundSources #Background sources object
        self.modules['Completeness'] = self.modules['SimulatedUniverse'].Completeness # completeness object
        self.modules['PlanetPhysicalModel'] = self.modules['SimulatedUniverse'].PlanetPhysicalModel # planet physical model object
        self.modules['PostProcessing'] = self.modules['SimulatedUniverse'].PostProcessing # postprocessing model object
        self.modules['TargetList'] = self.modules['SimulatedUniverse'].TargetList # target list object
        
        #grab sim and ensemble classes  
        SurveySim = get_module(specs['modules']['SurveySimulation'], 'SurveySimulation')
        SurveyEns = get_module(specs['modules']['SurveyEnsemble'], 'SurveyEnsemble')

        #replace modules dict with instantiated objects 
        inputMods = specs.pop('modules')
        specs['modules'] = self.modules

        #generate sim and ensemble objects
        self.modules['SurveySimulation'] = SurveySim(**specs)
        self.modules['SurveyEnsemble'] = SurveyEns(**specs)

        #make all objects accessible from the top level
        for modName in specs['modules'].keys():
            setattr(self, modName, specs['modules'][modName])

    def start_logging(self, specs):
        r"""Set up logging object so other modules can use logging.info(), logging.warning, etc.

        Two entries in the specs dictionary are used:
        logfile: if not present, logging is turned off; if supplied, but empty, a
            temporary file is generated; otherwise, the named file is opened for writing.
        loglevel: if present, the given level is used, else the logging level is INFO.
            Valid levels are: CRITICAL, ERROR, WARNING, INFO, DEBUG (case is ignored).

        Args:
            specs: dictionary

        Returns:
            logfile: string
                The name of the log file, or None if there was none, in case the tempfile needs to be recorded.
        """
        # get the logfile name
        if 'logfile' not in specs:
            return None # this leaves the default logger in place, so logger.warn will appear on stderr
        logfile = specs['logfile']
        if not logfile:
            (dummy,logfile) = tempfile.mkstemp(suffix='.log', prefix='EXOSIMS.', dir='/tmp', text=True)
        else:
            # ensure we can write it
            try:
                f = open(logfile, 'w')
                f.close()
            except (IOError, OSError) as e:
                print '%s: Failed to open logfile "%s"' % (__file__, logfile)
                return None
        # get the logging level
        if 'loglevel' in specs:
            loglevel = specs['loglevel'].upper()
        else:
            loglevel = 'INFO'
        # convert string to a logging.* level
        numeric_level = getattr(logging, loglevel)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)

        # set up the top-level logger
        root_logger = logging.getLogger(__name__.split('.')[0])
        root_logger.setLevel(numeric_level)
        # do not propagate EXOSIMS messages to higher loggers in this case
        root_logger.propagate = False
        # create a handler that outputs to the named file
        handler = logging.FileHandler(logfile, mode='w')
        handler.setLevel(numeric_level)
        # logging format
        formatter = logging.Formatter('%(levelname)s: %(filename)s(%(lineno)s): %(funcName)s: %(message)s')
        handler.setFormatter(formatter)
        # add the handler to the logger
        root_logger.addHandler(handler)

        # use the logger
        print '%s: Beginning logging to "%s" at level %s' % (os.path.basename(__file__), logfile, loglevel)
        logger = logging.getLogger(__name__)
        logger.info('Starting log.')
        return logfile

    def random_seed_initialize(self, specs):
        r"""Initialize random number seed for simulation repeatability.

        Algorithm: Get a large but printable integer from the system generator, which is seeded 
        automatically, and use this number to seed the numpy generator.  Otherwise, if a seed was
        given explicitly, use it instead."""
        if 'seed' in specs:
            seed = specs['seed']
        else:
            seed = py_random.randint(1,1e9)
        print 'MissionSim: Seed is: ', seed
        # give this seed to numpy
        np.random.seed(seed)

    def genOutSpec(self, tofile=None):
        """
        Join all _outspec dicts from all modules into one output dict
        and optionally write out to JSON file on disk.
        """

        out = {}
        out['modules'] = {}
        for modName in self.modules.keys():
            for key in self.modules[modName]._outspec.keys():
                if isinstance(self.modules[modName]._outspec[key],np.ndarray):
                    out[key] = list(self.modules[modName]._outspec[key])
                else:
                    out[key] = self.modules[modName]._outspec[key]

            #and grab the module file (just name if its in EXOSIMS)
            mod = self.modules[modName].__module__
            if mod.split('.')[0] == 'EXOSIMS':
                mod = mod.split('.')[-1]
            else:
                mod = os.path.splitext(inspect.getfile(self.modules[modName].__class__))[0]+'.py'

            out['modules'][modName] = mod

        #preserve star catalog name
        out['modules']['StarCatalog'] = self.StarCatalog

        if tofile is not None:
            with open(tofile, 'w') as outfile:
                json.dump(out, outfile, sort_keys=True, indent=4, ensure_ascii=False, separators=(',', ': '))

        return out


