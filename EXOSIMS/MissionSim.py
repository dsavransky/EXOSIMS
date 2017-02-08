import sys, json, inspect, subprocess
import copy
import re
import logging
import tempfile
from EXOSIMS.util.get_module import get_module
import random as py_random
import numpy as np
import astropy.units as u
import os.path


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

    def __init__(self,scriptfile=None,**specs):
        """Initializes all modules from a given script file or specs dictionary.
        
        Input: 
            scriptfile:
                JSON script file.  If not set, assumes that 
                dictionary has been passed through specs.
            specs: 
                Dictionary containing additional user specification values and 
                desired module names.
        """
        
        if scriptfile is not None:
            assert os.path.isfile(scriptfile), "%s is not a file."%scriptfile
            
            try:
                script = open(scriptfile).read()
                specs_from_file = json.loads(script)
                specs_from_file.update(specs)
            except ValueError as err:
                print "Error: %s: Input file `%s' improperly formatted." % (self._modtype, scriptfile)
                print "Error: JSON error was: ", err
                # re-raise here to suppress the rest of the backtrace.
                # it is only confusing details about the bowels of json.loads()
                raise ValueError(err)
            except:
                print "Error: %s: %s", (self._modtype, sys.exc_info()[0])
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
        seed = self.random_seed_initialize(specs)
        self._outspec['seed'] = seed
        
        #create the ensemble object first, before any specs are updated
        SurveyEns = get_module(specs['modules']['SurveyEnsemble'],'SurveyEnsemble')
        sens = SurveyEns(**specs)
        
        #preserve star catalog name
        self.StarCatalog = specs['modules']['StarCatalog']
        
        #initialize top level, import modules
        self.modules = {}
        self.modules['SimulatedUniverse'] = get_module(specs['modules']\
                ['SimulatedUniverse'],'SimulatedUniverse')(**specs)
        self.modules['Observatory'] = get_module(specs['modules']\
                ['Observatory'],'Observatory')(**specs)
        self.modules['TimeKeeping'] = get_module(specs['modules']\
                ['TimeKeeping'],'TimeKeeping')(**specs)
        
        # collect sub-initializations
        SU = self.modules['SimulatedUniverse']
        self.modules['PlanetPopulation'] = SU.PlanetPopulation
        self.modules['PlanetPhysicalModel'] = SU.PlanetPhysicalModel
        self.modules['OpticalSystem'] = SU.OpticalSystem
        self.modules['ZodiacalLight'] = SU.ZodiacalLight
        self.modules['BackgroundSources'] = SU.BackgroundSources
        self.modules['PostProcessing'] = SU.PostProcessing
        self.modules['Completeness'] = SU.Completeness
        self.modules['TargetList'] = SU.TargetList
        
        # replace modules dict with instantiated objects 
        SurveySim = get_module(specs['modules']['SurveySimulation'], 'SurveySimulation')
        inputMods = specs.pop('modules')
        specs['modules'] = self.modules
        
        # generate sim object
        self.modules['SurveySimulation'] = SurveySim(**specs)
        self.modules['SurveyEnsemble'] = sens
        
        # make all objects accessible from the top level
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
        
        return seed

    def genOutSpec(self, tofile=None):
        """
        Join all _outspec dicts from all modules into one output dict
        and optionally write out to JSON file on disk.
        """
        
        # start with a copy of our own module's _outspec
        out = copy.copy(self._outspec)
        
        # add in all module _outspec's
        for module in self.modules.values():
            out.update(module._outspec)
        
        # add in the specific module names used
        out['modules'] = {}
        for (mod_name, module) in self.modules.items():
            # find the module file 
            mod_name_full = module.__module__
            if mod_name_full.startswith('EXOSIMS'):
                # take just its short name if it is in EXOSIMS
                mod_name_short = mod_name_full.split('.')[-1]
            else:
                # take its full path if it is not in EXOSIMS - changing .pyc -> .py
                mod_name_short = re.sub('\.pyc$', '.py', inspect.getfile(module.__class__))
            out['modules'][mod_name] = mod_name_short
        
        # add in the SVN revision
        path = os.path.split(inspect.getfile(self.__class__))[0]
        rev = subprocess.Popen("git log -1 "+path+"| grep \"commit\" | awk '{print $2}'", stdout=subprocess.PIPE, shell=True)
        (gitRev, err) = rev.communicate()
        if isinstance(gitRev, basestring) & (len(gitRev) > 0):
            out['Revision'] = "Github last commit " + gitRev[:-1]
        # if not an SVN repository, add in the Github last commit
        else:
            rev = subprocess.Popen("svn info "+path+"| grep \"Revision\" | awk '{print $2}'", stdout=subprocess.PIPE, shell=True)
            (svnRev, err) = rev.communicate()
            if isinstance(svnRev, basestring) & (len(svnRev) > 0):
                out['Revision'] = "SVN revision is " + svnRev[:-1]
            else: 
                out['Revision'] = "Not a valid Github or SVN revision."
        print out['Revision']
        
        # preserve star catalog name
        # TODO: why is this special-cased?
        out['modules']['StarCatalog'] = self.StarCatalog
        
        # dump to file
        if tofile is not None:
            with open(tofile, 'w') as outfile:
                json.dump(out, outfile, sort_keys=True, indent=4, ensure_ascii=False, separators=(',', ': '),
                          default=array_encoder)
        
        # return it as well
        return out

def array_encoder(obj):
    r"""Encodes numpy arrays, astropy Time's, and astropy Quantity's, into JSON.
    
    Called from json.dump for types that it does not already know how to represent,
    like astropy Quantity's, numpy arrays, etc.  The json.dump() method encodes types
    like integers, strings, and lists itself, so this code does not see these types.
    Likewise, this routine can and does return such objects, which is OK as long as 
    they unpack recursively into types for which encoding is known."""
    
    from astropy.time import Time
    if isinstance(obj, Time):
        # astropy Time -> time string
        return obj.fits # isot also makes sense here
    if isinstance(obj, u.quantity.Quantity):
        # note: it is possible to have a numpy ndarray wrapped in a Quantity.
        # NB: alternatively, can return (obj.value, obj.unit.name)
        return obj.value
    if isinstance(obj, (np.ndarray, np.number)):
        # ndarray -> list of numbers
        return obj.tolist()
    if isinstance(obj, (complex, np.complex)):
        # complex -> (real, imag) pair
        return [obj.real, obj.imag]
    if callable(obj):
        # this case occurs for interpolants like PSF and QE
        # We cannot simply "write" the function to JSON, so we make up a string
        # to keep from throwing an error.
        # The fix is simple: when generating the interpolant, add a _outspec attribute
        # to the function (or the lambda), containing (e.g.) the fits filename, or the
        # explicit number -- whatever string was used.  Then, here, check for that 
        # attribute and write it out instead of this dummy string.  (Attributes can
        # be transparently attached to python functions, even lambda's.)
        return 'interpolant_function'
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode()
    # nothing worked, bail out
    
    return json.JSONEncoder.default(obj)

