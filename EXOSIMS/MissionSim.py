import sys, json, inspect
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
                print "%s improperly formatted."%scriptfile
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise
        else:
            specs_from_file = {}

        # extend given specs with file specs
        specs.update(specs_from_file)

        if 'modules' not in specs.keys():
            raise ValueError("No modules field found in script.")

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


