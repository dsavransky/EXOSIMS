import sys
from EXOSIMS.util.get_module import get_module

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
        PostProcessing (PostProcessing):
            PostProcessing class object
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
        Completeness (Completeness):
            Completeness class object
        BackgroundSources (BackgroundSources):
            Background Source class object
        ZodiacalLight (ZodiacalLight):
            Zodiacal light class object
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
            import json
            import os.path
        
            assert os.path.isfile(scriptfile), "%s is not a file."%scriptfile

            try:
                script = open(scriptfile).read()
                specs = json.loads(script)
            except ValueError:
                print "%s improperly formatted."%scriptfile
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise

        if 'modules' not in specs.keys():
            raise ValueError("No modules field found in script.")

        # get desired module names (prototype or specific)
        
        # import simulated universe class
        SimUni = get_module(specs['modules']['SimulatedUniverse'], 'SimulatedUniverse')
        # import observatory class
        Obs = get_module(specs['modules']['Observatory'], 'Observatory')
        # import timekeeping class
        TK = get_module(specs['modules']['TimeKeeping'], 'TimeKeeping')
        # import postprocessing class
        PP = get_module(specs['modules']['PostProcessing'], 'PostProcessing')
        
        #initialize top level
        self.modules = {}
        self.modules['SimulatedUniverse'] = SimUni(**specs)
        self.modules['Observatory'] = Obs(**specs)
        self.modules['TimeKeeping'] = TK(**specs)
        self.modules['PostProcessing'] = PP(**specs)
        
        #collect sub-initializations
        self.modules['OpticalSystem'] = self.modules['SimulatedUniverse'].OpticalSystem # optical system class object
        self.modules['PlanetPopulation'] = self.modules['SimulatedUniverse'].PlanetPopulation # planet population class object
        self.modules['ZodiacalLight'] = self.modules['SimulatedUniverse'].ZodiacalLight # zodiacal light class object
        self.modules['Completeness'] = self.modules['SimulatedUniverse'].Completeness # completeness class object
        self.modules['PlanetPhysicalModel'] = self.modules['SimulatedUniverse'].PlanetPhysicalModel # planet physical model class object
        self.modules['TargetList'] = self.modules['SimulatedUniverse'].TargetList # target list class object
        

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

        
