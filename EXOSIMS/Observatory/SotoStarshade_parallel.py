from EXOSIMS.Observatory.SotoStarshade_ContThrust import SotoStarshade_ContThrust
from EXOSIMS.Prototypes.TargetList import TargetList

import numpy as np
import sys
import ipyparallel as ipp

EPS = np.finfo(float).eps


class SotoStarshade_parallel(SotoStarshade_ContThrust):
    """ StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions, 
    and integrators to calculate occulter dynamics. 
    """
    
    def __init__(self,orbit_datapath=None,**specs): 

        SotoStarshade_ContThrust.__init__(self,**specs)  
        
        self.rc = ipp.Client()
        self.dview = self.rc[:]
        
        self.dview.block = False
        
        self.engine_ids = self.rc.ids
        self.nEngines = len(self.engine_ids)
        
        
    def run_ensemble(self,fun,nStars,tA,dtRange,m0):
        
        fTL = TargetList(**{"ntargs":nStars,"quadrant":0,'modules':{"StarCatalog": "FakeCatalog", \
                            "TargetList":" ","OpticalSystem": "Nemati", "ZodiacalLight": "Stark", "PostProcessing": " ", \
                            "Completeness": " ","BackgroundSources": "GalaxiesFaintStars", "PlanetPhysicalModel": " ", \
                            "PlanetPopulation": "KeplerLike1"}, "scienceInstruments": [{ "name": "imager"}],  \
                            "starlightSuppressionSystems": [{ "name": "HLC-565"}]   })
        
        sInds       = np.arange(0,fTL.nStars)
        ang         = self.star_angularSep(fTL, 0, sInds, tA) 
        sInd_sorted = np.argsort(ang)
        angles      = ang[sInd_sorted].to('deg').value
        
        self.lview = self.rc.load_balanced_view()
        
        async_res = []
        for j in range(nStars):
            filename = 'dmMap_'+str(int(j))+'_n'+str(int(nStars))+'T'+str(int(0))+'m'+str(int(m0*10))
            ar = self.lview.apply_async(fun,fTL,sInd_sorted[j],angles,tA,dtRange,m0,filename)  
            async_res.append(ar)
        
        ar= self.rc._asyncresult_from_jobs(async_res)
        while not ar.ready():
            ar.wait(60.)
            if ar.progress > 0:
                timeleft = ar.elapsed/ar.progress * (nStars - ar.progress)
                if timeleft > 3600.:
                    timeleftstr = "%2.2f hours"%(timeleft/3600.)
                elif timeleft > 60.:
                    timeleftstr = "%2.2f minutes"%(timeleft/60.)
                else:
                    timeleftstr = "%2.2f seconds"%timeleft
            else:
                timeleftstr = "who knows"
            
            print("%4i/%i tasks finished after %4i min. About %s to go." % (ar.progress, nStars, ar.elapsed/60, timeleftstr))
            sys.stdout.flush()
            
        print("Tasks complete.")
        sys.stdout.flush()
        res = [ar.get() for ar in async_res]
        
        return res