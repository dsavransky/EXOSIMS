from EXOSIMS.Prototypes.TargetList import TargetList
import numpy as np
import pandas as pd 
import astropy
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord

class PlandbTargetList(TargetList):
    """Target List based on the planet population from the plandb database.
    Should be used with the PlandbPlanets family of modules (e.g. PlandbUniverse)"""

    def __init__(self,**specs):
        #Mapping attribute name from the data columns
        #data columns are loaded from the pickle file in the PlanetPopulation module (PlandbPlanets)
        self.atts_mapping = {"Name" : "hostname",
                             "Spec" : "st_nspec",
                             "parx" : "sy_plx",
                             "Umag" : "sy_umag",
                             "Bmag" : "sy_bmag",
                             "Vmag" : "sy_vmag",
                             "Rmag" : "sy_rmag",
                             "Imag" : "sy_imag",
                             "Jmag" : "sy_jmag",
                             "Hmag" : "sy_hmag",
                             "Kmag" : "sy_kmag",
                             "dist" : "sy_dist",
                             "L" : "st_lum",
                             "pmra" : "sy_pmra",
                             "pmdec" : "sy_pmdec",
                             "rv" : "st_radv"}

        TargetList.__init__(self,**specs)

    def populate_target_list(self,**specs):

        PPop = self.PlanetPopulation
        Comp = self.Completeness
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight

        tmp = PPop.allplanetdata

        #filtering out targets (based on Working Angle)
        dist = tmp["sy_dist"].values*u.pc 
        

        mask = ~tmp["sy_dist"].isna() \
                & (np.arctan(PPop.sma*(1 + PPop.eccen)/dist) > OS.IWA) \
                & (np.arctan(PPop.sma*(1 - PPop.eccen)/dist) < OS.OWA)
        tmp = tmp[mask]

        #filtering out redundant targets
        targetStars = np.unique(tmp["hostname"].values, return_index=True)[1]
        tmp = tmp.iloc[targetStars]

        #filter missing Vmag and BV, for integration time calculation
        tmp = tmp[~tmp["sy_vmag"].isna()]
        BV = tmp["sy_bmag"] - tmp["sy_vmag"]
        tmp = tmp[~BV.isna()]
        
        #number of Targets
        self.nStars = len(tmp)
        assert self.nStars, "Target List is empty: nStars = %r"%self.nStars

        #assigning attributes to the target list
        for att in self.atts_mapping:
            att_val = tmp[self.atts_mapping[att]].values
            if type(att_val) == np.float64:
                setattr(self,att,att_val)
            else:
                if (att == "Name") or (att == "Spec"):
                    setattr(self,att,att_val.astype(str))
                else:
                    setattr(self,att,att_val)

        #astropy units
        self.parx = self.parx*u.mas
        self.dist = self.dist*u.pc
        self.pmra = self.pmra*u.mas/u.yr
        self.pmdec = self.pmdec*u.mas/u.yr
        self.rv = self.rv*u.km/u.s

        self.BC = -2.5*self.L - 26.832 - self.Vmag
        self.L = 10.**self.L
        self.MV = self.Vmag - 5*(np.log10(self.dist.to('pc').value) - 1)
        self.coords = SkyCoord(ra=tmp["ra"].values*u.deg, dec=tmp["dec"].values*u.deg,
                distance=self.dist)
        self.Binary_Cut = np.zeros(self.nStars, dtype=bool)

        #populate completeness values
        self.comp0 = Comp.target_completeness(self)

        #populate minimum integration time values
        self.tint0 = OS.calc_minintTime(self)

        #calculate 'true' and 'approximate' stellar masses
        self.stellar_mass()

        #include new attributes to the target list catalog attributes
        self.catalog_atts.append('comp0')
        self.catalog_atts.append('tint0')

    def filter_target_list(self,**specs):
        """dummy function (for further filtering)"""

        pass
