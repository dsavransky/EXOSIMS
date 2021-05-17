from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import numpy as np
import astropy.units as u

class SolarSystem(PlanetPopulation):
    """
    Population of Earth-Like Planets from Brown 2005 paper
    
    This implementation is intended to enforce this population regardless
    of JSON inputs.  The only inputs that will not be disregarded are erange
    and constrainOrbits.
    """

    def __init__(self, commonSystemInclinations=True, **specs):
        #prange comes from nowhere
        #eta is probability of planet occurance in a system. I set this to 1
        specs['erange'] = erange
        specs['constrainOrbits'] = constrainOrbits
        #pE = 0.26 # From Brown 2005 #0.33 listed in paper but q=0.26 is listed in the paper in the figure
        # specs being modified in JupiterTwin
        specs['eta'] = eta
        specs['arange'] = arange #*u.AU
        specs['Rprange'] = [1.,1.] #*u.earthRad
        #specs['Mprange'] = [1*MpEtoJ,1*MpEtoJ]
        specs['prange'] = prange
        specs['scaleOrbits'] = True

        self.prange = prange

        PlanetPopulation.__init__(self, **specs)

    def gen_plan_params(self,nPlans):
        """ Values taken From mallama2018PlantProperties Seidlemenn 1992
        """

        #mercury,venus,earth,mars,jupiter,saturn,uranus,neptune
        R_orig = np.asarray([2439.7*1000.,6051.8*1000.,6371.0*1000.,3389.92*1000.,69911.*1000.,58232.*1000.,25362.*1000.,24622.*1000.])*u.m
        a_orig = np.asarray([57.91*10.**9.,108.21*10.**9.,149.60*10.**9.,227.92*10.**9.,778.57*10.**9.,1433.53*10.**9.,2872.46*10.**9.,4495.*10.**9.])*u.m
        p_orig = np.asarray([0.142,0.689,0.434,0.150,0.538,0.499,0.488,0.442])
        e_orig = np.zeros(8)

        #Tile them
        numTiles = int(nPlans/8)
        R_tiled = np.tile(R_orig,(numTiles))
        a_tiled = np.tile(a_orig,(numTiles))
        p_tiled = np.tile(p_orig,(numTiles))
        e_tiled = np.tile(e_orig,(numTiles))

        return a_tiled,e_tiled,p_tiled,Rp_tiled

    def gen_physical_properties(self, **specs):
        """Generating universe based on SAG13 planet radius and period sampling.
        
        All parameters except for albedo and mass are sampled, while those are
        calculated via the physical model.
        
        """
        
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        TL = self.TargetList
        
        
        nPlans = 8*TL.nStars #occurrence rate per system is fixed at 8
        self.nPlans = nPlans
        plan2star = np.ones(nPlans)*8
        self.plan2star = plan2star.astype(int)
        
        # sample all of the orbital and physical parameters
        self.I, self.O, self.w = PPop.gen_angles(self.nPlans)
        if self.commonSystemInclinations == True: #OVERWRITE I with TL.I+dI
            self.I = TL.I[self.plan2star]

        self.a, self.e, self.p, self.Rp = PPop.gen_plan_params(self.nPlans)

        self.gen_M0()                                    # initial mean anomaly
        self.Mp = self.gen_solar_system_planet_mass(self,nPlan)   # mass #TODO grab from Tables


    def gen_solar_system_planet_mass(self,nPlan):
        """ Generated planet masses for each planet
        """
        Mp_orig = np.asarray([3.3022*10**23,4.869*10**24,5.9742*10**24,6.4191*10**23,1.8988*10**27,5.685*10**26,8.6625*10**25,1.0278*10**26])*u.kg
        #Tile them
        numTiles = int(nPlans/8)
        Mp_tiled = np.tile(Mp_orig,(numTiles))
        return Mp_tiled
