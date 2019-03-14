from EXOSIMS.PlanetPopulation.KeplerLike1 import KeplerLike1
from EXOSIMS.util.InverseTransformSampler import InverseTransformSampler
import astropy.units as u


class KeplerLike2(KeplerLike1):
    """
    Population based on Kepler radius distribution with RV-like semi-major axis
    distribution with exponential decay.
    
    NOTE: This is an exact clone of KeplerLike1, but uses (approximate)
    inverse transform sampling instead of simple rejection sampling for 
    performance improvements.
    
    Args: 
        \*\*specs:
            user specified values
            
    Attributes:
        smaknee (float):
            Location (in AU) of semi-major axis decay point (knee).
            Not an astropy quantity.
        esigma (float):
            Sigma value of Rayleigh distribution for eccentricity.
    
    Notes:
    1. The gen_mass function samples the Radius and calculates the mass from
    there.  Any user-set mass limits are ignored.
    2. The gen_albedo function samples the sma, and then calculates the albedos
    from there. Any user-set albedo limits are ignored.
    3. The Rprange is fixed to (1,22.6) R_Earth and cannot be overwritten by user
    settings (the JSON input will be ignored) 
    4. The radius piece-wise distribution provides the normalization required to
    get the proper overall eta.  The gen_radius method provided here normalizes
    in order to return exactly the number of samples requested.  A second method
    (gen_radius_nonorm) is provided for generating the simulated universe
    population. The latter assumes a poisson distribution for occurences in each
    bin.
    5.  Eccentricity is assumed to be Rayleigh distributed with a user-settable 
    sigma parameter (defaults to 0.25).
    
    """

    def __init__(self, smaknee=30, esigma=0.25, **specs):
        
        KeplerLike1.__init__(self, smaknee=smaknee, esigma=esigma, **specs)
        
        # unitless sma range
        ar = self.arange.to('AU').value
        self.sma_sampler = InverseTransformSampler(self.dist_sma, ar[0], ar[1])

    def gen_sma(self, n):
        """Generate semi-major axis values in AU
        
        Samples a power law distribution with exponential turn-off 
        determined by class attribute smaknee
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            a (astropy Quantity array):
                Semi-major axis in units of AU
        
        """
        n = self.gen_input_check(n)
        a = self.sma_sampler(n)*u.AU
        
        return a