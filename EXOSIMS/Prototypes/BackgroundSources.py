import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


class BackgroundSources(object):
    """BackgroundSources class prototype
    
    This module provides functionality to return the number density of background
    sources for a given target position and dark hole depth.

    Args:
        \*\*specs:
            user specified values

    Attributes: None

    """

    _modtype = "BackgroundSources"


    def __str__(self):
        """String representation of Background Sources module
        """

        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Background Sources class object attributes'

    def dNbackground(self, coords, intDepths):
        """Returns background source number densities
               
        Args:
            coords (array-like of SkyCoord):
                numpy ndarray or list of astropy SkyCoord objects representing
                the coordinates of one or more targets
            intDepths (array-like of Floats)
                numpy ndarray or list of floating point values representing 
                absolute magnitudes of the dark hole for each target.  
                Must be of same length as coords.
        Returns:
            dN (ndarray):
                Number density of background sources in number per square 
                arcminute.  Same length as inputs.
        """

        assert isinstance(intDepths,(tuple,list,np.ndarray)), \
                "intDepths is not array-like."
        if isinstance(coords,SkyCoord):
            assert coords.shape, "coords is not array-like."
        else:
            assert isinstance(coords,(tuple,list,np.ndarray)), \
                    "intDepths is not array-like."

        assert len(coords) == len(intDepths), "Input size mismatch."
        
        dN = np.zeros(len(intDepths))

        return dN
        
