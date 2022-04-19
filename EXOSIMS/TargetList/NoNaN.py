# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.io
from EXOSIMS.Prototypes.TargetList import TargetList
import re
import scipy.interpolate
import os.path
import inspect
import sys
import json
try:
    import cPickle as pickle
except:
    import pickle
try:
    import urllib2
except:
    import urllib
import pkg_resources
import sys

class NoNaN(TargetList):
    """Target list based on Gaia catalog inputs.
    
    Args:
        \*\*specs:
            user specified values
    
    """

    def __init__(self, **specs):

        TargetList.__init__(self, **specs)

    def populate_target_list(self, popStars=None, **specs):
        """ This function is actually responsible for populating values from the star
        catalog (or any other source) into the target list attributes.

        The prototype implementation does the following:
        
        Copy directly from star catalog and remove stars with any NaN attributes
        Calculate completeness and max integration time, and generates stellar masses.
        
        """
        
        SC = self.StarCatalog
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        Comp = self.Completeness
        
        # bring Star Catalog values to top level of Target List
        missingatts = []
        for att in self.catalog_atts:
            if not hasattr(SC,att):
                missingatts.append(att)
            else:
                if type(getattr(SC, att)) == np.ma.core.MaskedArray:
                    setattr(self, att, getattr(SC, att).filled(fill_value=float('nan')))
                else:
                    setattr(self, att, getattr(SC, att))
        for att in missingatts:
            self.catalog_atts.remove(att)
        
        # number of target stars
        self.nStars = len(self.Name)
        if self.explainFiltering:
            print("%d targets imported from star catalog."%self.nStars)

        if popStars is not None:
            tmp = np.arange(self.nStars)
            for n in popStars:
                tmp = tmp[self.Name != n ]

            self.revise_lists(tmp)

            if self.explainFiltering:
                print("%d targets remain after removing requested targets."%self.nStars)

#        if self.filterSubM:
#            self.subM_filter()
#
#        if self.fillPhotometry:
#            self.fillPhotometryVals()

        # filter out nan attribute values from Star Catalog
        self.nan_filter()
        if self.explainFiltering:
            print("%d targets remain after nan filtering."%self.nStars)

        # filter out target stars with 0 luminosity
#        self.zero_lum_filter()
#        if self.explainFiltering:
#            print("%d targets remain after removing requested targets."%self.nStars)

        if self.filter_for_char or self.earths_only:
            char_modes = list(filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes))
            # populate completeness values
            self.comp0 = Comp.target_completeness(self, calc_char_comp0=True)
            # Calculate intCutoff completeness
            self.comp_intCutoff = Comp.comp_per_intTime(OS.intCutoff, self, np.arange(self.nStars), ZL.fZ0, ZL.fEZ0, OS.WA0, char_modes[0])
            # populate minimum integration time values
            # self.tint0 = OS.calc_minintTime(self, use_char=True, mode=char_modes[0])
            for mode in char_modes[1:]:
                self.tint0 += OS.calc_minintTime(self, use_char=True, mode=mode)
        else:
            # populate completeness values
            self.comp0 = Comp.target_completeness(self)
            # Calculate intCutoff completeness
            char_modes = list(filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes))
            self.comp_intCutoff = Comp.comp_per_intTime(OS.intCutoff, self, np.arange(self.nStars), ZL.fZ, ZL.fEZ0, OS.WA0, char_modes[0])
            # populate minimum integration time values
            # self.tint0 = OS.calc_minintTime(self)

        # calculate 'true' and 'approximate' stellar masses
        self.vprint("Calculating target stellar masses.")
        self.stellar_mass()

        # Calculate Star System Inclinations
        self.I = self.gen_inclinations(self.PlanetPopulation.Irange)
        
        # include new attributes to the target list catalog attributes
        self.catalog_atts.append('comp0')
        # self.catalog_atts.append('tint0')

    def filter_target_list(self, **specs):
        self.nan_filter()
        if self.explainFiltering:
            print("%d targets remain after nan filtering."%self.nStars)
