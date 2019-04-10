# -*- coding: utf-8 -*-
"""
Template Plotting utility for post processing automation
Written by: Dean Keithly
Written on: 10/11/2018
"""

import os
from EXOSIMS.util.vprint import vprint

class fakeMultiRunAnalysis(object):
    """Template format for adding multiRunPostProcessing to any plotting utility
    multiRunPostProcessing method is a required method with the below inputs to work with runPostProcessing.py
    """
    _modtype = 'util'

    def __init__(self, args=None):
        vprint(args)
        vprint('fakeMultiRunAnalysis done')
        pass

    def multiRunPostProcessing(self, PPoutpath=None, folder=None):
        """This is called by runPostProcessing
        Args:
            PPoutpath (string) - output path to place data in
            folder (string) - full filepath to folder containing runs
        Return:
        """
        pass