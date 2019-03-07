# -*- coding: utf-8 -*-
"""
Makes Use of Daniel Garrett's Depth of Search Code
Written by: Dean Keithly
Written on: 1/27/2019
"""

import os
from EXOSIMS.util.vprint import vprint
import DoSFuncs
import datetime
import re

class depthOfSearch(object):
    """Template format for adding singleRunPostProcessing to any plotting utility
    singleRunPostProcessing method is a required method with the below inputs to work with runPostProcessing.py
    """
    _modtype = 'util'

    def __init__(self, args=None):
        vprint(args)
        vprint('depthOfSearch done')
        pass

    def singleRunPostProcessing(self, PPoutpath, folder):
        """This is called by runPostProcessing
        Args:
            PPoutpath (string) - output path to place data in
            folder (string) - full filepath to folder containing runs
        """
        #Loading Outspec
        outspecPath = os.path.join(folder,'outspec.json')

        #Done plotting Comp vs intTime of Observations
        date = unicode(datetime.datetime.now())
        date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
        fname1 = 'DoS_' + folder.split('/')[-1] + '_' + date
        #plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        #plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        #plt.savefig(os.path.join(PPoutpath, fname + '.eps'))

        # create DoSFuncs object with sample script and all default options
        dos = DoSFuncs.DoSFuncs(path=outspecPath)
        # plot depth of search
        dos.plot_dos('all', 'Depth of Search',os.path.join(PPoutpath, fname1))
        # plot expected planets
        fname2 = 'ExpectedPlanets_' + folder.split('/')[-1] + '_' + date
        dos.plot_nplan('all', 'Expected Planets',os.path.join(PPoutpath, fname2))

        pass