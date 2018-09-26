# -*- coding: utf-8 -*-
#fakeMultiRunAnalysis.py

import os
from EXOSIMS.util.vprint import vprint

class fakeMultiRunAnalysis(object):
    """For testing runPostProcessing.py
    """
    _modtype = 'util'

    def __init__(self, args):
        vprint(args)
        vprint('fakeMultiRunAnalysis done')
        pass

    def multiRunPostProcessing(self, PPoutpath, folder):
        """This is called by runPostProcessing
        """
        pass