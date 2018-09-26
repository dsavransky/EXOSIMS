# -*- coding: utf-8 -*-

import os
from EXOSIMS.util.vprint import vprint

class fakeSingleRunAnalysis(object):
    """For testing runPostProcessing.py
    """
    _modtype = 'util'

    def __init__(self, args):
        vprint(args)
        vprint('fakeSingleRunAnalysis done')
        pass

    def singleRunPostProcessing(self, PPoutpath, folder):
        """This is called by runPostProcessing
        """
        pass