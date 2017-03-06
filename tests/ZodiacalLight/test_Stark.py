r"""Test code for Stark module within EXOSIMS ZodiacalLight.

As of 10/2016, this test is failing.

Cate Liu, IPAC, 2016"""

import os
from os.path import join, dirname, abspath, isfile
import unittest
import xlrd
import numpy as np
import astropy.units as u
import EXOSIMS
from EXOSIMS import MissionSim
from EXOSIMS.ZodiacalLight.Stark import Stark
from tests.TestSupport.Info import resource_path


class TestStark(unittest.TestCase):
    def setUp(self):
        scriptfile = resource_path('test-scripts/ipac_testscript.json')
        #scriptfile = os.path.abspath(scriptfile)
        self.sim = MissionSim.MissionSim(scriptfile)
        self.spec = self.sim.genOutSpec()
        self.targetlist = self.sim.modules['TargetList']
        pass
    
    def tearDown(self):
        pass
    
    def test_fZ(self):
        obj = Stark()
        #print self.targetlist.Name[0:20]
        #print self.targetlist.coords[0:20]
        
        bn, bn1 = self.readXlxs()
        
        sind = [v for v in range(20)]
        lam = 0.5 * u.um
        r_sc = np.zeros((len(sind), 3)) * u.km
        
        res = obj.fZ(self.targetlist, sind, lam, r_sc)
        expected = bn
        #print res.value
        #print expected
        
        np.testing.assert_allclose(res.value, expected, rtol=1e-04, atol=0., verbose = True)
        
        sind = [v for v in range(20)]
        lam = 1.0 * u.um
        r_sc = np.zeros((len(sind), 3)) * u.km
        
        res = obj.fZ(self.targetlist, sind, lam, r_sc)
        expected = bn1
        print res.value
        print expected
        
        np.testing.assert_allclose(res.value, expected, rtol=1e-04, atol=0.)
        
    def readXlxs(self):
        """ read in Partick Lowrance's background_validation.xlsx file"""
        
        xlsxfile = resource_path('ZodiacalLight/StarkOutput.xlsx')
        xlsxfile = os.path.abspath(xlsxfile)
        
        if not isfile(xlsxfile):
            print "Error: Reference file not found: %s" % xlsxfile
               
        book = xlrd.open_workbook(xlsxfile)
        sheet = book.sheet_by_index(1)
                   
        ##determine the beginning row
        row0 = 5
        
        ##read in the Zodi brightness column for wl=0.5
        col = 6
        rown = 25
        bn = []
        for irow in range(row0, rown):
            crow = sheet.cell(irow, col).value
            bn.append(crow)
        
        ##read in the Zodi brightness column for wl=1.0
        row0= 26
        rown = 46
        col = 6
        bn1 = []
        for irow in range(row0, rown):
            crow = sheet.cell(irow, col).value
            bn1.append(crow)
            
        return bn, bn1
    
    
if __name__ == "__main__":
    unittest.main()

