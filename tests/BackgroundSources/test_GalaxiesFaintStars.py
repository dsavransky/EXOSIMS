r"""Test code for GalaxiesFaintStars module within EXOSIMS BackgroundSources.

Status: Most of these tests are failing.

Cate Liu, IPAC, 2016"""

import os
from os.path import join, dirname, abspath, isfile
import random
import math
import xlrd
import unittest
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactic
import EXOSIMS
from EXOSIMS.BackgroundSources.GalaxiesFaintStars import GalaxiesFaintStars
from tests.TestSupport.Info import resource_path

class TestDnBackground(unittest.TestCase):
    def setUp(self):
        self.mags=np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        self.glon=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        pass
        
    def tearDown(self):
        #del self.mags
        pass
        
    def test_dNbackground1(self):
        r"""Test dN against pre-computed version.

        Method: Check against pre-computed values."""
        obj = GalaxiesFaintStars()
        mags = np.array([28., 28., 29., 24., 30.])
        try:
            coords= SkyCoord(frame=Galactic,
                             l=[0., 157.9, 15., 222., 312.],
                             b=[-30., 15., 45., 60., -28.],
                             unit="deg")
        except:
            print "Error in construcing SkyCoord!"
            raise Exception("Error in construcing SkyCoord!")
            
        dn = obj.dNbackground(coords, mags)
        expected_dn = np.array([61.7095, 112.746, 117.198, 4.46323, 248.726])
        # TODO: the dNbackground method only works for mags between 15 and 25.
        ok = np.logical_and(mags >= 15, mags <= 25)
        np.testing.assert_allclose(dn[ok].value, expected_dn[ok], rtol=1e-07, atol=1e-03)
        
    #@unittest.skip('Test raises exception because magnitudes are outside of [15,25]')
    def test_dNbackground2(self):
        obj = GalaxiesFaintStars()
        
        mags = np.array([28., 28., 28., 28., 28., 28., 28., 28., 28., 28., 28., 28., 28., 28., 28.])
        glon = [0., 157.9, 15., 222., -312., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        glat = [-2., -5., -10., -16., -45., -60.,-30., 0., 2., 5., 10., 16., 45., 60., -28.]
        try:
            coords= SkyCoord(frame=Galactic, l=glon, b=glat, unit="deg")
        except:
            print "Error in construcing SkyCoord!"
            raise Exception("Error in construcing SkyCoord!")
            
        dn = obj.dNbackground(coords, mags)
        expected_dn = np.array([235.43525751,  158.52427624,  112.74620139,   70.42834286,   56.86994772,
                                 56.86994772,   61.70957726,  235.43525751,  235.43525751,  158.52427624,   
                                112.74620139,   70.42834286,   56.86994772,   56.86994772,   61.70957726])
        # TODO: the dNbackground method only works for mags between 15 and 25.
        # FIXME: this, therefore, strips out the whole test
        ok = np.logical_and(mags >= 15, mags <= 25)
        np.testing.assert_allclose(dn.value, expected_dn, rtol=1e-07, atol=1e-03)

    def test_dNbackground3(self):
        obj=GalaxiesFaintStars()
        
        ## read in Patrick's validation numbers
        [glats, mags, tot_counts] = self.readXlxs()

        ##make up glon
        glon = random.sample(xrange(0, 360), len(glats))

        try:
            coords= SkyCoord(frame=Galactic, l=glon, b=glats, unit="deg")
        except:
            print "Error in construcing SkyCoord!"
            raise Exception("Error in construcing SkyCoord!")
        
        for irow in range(len(tot_counts)):
            if mags[irow] < 15 or mags[irow] > 25:
                continue
            cmags = np.empty(len(glats))
            cmags.fill(mags[irow])
            expected_dn0 = tot_counts[irow]
            dn = obj.dNbackground(coords, cmags)
            np.testing.assert_allclose(dn.value, expected_dn0, rtol=1e-07, atol=1e-03)
    
    def test_dNbackground4(self):
        """
        This test grid is from Rhonda Morgan via Patrick Lowrance:
        glats=[15, 75, -1, -75, -15, 47, 1, -30, 6, 81, 21, -81, -21, 32, -6, -2.5, 19, 56, 28, -56, -28, 2.4, -19]
        """
        obj=GalaxiesFaintStars()
        
        ## read in Patrick's validation numbers
        [glats, mags, tot_counts] = self.readXlxs()
        
        rmglats = [15, 75, -1, -75, -15, 47, 1, -30, 6, 81, 21, -81, -21, 32, -6, -2.5, 19, 56, 28, -56, -28, 2.4, -19]
        rmglons = random.sample(xrange(0, 360), len(rmglats))
        
        for (i, glat) in enumerate(rmglats):
            try:
                coords= SkyCoord(frame=Galactic, l=[rmglons[i]], b=[glat], unit="deg")
            except:
                print "Error in construcing SkyCoord!"
                raise Exception("Error in construcing SkyCoord!")
            
            ##determine which total counts nubmer to use from Patrick's table
            ind = 0
            lat = math.fabs(glat)
            if lat >= 75. and lat <= 90.:
                ind = 6
            elif lat < 75. and lat >= 45.:
                ind = 5
            elif lat < 45. and lat > 25.:
                ind = 4
            elif lat <= 25. and lat > 15.:
                ind = 3
            elif lat <= 15. and lat > 7.:
                ind = 2
            elif lat <= 7. and lat > 3.:
                ind = 1
            elif lat <= 3. and lat >= 0.:
                ind = 0
            
            for (j, mag) in enumerate(mags):
                # dNbackground only works for mags in [15,25]
                if mag < 15 or mag > 25:
                    continue
                cmags = np.empty(1)
                cmags.fill(mag)
                expected_dn0 = tot_counts[j][ind]
                dn = obj.dNbackground(coords, cmags)
                np.testing.assert_allclose(dn.value, expected_dn0, rtol=1e-07, atol=1e-03)
        
    def readXlxs(self):
        """ read in Partick Lowrance's background_validation.xlsl file"""
        xlsxfile = resource_path('BackgroundSources/background_validation.xlsx')
        #xlsxfile = os.path.join(EXOSIMS.__path__[0],'../test','background_validation.xlsx')
        #xlsxfile = os.path.abspath(xlsxfile)
        #print xlsxfile
        
        if not isfile(xlsxfile):
            print "Error: File doesn't exist: %s" % xlsxfile
               
        book = xlrd.open_workbook(xlsxfile)
        sheet = book.sheet_by_index(0)
                   
        ##determine the start row for 'Total counts'
        row0 = 0
        for irow in range(sheet.nrows):
            cell00 = sheet.cell(irow, 0).value
            if (cell00 == 'Total counts'):
                row0 = irow + 1  #the starting row for the total counts table
                break
        
        ##get headers (glats/mags)
        # Cell Types: 0=Empty, 1=Text, 2=Number, 3=Date, 4=Boolean, 5=Error, 6=Blank
        glats = [sheet.cell(row0, col_index) for col_index in xrange(1, sheet.ncols)]
        glats = [g.value for g in glats if not g.ctype ==0]  ##empty columns should be the trainling ones, or we have a problem!
        mags = [sheet.cell(row_index, 0).value for row_index in xrange(row0+1, sheet.nrows)]
        
        ##read in the actual total counts
        tot_counts = []
        for irow in range(row0+1, sheet.nrows):
            crow = [sheet.cell(irow, ind).value for ind in xrange(1, len(glats)+1)]
            tot_counts.append(crow)
        
        return (glats, mags, tot_counts)
        
if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestDnBackground)
    #unittest.TextTestRunner().run(suite)
    unittest.main()
    
