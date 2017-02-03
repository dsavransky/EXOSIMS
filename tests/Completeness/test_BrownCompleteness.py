import os
import unittest
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import EXOSIMS
from EXOSIMS import MissionSim
from EXOSIMS.Completeness import BrownCompleteness
from EXOSIMS.Prototypes import TargetList
from tests.TestSupport.Info import resource_path

r"""BrownCompleteness module unit test

Currently the test here does not pass.

Paul Nunez, JPL, Aug. 2016
"""

# json "script" that allows creation of the EXOSIMS object for these tests
scriptfile = resource_path('test-scripts/template_completeness_testing.json')
# directory where output graphics are put
output_dir = resource_path('test-output')
# reference completeness
data = np.loadtxt(resource_path('Completeness/BrownCompleteness_EarthTwin_hist20160715.txt'))

class Test_BrownCompleteness(unittest.TestCase):
    # make plots interactive, or just save plots to files
    interactive = False

    def setUp(self):
        #scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','template_completeness_testing.json') 
        sim = MissionSim.MissionSim(scriptfile)
        self.targetlist = sim.TargetList
        #self.opticalsystem = sim.OpticalSystem
        #self.planetpop = sim.PlanetPopulation
        self.completeness = sim.Completeness
        print 'Loaded %d stars' % self.targetlist.nStars 

    def test_genplans(self):
        assert self.targetlist.nStars > 0
        # xedges is array of separation values for interpolant
        xd = np.array([0, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000, 1.8000, 1.9000, 2.0000,  2.1])
        yd = np.array([0, 20.0000, 21.3072, 22.5000, 23.8072, 25.0000, 26.3072, 27.5000, 28.8072, 30.0000, 31.3072])

        nplan = np.min([1e6,self.completeness.Nplanets])

        # Generate distribution of planets
        s, dMag = self.completeness.genplans(nplan)

        # get histogram
        h, yedges, xedges = np.histogram2d(dMag, s.to('AU').value, bins=(yd,xd))
        h = h / np.float(nplan)
        X, Y = np.meshgrid(xedges, yedges)

        # reshape data
        rhonda = data.T
        rhonda1 = rhonda[0:(len(yd)-1), 0:(len(xd)-1)]

        # wide figure to accommodate subplots
        plt.figure(1, figsize=(12,4))
        # panel 1: exosims completeness
        plt.subplot(131)
        plt.pcolor(xedges, yedges, h)
        plt.xlim(0,2)
        plt.ylim(20,30)
        plt.title('Completeness: EXOSIMS')
        plt.xlabel('Separation [AU]')
        plt.ylabel('Delta Magnitude')
        plt.colorbar()
        # panel 2: reference completeness
        plt.subplot(132)
        plt.pcolor(xd, yd, rhonda)
        plt.xlim(0,2)
        plt.ylim(20,30)
        plt.title("Completeness: Matlab Ref.")
        plt.xlabel('Separation [AU]')
        plt.ylabel('Delta Magnitude')
        plt.colorbar()
        # panel 3: the difference
        plt.subplot(133)
        plt.pcolor(xd, yd, rhonda1 - h)
        plt.xlim(0,2)
        plt.ylim(20,30)
        plt.title("Ref - EXOSIMS")
        plt.xlabel('Separation [AU]')
        plt.ylabel('Delta Magnitude')
        plt.colorbar()
        if self.interactive:
            plt.show()
        else:
            plt.savefig(os.path.join(output_dir, 'brown-completeness.png'))
            plt.savefig(os.path.join(output_dir, 'brown-completeness.pdf'))
            print 'Note: completeness plot placed in %s' % output_dir

        #This array quantifies the difference between exosim and rhonda's result.
        significance = np.array([[0]*(len(xd)-1)]*(len(yd)-1))
        for i in range(len(xd)-1):
            for j in range(len(yd)-1):                
                if rhonda1[j][i] > 0 or h[j][i] > 0:
                    significance[j][i] = nplan**0.5 * (rhonda1[j][i]-h[j][i]) / (10*rhonda1[j][i]+h[j][i])**0.5

        # Plot the significance
        plt.figure(2)
        plt.pcolor(xd, yd, significance)
        plt.xlim(0,2)
        plt.ylim(20,30)
        plt.title("Significance")
        plt.xlabel('Separation [AU]')
        plt.ylabel('Delta Magnitude')
        plt.colorbar()
        #plt.show()
        plt.title("Significance Histogram: Ref - EXOSIMS")
        plt.xlabel("Significance: (Ref - EXOSIMS) / sigma")
        plt.ylabel("Number of pixels")
        plt.hist((significance).flatten()[np.asarray(significance).flatten()!=0], bins=20)
        if self.interactive:
            plt.show()
        else:
            plt.savefig(os.path.join(output_dir, 'brown-significance.png'))
            plt.savefig(os.path.join(output_dir, 'brown-significance.pdf'))
            print 'Note: completeness comparison plot placed in %s' % output_dir

        #Create an array of 5 (sigma) to compare with significance
        five_sigma = np.array([[5.0]*(len(xd)-1)]*(len(yd)-1))

        # this is the core of the test
        #  -- as of 09/2016, it fails by a large margin
        np.testing.assert_array_less(significance, five_sigma)
        
        
if __name__ == '__main__':
    unittest.main()



