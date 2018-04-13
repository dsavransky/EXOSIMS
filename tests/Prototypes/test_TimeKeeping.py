#!/usr/local/bin/python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>

r"""TimeKeeping module unit tests

Michael Turmon, JPL, Mar/Apr 2016
"""

import sys
import unittest
import StringIO
from collections import namedtuple
from EXOSIMS.Prototypes.TimeKeeping import TimeKeeping
from EXOSIMS.Prototypes.Observatory import Observatory
from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
import numpy as np
import astropy.units as u
from astropy.time import Time
import pdb

class TestTimeKeepingMethods(unittest.TestCase):
    r"""Test TimeKeeping class."""

    def setUp(self):
        # print '[setup] ',
        # do not instantiate it
        self.fixture = TimeKeeping

    def tearDown(self):
        pass

    def test_init(self):
        r"""Test of initialization and __init__.
        """
        tk = self.fixture()
        self.assertEqual(tk.currentTimeNorm.to(u.day).value, 0.0)
        self.assertEqual(type(tk._outspec), type({}))
        # check for presence of one class attribute
        self.assertGreater(tk.missionLife.value, 0.0)

    def test_str(self):
        r"""Test __str__ method, for full coverage."""
        tk = self.fixture()
        # replace stdout and keep a reference
        original_stdout = sys.stdout
        sys.stdout = StringIO.StringIO()
        # call __str__ method
        result = tk.__str__()
        # examine what was printed
        contents = sys.stdout.getvalue()
        self.assertEqual(type(contents), type(''))
        self.assertIn('currentTimeNorm', contents)
        sys.stdout.close()
        # it also returns a string, which is not necessary
        self.assertEqual(type(result), type(''))
        # put stdout back
        sys.stdout = original_stdout

    def test_allocate_time(self):
        r"""Test allocate_time method.

        Approach: Ensure ordinary allocations succeed, and that erroneous allocations
        fail.  Ensure allocations increase the mission time state variables correctly.
        """

        print 'allocate_time()'
        tk = self.fixture(OBduration=10.0)
        ## 1: Basic allocations

        # basic allocation: no time
        t0 = tk.currentTimeNorm.copy()
        tk.allocate_time(0.0*u.day)
        self.assertEqual(tk.currentTimeNorm,0.0*u.day)

        # basic allocation: a day
        tk.allocate_time(1.0*u.day)
        self.assertEqual(tk.currentTimeNorm-t0,1.0*u.day)

        # basic allocation: longer than timekeeping window
        # should put you at start of next block
        OBnum = tk.OBnumber
        tk.allocate_time(tk.OBduration + 1.0*u.day)
        self.assertEqual(OBnum+1,tk.OBnumber)
        self.assertEqual(tk.OBstartTimes[-1],tk.currentTimeNorm)

        ## 2: Consistency
        # do not allow to go beyond the first observation window
        #tk = self.fixture(missionPortion=1.0, missionLife=(2*n_test*scale).to(u.year).value)
        tk = self.fixture(OBduration=100.)
        n_test = 10
        scale = tk.OBduration / (n_test + 1) # scale of allocations [day]
        # series of (small) basic allocations
        for _ in range(n_test):
            t0n = tk.currentTimeNorm.copy()
            t0a = tk.currentTimeAbs.copy()
            dt = scale * np.random.rand()
            tk.allocate_time(dt)
            t1n = tk.currentTimeNorm
            t1a = tk.currentTimeAbs
            # print('timeDifference ' + str((t1n-t0n)))
            # print('dt             ' + str(dt))
            try:
                self.assertAlmostEqual((t1n - t0n).to(u.day).value, dt.to(u.day).value, places=8)
            except:
                self.assertAlmostEqual((t1n - t0n).to(u.day).value, (tk.OBstartTimes[tk.OBnumber]-t0n).to(u.day).value, places=8)
            try:
                self.assertAlmostEqual((t1a - t0a).to(u.day).value, dt.to(u.day).value, places=8)
            except:
                self.assertAlmostEqual((t1a - t0a).to(u.day).value, (tk.OBstartTimes[tk.OBnumber] + tk.missionStart - t0a).to(u.day).value, places=8)

    # def test_allocate_time_long(self):
    #     r"""Test allocate_time method (cumulative).

    #     Approach: Ensure mission allocated time is close to the duty cycle.
    #     """
    #     print 'allocate_time() cumulative'
    #     # go beyond observation window
    #     life = 6.0 * u.year
    #     ratio = 0.2
    #     tk = self.fixture(missionLife=life.to(u.year).value, missionPortion=ratio, OBduration=100.0)
        
    #     # allocate blocks of size dt until mission ends
    #     dt = 2.0 * u.day
    #     dt_all = 0 * dt
    #     while not tk.mission_is_over():
    #         tk.allocate_time(dt)
    #         dt_all += dt

    #     sumOB =sum([a - b for a, b in zip(tk.OBendTimes, tk.OBstartTimes)])
    #     #ratio_hat = (sumOB.to('day')/life.to('day')).value
    #     print(tk.OBstartTimes)
    #     ratio_hat = (tk.OBnumber*tk.OBduration.to('day')/life.to('day')).value
    #     print(ratio_hat)
    #     ratio_delta = (dt/life.to('day')).value * ratio
    #     self.assertAlmostEqual(ratio_hat, ratio, delta=ratio_delta)
    #     # # ensure mission allocated time is close enough to the duty cycle, "ratio"
    #     # ratio_hat = ((dt_all / life) + 0.0).value # ensures is dimensionless
    #     # # the maximum duty cycle error is the block-size (dt) divided by the total
    #     # # allocation window length (our instrument and the other instrument, or duration/ratio):
    #     # # dt / (duration / ratio) = (dt / duration) * ratio
    #     # ratio_delta = ((dt / tk.OBduration) + 0.0).value * ratio
    #     # self.assertAlmostEqual(ratio_hat, ratio, delta=ratio_delta)


    def test_mission_is_over(self):
        r"""Test mission_is_over method.

        Approach: Allocate time until mission completes.  Check that the mission terminated at
        the right time.
        """
        life = 0.1 * u.year
        tk = self.fixture(missionLife=life.to(u.year).value, missionPortion=1.0)
        Obs = Observatory(settlingTime=0.5)
        OS = OpticalSystem(ohTime=0.5)
        allModes = OS.observingModes
        det_mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        dt = 1 * u.day  # allocate blocks of size dt


        # 1) mission not over
        self.assertFalse(tk.mission_is_over(Obs, det_mode)) #the mission has just begun

        # 2) exoplanetObsTime exceeded
        tk.exoplanetObsTime = 1.1*tk.missionLife*tk.missionPortion # set exoplanetObsTime to failure condition
        self.assertTrue(tk.mission_is_over(Obs, det_mode))
        tk.exoplanetObsTime = 0.*tk.missionLife*tk.missionPortion # reset exoplanetObsTime

        # 3) missionLife exceeded
        tk.currentTimeNorm = 1.1*tk.missionLife
        tk.currentTimeAbs = tk.missionStart + 1.1*tk.missionLife
        self.assertTrue(tk.mission_is_over(Obs, det_mode))
        tk.currentTimeNorm = 0*u.d
        tk.currentTimeAbs = tk.missionStart

        # 4) OBendTimes Exceeded

        # 2) Allocate a single Day
        while not tk.mission_is_over(Obs, det_mode):
            success = tk.allocate_time(dt, Obs, det_mode, False)
        self.assertTrue(tk.mission_is_over(Obs, det_mode))
        # ensure mission terminates within delta/2 of its lifetime
        #   note: if missionPortion is not 1, the mission can end outside an observation window,
        #   and the current time will not be close to the lifetime.
        self.assertAlmostEqual(tk.currentTimeNorm.to(u.day).value, life.to(u.day).value, delta=dt.to(u.day).value/2)
        # allocate more time, and ensure mission is still over
        tk.allocate_time(dt)
        self.assertTrue(tk.mission_is_over(Obs, det_mode))

    def test_advancetToStartOfNextOB(self):
        r""" Test advancetToStartOfNextOB method
        """  
        life = 2.0*u.year
        obdur = 15
        missPor = 0.6
        tk = self.fixture(missionLife=life.to(u.year).value, OBduration=obdur, missionPortion=missPor)

        tNowNorm1 = tk.currentTimeNorm.copy()
        tNowAbs1 = tk.currentTimeAbs.copy()
        OBnumstart = tk.OBnumber #get initial OB number
        tStart1 = tk.OBstartTimes[tk.OBnumber].copy()
        tk.advancetToStartOfNextOB()
        OBnumend = tk.OBnumber
        tStart2 = tk.OBstartTimes[tk.OBnumber].copy()
        tNowNorm2 = tk.currentTimeNorm.copy()
        tNowAbs2 = tk.currentTimeAbs.copy()
        self.assertEqual(OBnumend-OBnumstart,1)#only one observation block has been incremented

        self.assertEqual((tStart2-tStart1).value,obdur/missPor)#The mission advances
        self.assertEqual((tNowNorm2-tNowNorm1).value,obdur/missPor)
        self.assertEqual((tNowAbs2-tNowAbs1).value,obdur/missPor)

    # def test_get_tStartNextOB(self):
    #     r"""Test get_tStartNextOB Method
    #     """
    #     life = 2.0*u.year
    #     obdur = 15
    #     missPor = 0.6
    #     tk = self.fixture(missionLife=life.to(u.year).value, OBduration=obdur, missionPortion=missPor)

    #     tStartNextOB = tk.get_tStartNextOB()
    #     tk.advancetToStartOfNextOB()
    #     self.assertEqual(tk.OBstartTimes[tk.OBnumber],tStartNextOB)

    # def test_get_tEndThisOB(self):
    #     """Test get_tEndThisOB
    #     """
    #     life = 2.0*u.year
    #     obdur = 15
    #     missPor = 0.6
    #     tk = self.fixture(missionLife=life.to(u.year).value, OBduration=obdur, missionPortion=missPor)
    #     self.assertEqual(tk.OBendTimes[tk.OBnumber],tk.get_tEndThisOB())

if __name__ == '__main__':
    unittest.main()
