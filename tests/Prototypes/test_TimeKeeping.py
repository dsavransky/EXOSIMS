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
import numpy as np
import astropy.units as u
from astropy.time import Time

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
        tk = self.fixture()
        ## 1: Basic allocations
        # basic allocation: a day
        success = tk.allocate_time(1.0*u.day)
        self.assertTrue(success)
        # basic allocation: no time
        success = tk.allocate_time(0.0*u.day)
        self.assertTrue(success)
        # basic allocation: negative
        success = tk.allocate_time(-1.0*u.day)
        self.assertFalse(success)
        # basic allocation: longer than timekeeping window
        success = tk.allocate_time(tk.duration + 1.0*u.day)
        self.assertFalse(success)
        ## 2: Consistency
        # do not allow to go beyond the first observation window
        #tk = self.fixture(missionPortion=1.0, missionLife=(2*n_test*scale).to(u.year).value)
        tk = self.fixture()
        n_test = 100
        scale = tk.duration / (n_test + 1) # scale of allocations [day]
        # series of (small) basic allocations
        for _ in range(n_test):
            t0n = tk.currentTimeNorm
            t0a = tk.currentTimeAbs
            dt = scale * np.random.rand()
            success = tk.allocate_time(dt)
            self.assertTrue(success)
            t1n = tk.currentTimeNorm
            t1a = tk.currentTimeAbs
            self.assertAlmostEqual((t1n - t0n).to(u.day).value, dt.to(u.day).value, places=8)
            self.assertAlmostEqual((t1a - t0a).to(u.day).value, dt.to(u.day).value, places=8)


    def test_allocate_time_long(self):
        r"""Test allocate_time method (cumulative).

        Approach: Ensure mission allocated time is close to the duty cycle.
        """

        print 'allocate_time() cumulative'
        # go beyond observation window
        life = 6.0 * u.year
        ratio = 0.2
        tk = self.fixture(missionLife=life.to(u.year).value, missionPortion=ratio)
        # allocate blocks of size dt until mission ends
        dt = 2.0 * u.day
        dt_all = 0 * dt
        while not tk.mission_is_over():
            success = tk.allocate_time(dt)
            self.assertTrue(success)
            dt_all += dt
        # ensure mission allocated time is close enough to the duty cycle, "ratio"
        ratio_hat = ((dt_all / life) + 0.0).value # ensures is dimensionless
        # the maximum duty cycle error is the block-size (dt) divided by the total
        # allocation window length (our instrument and the other instrument, or duration/ratio):
        # dt / (duration / ratio) = (dt / duration) * ratio
        ratio_delta = ((dt / tk.duration) + 0.0).value * ratio
        self.assertAlmostEqual(ratio_hat, ratio, delta=ratio_delta)


    def test_mission_is_over(self):
        r"""Test mission_is_over method.

        Approach: Allocate time until mission completes.  Check that the mission terminated at
        the right time.
        """

        print 'mission_is_over()'
        life = 2.0 * u.year
        tk = self.fixture(missionLife=life.to(u.year).value, missionPortion=1.0)
        # allocate blocks of size dt
        dt = 1 * u.day
        # mission not over
        self.assertFalse(tk.mission_is_over())
        while not tk.mission_is_over():
            success = tk.allocate_time(dt)
            self.assertTrue(success)
        self.assertTrue(tk.mission_is_over())
        # ensure mission terminates within delta/2 of its lifetime
        #   note: if missionPortion is not 1, the mission can end outside an observation window,
        #   and the current time will not be close to the lifetime.
        self.assertAlmostEqual(tk.currentTimeNorm.to(u.day).value, life.to(u.day).value, delta=dt.to(u.day).value/2)
        # allocate more time, and ensure mission is still over
        success = tk.allocate_time(dt)
        self.assertTrue(success) # allocation beyond mission duration is allowed
        self.assertTrue(tk.mission_is_over())

    @unittest.skip("deprecated method")
    def test_update_times(self):
        pass

    @unittest.skip("deprecated method")
    def test_duty_cycle(self):
        pass

if __name__ == '__main__':
    unittest.main()
