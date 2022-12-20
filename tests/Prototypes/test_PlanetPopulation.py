r"""Test code for PlanetPopulation Prototype module within EXOSIMS.

Cate Liu, IPAC, 2016"""

import unittest
from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import numpy as np
import scipy.stats


class TestPlanetPopulation(unittest.TestCase):
    def setUp(self):

        self.spec = {"modules": {"PlanetPhysicalModel": "PlanetPhysicalModel"}}
        self.kscrit = 0.01
        self.nsamp = 10000

    def tearDown(self):
        pass

    def test_gen_angles(self):
        """
        Test generation of orientation angles.

        We expect long. and periapse to be uniformly distributed and
        inclination to be sinusoidally distributed.

        Edit made by Sonny Rappaport, Cornell, July 2021:
        SciPY update has broken this method, so use KS test to check inclination
        distribution and alter usage of chi^2 test for the uniform distributions
        """

        pp = PlanetPopulation(**self.spec)
        I, O, w = pp.gen_angles(self.nsamp)

        # O & w are expected to be uniform
        for j, (param, param_range) in enumerate(zip([O, w], [pp.Orange, pp.wrange])):
            pval = scipy.stats.kstest(
                param.value, scipy.stats.uniform.cdf, args=tuple(param_range.value)
            ).pvalue
            if pval < self.kscrit:
                _, param, param = pp.gen_angles(self.nsamp)
                pval = scipy.stats.kstest(
                    param.value, scipy.stats.uniform.cdf, args=tuple(param_range.value)
                ).pvalue

            self.assertGreater(
                pval,
                self.kscrit,
                "{} does not appear uniform.".format(["Omega", "omega"][j]),
            )

        # cdf of the sin distribution for ks test
        sin_cdf = lambda x: (-np.cos(x) / 2 + 0.5)

        pval = scipy.stats.kstest(I, sin_cdf).pvalue

        # allowed one do-over for noise
        if pval <= self.kscrit:
            I, _, _ = pp.gen_angles(self.nsamp)
            pval = scipy.stats.kstest(I, sin_cdf).pvalue

        self.assertGreater(pval, self.kscrit, "I does not appear sinusoidal")

    def test_gen_plan_params(self):
        """
        Test generation of planet orbital and phyiscal properties.

        We expect eccentricity and albedo to be uniformly distributed
        and sma and radius to be log-uniform

        Edit made by Sonny Rappaport, Cornell, July 2021:
        SciPY update has broken this method, so use KS test to check log-uniform
        distribution and alter usage of chi^2 test for the uniform distributions
        """
        pp = PlanetPopulation(**self.spec)
        a, e, p, Rp = pp.gen_plan_params(self.nsamp)

        # expect e and p to be uniform
        for j, (param, param_range) in enumerate(zip([e, p], [pp.erange, pp.prange])):
            pval = scipy.stats.kstest(
                param,
                scipy.stats.uniform.cdf,
                args=(param_range[0], param_range[1] - param_range[0]),
            ).pvalue

            if pval <= self.kscrit:
                tmp = pp.gen_plan_params(self.nsamp)
                pval = scipy.stats.kstest(
                    tmp[j + 1],
                    scipy.stats.uniform.cdf,
                    args=(param_range[0], param_range[1] - param_range[0]),
                ).pvalue

            self.assertGreater(
                pval,
                self.kscrit,
                "{} does not appear uniform.".format(["eccentricity", "albedo"][j]),
            )

        # expect a and Rp to be log-uniform
        for j, (param, param_range) in enumerate(
            zip([a.value, Rp.value], [pp.arange.value, pp.Rprange.value])
        ):
            pval = scipy.stats.kstest(
                param, scipy.stats.loguniform.cdf, args=tuple(param_range)
            ).pvalue

            if pval < self.kscrit:
                a2, _, _, R2 = pp.gen_plan_params(self.nsamp)
                pval = scipy.stats.kstest(
                    [a2.value, R2.value][j],
                    scipy.stats.loguniform.cdf,
                    args=tuple(param_range),
                ).pvalue

            self.assertGreater(
                pval,
                self.kscrit,
                "{} does not appear log-uniform.".format(["sma", "planet radius"][j]),
            )

    def test_checkranges(self):
        """
        Test that check ranges is doing what it should do

        """

        pp = PlanetPopulation(arange=[10, 1], **self.spec)
        self.assertTrue(pp.arange[0].value == 1)
        self.assertTrue(pp.arange[1].value == 10)

        with self.assertRaises(AssertionError):
            pp = PlanetPopulation(prange=[-1, 1], **self.spec)

        with self.assertRaises(AssertionError):
            pp = PlanetPopulation(erange=[-1, 1], **self.spec)

        with self.assertRaises(AssertionError):
            pp = PlanetPopulation(arange=[0, 1], **self.spec)

        with self.assertRaises(AssertionError):
            pp = PlanetPopulation(Rprange=[0, 1], **self.spec)

        with self.assertRaises(AssertionError):
            pp = PlanetPopulation(Mprange=[0, 1], **self.spec)


if __name__ == "__main__":
    unittest.main()
