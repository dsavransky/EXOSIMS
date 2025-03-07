import numpy as np
from scipy.interpolate import interp1d
import numbers
from EXOSIMS.util._numpy_compat import copy_if_needed


class InverseTransformSampler:
    """
    Approximate Inverse Transform Sampler for arbitrary distributions
    defined via a PDF encoded as a function (or lambda function)

    Args:
        f (function):
            Probability density function.  Must be able to operate
            on numpy ndarrays.  Function does not need to be
            normalized over the sampling interval.
        xMin (float):
            Minimum of interval to sample (inclusive).
        xMax (float):
            Maximum of interval to sample (inclusive).
        nints (int):
            Number of intervals to use in approximating CDF.
            Defaults to 10000

    Attributes:
        f, xMin, xMax
            As above


    Notes:
        If xMin == xMax, return values will all exactly equal xMin.
        To sample call the object with the desired number of samples.
    """

    def __init__(self, f, xMin, xMax, nints=10000):

        # validate inputs
        assert (
            isinstance(xMin, numbers.Number)
            and isinstance(xMax, numbers.Number)
            and isinstance(nints, numbers.Number)
        ), "xMin, xMax, and nints must be numbers."
        self.xMin = float(xMin)
        self.xMax = float(xMax)
        nints = int(nints)

        assert hasattr(f, "__call__"), "f must be callable."

        if self.xMin != self.xMax:
            ints = np.linspace(self.xMin, self.xMax, nints + 1)  # interval edges
            x = np.diff(ints) / 2.0 + ints[:-1]  # interval midpoints
            fX = f(x)
            if not isinstance(fX, np.ndarray):
                fX = np.array(fX, copy=copy_if_needed, ndmin=1)

            if len(fX) == 1:
                fX = float(fX) * np.ones(x.shape)
            F = np.hstack([0, np.cumsum(fX)])
            F /= F[-1]

            self.Finv = interp1d(F, ints)

    def __call__(self, numTest=1):
        """
        A call to the object with the number of samples will
        return the sampled distribution.
        """

        assert isinstance(numTest, numbers.Number), "numTest must be an integer."
        numTest = int(numTest)

        if self.xMin == self.xMax:
            return np.zeros(numTest) + self.xMin

        return self.Finv(np.random.uniform(size=numTest))
