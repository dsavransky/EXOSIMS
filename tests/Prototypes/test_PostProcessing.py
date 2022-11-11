import unittest
import numpy as np
import os
import EXOSIMS
from EXOSIMS.Prototypes import PostProcessing
import numpy as np

r"""PostProcessing module unit tests

Paul Nunez, JPL, Aug. 2016
"""

# need a dummy BackgroundSources
specs = {"modules": {"BackgroundSources": " "}}


class Test_PostProcessing_prototype(unittest.TestCase):
    def setUp(self):
        self.TL = {}
        self.mode = {"SNR": 5.0}

    def test_nontrivialFAP(self):
        obj = PostProcessing.PostProcessing(FAP=0.1, MDP=0.0, **specs)

        # Test a case for which the false alarm prob is 0.1
        FAs = np.zeros(1000, dtype=bool)
        for j in np.arange(len(FAs)):
            FA, _ = obj.det_occur(np.array([5]), self.mode, self.TL, 0, 0)
            FAs[j] = FA
        # ~0.3% of the time this test should fail! due to random number gen.
        np.testing.assert_allclose(
            len(FAs) * obj.FAP, len(np.where(FAs)[0]), rtol=0.3, atol=0.0
        )


if __name__ == "__main__":
    unittest.main()
