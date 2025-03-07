""" Finds the maximum consecutive number of true values in a boolean array
Written by: Dean Keithly
Written On: 7/7/2021
"""

import numpy as np


def maxConsecutiveTrue(arr):
    """Finds largest number of consecutive True booleans in the array
    Args:
        ndarray:
            arr - boolean array
    Returns:
        float:
            maxNum
    """
    maxNum = 0
    cumNum = 0
    for i in np.arange(len(arr)):
        if arr[i]:
            cumNum += 1
        else:
            cumNum = 0
        if cumNum > maxNum:
            maxNum = cumNum
    return maxNum
