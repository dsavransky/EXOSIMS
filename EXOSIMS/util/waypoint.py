import numpy as np
import os.path
import astropy.units as u


def waypoint(comps, intTimes, duration, mpath, tofile):
    """Generates waypoint dictionary for MissionSim

    Args:
        comps (array):
            An array of completeness values for all stars
        intTimes (array):
            An array of predicted integration times for all stars
        duration (int):
            The length of time allowed for the waypoint calculation, defaults to 365
        mpath (string):
            The path to the directory to save a plot in.
        tofile (string):
            Name of the file containing a plot of total completeness over mission time,
            by default genWaypoint does not create this plot

    Returns:
        dict:
            Output dictionary containing the number of stars visited, the total
            completness achieved, and the amount of time spent integrating.

    """

    CbT = comps / intTimes
    sInds_sorted = np.argsort(CbT)[::-1]

    # run through sorted sInds until end of duration
    intTime_sum = 0 * u.d
    comp_sum = 0
    num_stars = 0
    comp_sums = []
    intTime_sums = []

    for sInd in sInds_sorted:
        if intTime_sum + intTimes[sInd] > duration * u.d:
            break

        intTime_sum += intTimes[sInd]
        comp_sum += comps[sInd]
        num_stars += 1
        comp_sums.append(comp_sum)
        intTime_sums.append(intTime_sum.value)

    # if a filename is specified, create a plot.
    if tofile is not None:  # pragma: no cover
        import matplotlib.pyplot as plt

        plt.scatter(intTime_sums, comp_sums, s=4, color="0.25")
        plt.ylabel("Total Completeness")
        plt.xlabel("Time (d)")
        plt.title("Total Completeness Over {} Star Visits".format(num_stars))
        plt.grid(True)
        plt.savefig(os.path.join(mpath, tofile))

    return {
        "numStars": num_stars,
        "Total Completeness": comp_sum,
        "Total intTime": intTime_sum,
    }
