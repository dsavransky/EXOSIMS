import requests
import pandas
import numpy as np
import os
from io import BytesIO
import glob
import time
from EXOSIMS.util.get_dirs import get_downloads_dir


def queryExoplanetArchive(querystring, filename=None):
    """
    Query the exoplanet archive, optionally save results to disk, and return the
    result as a pandas dataframe.

    Args:
        querystring (str):
            Exact query string to use. Do not include format (csv will be specified).
            See: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
            for details. A valid string is: "select+*+from+pscomppars"
        filename (str):
            Full path to save file.  If None (default) results are not written to disk.
            Data will be written in pickle format.

    Returns:
        pandas.DataFrame:
            Result of query
    """

    query = """https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={}&format=csv""".format(
        querystring
    )

    r = requests.get(query)
    data = pandas.read_csv(BytesIO(r.content))

    if filename is not None:
        data.to_pickle(filename)

    return data


def getExoplanetArchivePS(forceNew=False, **specs):
    """
    Get the contents of the Exoplanet Archive's Planetary Systems table and cache
    results.  If a previous query has been saved to disk, load that.

    Args:
        forceNew (bool):
            Run a fresh query even if results exist on disk.

    Returns:
        pandas.DataFrame:
            Planetary Systems table
    """

    ddir = get_downloads_dir(**specs)

    # look for existing files and return newest
    if not (forceNew):
        files = glob.glob(os.path.join(ddir, "exoplanetArchivePS_*.pkl"))
        if files:
            files = np.sort(np.array(files))[-1]
            data = pandas.read_pickle(files)
            print("Loaded data from %s" % files)
            return data

    # if we're here, we need a fresh version
    filename = "exoplanetArchivePS_{}.pkl".format(time.strftime("%Y%m%d%H%M%S"))
    filename = os.path.join(ddir, filename)
    querystring = r"select+*+from+ps"

    return queryExoplanetArchive(querystring, filename=filename)


def getExoplanetArchivePSCP(forceNew=False, **specs):
    """
    Get the contents of the Exoplanet Archive's Planetary Systems Composite Parameters
    table and cache results.  If a previous query has been saved to disk, load that.

    Args:
        forceNew (bool):
            Run a fresh query even if results exist on disk.

    Returns:
        pandas.DataFrame:
            Planetary Systems composited parameters table
    """

    ddir = get_downloads_dir(**specs)

    # look for existing files and return newest
    if not (forceNew):
        files = glob.glob(os.path.join(ddir, "exoplanetArchivePSCP_*.pkl"))
        if files:
            files = np.sort(np.array(files))[-1]
            data = pandas.read_pickle(files)
            print("Loaded data from %s" % files)
            return data

    # if we're here, we need a fresh version
    filename = "exoplanetArchivePSCP_{}.pkl".format(time.strftime("%Y%m%d%H%M%S"))
    filename = os.path.join(ddir, filename)
    querystring = r"select+*+from+pscomppars"

    return queryExoplanetArchive(querystring, filename=filename)
