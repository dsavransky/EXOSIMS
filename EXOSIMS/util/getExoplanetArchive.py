import requests
import pandas  # type: ignore
import numpy as np
import os
from io import BytesIO
import glob
import time
from EXOSIMS.util.get_dirs import get_downloads_dir
from typing import Optional, Dict, Any
from requests.exceptions import ReadTimeout


def queryExoplanetArchive(
    querystring: str, filename: Optional[str] = None
) -> pandas.DataFrame:
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

    query = (
        """https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"""
        """query={}&format=csv"""
    ).format(querystring)

    r = requests.get(query)
    data = pandas.read_csv(BytesIO(r.content))

    if filename is not None:
        data.to_pickle(filename)

    return data


def getExoplanetArchivePS(
    forceNew: bool = False, **specs: Dict[Any, Any]
) -> pandas.DataFrame:
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


def getExoplanetArchivePSCP(forceNew: bool = False, **specs: Any) -> pandas.DataFrame:
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


def getExoplanetArchiveAliases(name: str) -> Optional[Dict[str, Any]]:
    """Query the exoplanet archive's system alias service and return results

    See: https://exoplanetarchive.ipac.caltech.edu/docs/sysaliases.html

    Args:
        name (str):
            Target name to resolve

    Returns:
        dict or None:
            Dictionary

    .. note::

        This has a tendency to get stuck when run in a loop. This is set up to
        fail after 10 seconds and retry once with a 30 second timeout.

    """

    query = (
        """https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Lookup/"""
        f"""nph-aliaslookup.py?objname={name}"""
    )

    try:
        r = requests.get(query, timeout=10)
    except ReadTimeout:
        r = requests.get(query, timeout=30)

    data = r.json()

    if data["manifest"]["lookup_status"] != "OK":
        return {}

    return data["system"]
