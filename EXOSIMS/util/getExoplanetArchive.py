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


def cacheExoplanetArchiveQuery(
    basestr: str, querystring: str, forceNew: bool = False, **specs: Any
) -> pandas.DataFrame:
    """
    Look for cached query results, and return newest one.  If none exist, execute the
    query.

    Args:
        basestr (str):
            Base of the cache filename.
        querystring (str):
            Exact query string to use. See ``queryExoplanetArchive`` for details
        forceNew (bool):
            Run a fresh query even if results exist on disk.
        **specs (any):
            Any additional kewyords to pass to ``get_downloads_dir``

    Returns:
        pandas.DataFrame:
            Result of query
    """

    ddir = get_downloads_dir(**specs)

    # look for existing files and return newest
    if not (forceNew):
        files = glob.glob(os.path.join(ddir, f"{basestr}_*.pkl"))
        if files:
            files = np.sort(np.array(files))[-1]
            data = pandas.read_pickle(files)
            print(f"Loaded data from {files}")
            return data

    # if we're here, we need a fresh version
    filename = f"{basestr}_{time.strftime('%Y%m%d%H%M%S')}.pkl"
    filename = os.path.join(ddir, filename)

    return queryExoplanetArchive(querystring, filename=filename)


def getExoplanetArchivePS(
    forceNew: bool = False, **specs: Dict[Any, Any]
) -> pandas.DataFrame:
    """
    Get the contents of the Exoplanet Archive's Planetary Systems table and cache
    results.  If a previous query has been saved to disk, load that.

    Args:
        forceNew (bool):
            Run a fresh query even if results exist on disk. Defaults False.
        **specs (any):
            Any additional kewyords to pass to ``cacheExoplanetArchiveQuery``

    Returns:
        pandas.DataFrame:
            Planetary Systems table
    """

    basestr = "exoplanetArchivePS"
    querystring = r"select+*+from+ps"

    return cacheExoplanetArchiveQuery(basestr, querystring, forceNew=forceNew, **specs)


def getExoplanetArchivePSCP(forceNew: bool = False, **specs: Any) -> pandas.DataFrame:
    """
    Get the contents of the Exoplanet Archive's Planetary Systems Composite Parameters
    table and cache results.  If a previous query has been saved to disk, load that.

    Args:
        forceNew (bool):
            Run a fresh query even if results exist on disk.
        **specs (any):
            Any additional kewyords to pass to ``cacheExoplanetArchiveQuery``

    Returns:
        pandas.DataFrame:
            Planetary Systems composited parameters table
    """

    basestr = "exoplanetArchivePSCP"
    querystring = r"select+*+from+pscomppars"

    return cacheExoplanetArchiveQuery(basestr, querystring, forceNew=forceNew, **specs)


def getHWOStars(forceNew: bool = False, **specs: Any) -> pandas.DataFrame:
    """
    Get the contents of the ExEP HWO Star List and cache results.  If a previous query
    has been saved to disk, load that.

    Args:
        forceNew (bool):
            Run a fresh query even if results exist on disk.
        **specs (any):
            Any additional kewyords to pass to ``cacheExoplanetArchiveQuery``

    Returns:
        pandas.DataFrame:
            Planetary Systems composited parameters table

    See: https://exoplanetarchive.ipac.caltech.edu/docs/2645_NASA_ExEP_Target_List_HWO_Documentation_2023.pdf  # noqa: E501
    """

    basestr = "HWOStarList"
    querystring = r"select+*+from+di_stars_exep"

    return cacheExoplanetArchiveQuery(basestr, querystring, forceNew=forceNew, **specs)


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
