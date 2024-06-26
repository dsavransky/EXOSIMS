"""
The get_dirs utility module contains functions which set up and find the cache and
download folders for EXOSIMS.

These folders are set up similar to Astropy where POSIX systems give:

/home/user/.EXOSIMS/cache
/home/user/.EXOSIMS/downloads

and Windows systems generally give:

C:/Users/User/.EXOSIMS/cache
C:/Users/User/.EXOSIMS/downloads

An additional function is given to download a file from a website and store in the
downloads folder.
"""

import os
from typing import Optional


def get_home_dir() -> str:
    """
    Finds the Home directory for the system.

    Returns:
        str:
            Path to Home directory
    """

    # POSIX system
    if os.name == "posix":
        if "HOME" in os.environ:
            homedir = os.environ["HOME"]
        else:
            raise OSError("Could not find POSIX home directory")
    # Windows system
    elif os.name == "nt":
        # msys shell
        if "MSYSTEM" in os.environ and os.environ.get("HOME"):
            homedir = os.environ["HOME"]
        # network home
        elif "HOMESHARE" in os.environ:
            homedir = os.environ["HOMESHARE"]
        # local home
        elif "HOMEDRIVE" in os.environ and "HOMEPATH" in os.environ:
            homedir = os.path.join(os.environ["HOMEDRIVE"], os.environ["HOMEPATH"])
        # user profile?
        elif "USERPROFILE" in os.environ:
            homedir = os.path.join(os.environ["USERPROFILE"])
        # something else?
        else:
            try:
                import winreg as wreg

                shell_folders = (
                    r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
                )
                key = wreg.OpenKey(wreg.HKEY_CURRENT_USER, shell_folders)
                homedir = wreg.QueryValueEx(key, "Personal")[0]
                key.Close()
            except Exception:
                # try home before giving up
                if "HOME" in os.environ:
                    homedir = os.environ["HOME"]
                else:
                    raise OSError("Could not find Windows home directory")
    else:
        # some other platform? try HOME to see if it works
        if "HOME" in os.environ:
            homedir = os.environ["HOME"]
        else:
            raise OSError("Could not find home directory on your platform")

    assert os.path.isdir(homedir) and os.access(homedir, os.R_OK | os.W_OK | os.X_OK), (
        f"Identified {homedir} as home directory, but it does not exist "
        "or is not accessible/writeable"
    )

    return homedir


def get_exosims_dir(dirtype: str, indir: Optional[str] = None) -> str:
    """
    Return path of EXOSIMS input/output directory.  Nominally this is either for the
    cache directory or the downloads directory, but others may be added in the future.

    Order of selection priority is:

    1. Input path (typically taken from JSON spec script)
    2. Environment variable (EXOSIMS_DIRTYPE_DIR)
    3. Default (nominally $HOME/.EXOSIMS/dirtype for whatever $HOME is returned by
       get_home_dir)

    In each case, the directory is checked for read/write/access permissions.  If
    any permissions are missing, will return default path. If default is still not
    useable, will throw AssertionError.

    Args:
        dirtype (str):
            Directory type (currently limited to 'cache' or 'downloads'
        indir (str):
            Full path (may include environment variables and other resolveable
            elements).  If set, will be tried first.

    Returns:
        str:
            Path to EXOSIMS directory specified by dirtype
    """

    assert dirtype in [
        "cache",
        "downloads",
    ], "Directory type must be 'cache' or 'downloads'"

    outdir = None  # try options until this is set

    # try input if given
    if indir is not None:
        # expand path
        indir = os.path.normpath(os.path.expandvars(indir))
        # if it doesn't exist, try creating it
        if not (os.path.isdir(indir)):
            try:
                os.makedirs(indir)
            except PermissionError:
                print("Cannot create directory: {}".format(indir))

        # if indir exists and has rwx permission, we're done
        if os.path.isdir(indir) and os.access(indir, os.R_OK | os.W_OK | os.X_OK):
            outdir = indir

    # if outidr has not yet been set, let's try looking for an environment var
    if outdir is None:
        envvar = "EXOSIMS_" + dirtype.upper() + "_DIR"
        if envvar in os.environ:
            envdir = os.path.normpath(os.path.expandvars(os.environ[envvar]))

            if not (os.path.isdir(envdir)):
                try:
                    os.makedirs(envdir)
                except PermissionError:
                    print("Cannot create directory: {}".format(envdir))

            # if envdir exists and has rwx permission, we're done
            if os.path.isdir(envdir) and os.access(envdir, os.R_OK | os.W_OK | os.X_OK):
                outdir = envdir

    # if you're here and outdir still not set, fall back to default
    if outdir is None:
        home = get_home_dir()
        path = os.path.join(home, ".EXOSIMS")
        if not os.path.isdir(path):
            try:
                os.makedirs(path)
            except PermissionError:
                print("Cannot create directory: {}".format(path))

        outdir = os.path.join(path, dirtype)
        if not os.path.isdir(outdir):
            try:
                os.makedirs(outdir)
            except PermissionError:
                print("Cannot create directory: {}".format(outdir))

    # ensure everything worked out
    assert os.access(outdir, os.F_OK), "Directory {} does not exist".format(outdir)
    assert os.access(outdir, os.R_OK), "Cannot read from directory {}".format(outdir)
    assert os.access(outdir, os.W_OK), "Cannot write to directory {}".format(outdir)
    assert os.access(outdir, os.X_OK), "Cannot execute directory {}".format(outdir)

    return outdir


def get_cache_dir(cachedir: Optional[str] = None) -> str:
    """
    Return EXOSIMS cache directory.  Order of priority is:
    1. Input (typically taken from JSON spec script)
    2. EXOSIMS_CACHE_DIR environment variable
    3. Default in $HOME/.EXOSIMS/cache (for whatever $HOME is returned by get_home_dir)

    In each case, the directory is checked for read/write/access permissions.  If
    any permissions are missing, will return default path.

    Returns:
        str:
            Path to EXOSIMS cache directory
    """

    cache_dir = get_exosims_dir("cache", cachedir)
    return cache_dir


def get_downloads_dir(downloadsdir: Optional[str] = None) -> str:
    """
    Return EXOSIMS downloads directory.  Order of priority is:

    1. Input (typically taken from JSON spec script)
    2. EXOSIMS_CACHE_DIR environment variable
    3. Default in $HOME/.EXOSIMS/downloads
       (for whatever $HOME is returned by get_home_dir)

    In each case, the directory is checked for read/write/access permissions.  If
    any permissions are missing, will return default path.


    Returns:
        str:
            Path to EXOSIMS downloads directory
    """

    downloads_dir = get_exosims_dir("downloads", downloadsdir)

    return downloads_dir


def get_paths(qFile=None, specs=None, qFargs=None):
    """
    This function gets EXOSIMS paths in priority order:

    #. Argument specified path (runQueue argument)
    #. Queue file specified path
    #. JSON input specified path
    #. Environment Variable
    #. Current working directory

    * Used by TimeKeeping to search for Observing Block Schedule Files
    * Used by runQueue to get Script Paths, specify run output dir, and runLog.csv
      location
    * All ENVIRONMENT set keys must contain the keyword 'EXOSIMS'

    Args:
        qFile (str):
            Queue file
        specs (dict):
            fields from a json script
        qFargs (dict):
            arguments from the queue JSON file

    Returns:
        dict:
            dictionary containing paths to folders where each of these are located

    """
    pathNames = [
        "EXOSIMS_SCRIPTS_PATH",  # folder for script files
        "EXOSIMS_OBSERVING_BLOCK_CSV_PATH",  # folder for Observing Block CSV files
        "EXOSIMS_FIT_FILES_FOLDER_PATH",  # folder for fit files
        "EXOSIMS_PLOT_OUTPUT_PATH",  # folder for plots to be output
        "EXOSIMS_RUN_SAVE_PATH",  # folder where analyzed data is output
        "EXOSIMS_RUN_LOG_PATH",  # folder where runLog.csv is saved
        "EXOSIMS_QUEUE_FILE_PATH",
    ]  # full file path to queue file
    paths = dict()

    # 1. Set current working directory for all paths
    for p in pathNames:
        paths[p] = os.getcwd()
    paths["EXOSIMS_RUN_LOG_PATH"] = get_cache_dir(
        None
    )  # specify defauly for runLog.csv to be cache dir

    # 2. Grab Environment Set Paths and overwrite
    for key in os.environ.keys():
        if "EXOSIMS" in key:
            paths[key] = os.environ.get(key)

    # 3. Use JSON script specified path
    if specs is not None:
        keysInSpecs = [key for key in specs["paths"].keys() if key in pathNames]
        for key in keysInSpecs:
            paths[key] = specs["paths"][key]

    # 4. Use queue file script specified path
    if qFile is not None:
        keysInQFile = [key for key in qFile["paths"].keys() if key in pathNames]
        for key in keysInQFile:
            paths[key] = qFile["paths"][key]

    # 5. Use argument specified path from runQueue specifications
    if qFargs is not None:
        keysPassedInRunQ = [key for key in qFargs.keys() if key in pathNames]
        for key in keysPassedInRunQ:
            paths[key] = qFargs[key]

    # add checks here

    return paths
