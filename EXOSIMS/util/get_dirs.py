import os

"""
The get_dirs utility module contains functions which set up and find the cache and download
folders for EXOSIMS. These folders are set up similar to Astropy where POSIX systems give:

/home/user/.EXOSIMS/cache
/home/user/.EXOSIMS/downloads

and Windows systems generally give:

C:/Users/User/.EXOSIMS/cache
C:/Users/User/.EXOSIMS/downloads
 
An additional function is given to download a file from a website and store in the 
downloads folder.
"""

def get_home_dir():
    """
    Finds the Home directory for the system.

    Returns:
        homedir (str):
            Path to Home directory
    """

    # POSIX system
    if os.name == 'posix':
        if 'HOME' in os.environ:
            homedir = os.environ['HOME']
        else:
            raise OSError('Could not find POSIX home directory')
    # Windows system
    elif os.name == 'nt':
        # msys shell
        if 'MSYSTEM' in os.environ and os.environ.get('HOME'):
            homedir = os.environ['HOME']
        # network home
        elif 'HOMESHARE' in os.environ:
            homedir = os.environ['HOMESHARE']
        # local home
        elif 'HOMEDRIVE' in os.environ and 'HOMEPATH' in os.environ:
            homedir = os.path.join(os.environ['HOMEDRIVE'],os.environ['HOMEPATH'])
        # user profile?
        elif 'USERPROFILE' in os.environ:
            homedir = os.path.join(os.environ['USERPROFILE'])
        # something else?
        else:
            try:
                import winreg as wreg
                shell_folders = r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
                key = wreg.OpenKey(wreg.HKEY_CURRENT_USER, shell_folders)
                homedir = wreg.QueryValueEx(key, 'Personal')[0]
                key.Close()
            except Exception:
                # try home before giving up
                if 'HOME' in os.environ:
                    homedir = os.environ['HOME']
                else:
                    raise OSError('Could not find Windows home directory')
    else:
        # some other platform? try HOME to see if it works
        if 'HOME' in os.environ:
            homedir = os.environ['HOME']
        else:
            raise OSError('Could not find home directory on your platform')

    return homedir

def get_cache_dir(cachedir):
    """
    Finds the EXOSIMS cache directory for the Json spec file.
    If cachedir is given, checks if it already exists, attempts to create
    the folder, reverts to default if unable.
    If cachedir is None, the default will be used.

    Returns:
        cache_dir (str):
            Path to EXOSIMS cache directory
    """

    if cachedir is not None:
        # if cachedir is already a directory and can be read from, written to, and executed
        if os.path.isdir(cachedir) and os.access(cachedir, os.R_OK|os.W_OK|os.X_OK):
            cache_dir = cachedir
        else:
            # try to add cachedir as a directory
            try:
                os.mkdir(cachedir)
                cache_dir = cachedir
            except Exception:
                print('Cannot write to cache directory specified: {}'.format(cachedir))
                print('Attempting to use default cache directory')
                # use default here
                home = get_home_dir()
                path = os.path.join(home,'.EXOSIMS')
                if not os.path.isdir(path) and os.access(home, os.R_OK|os.W_OK|os.X_OK):
                    os.mkdir(path)
                cache_dir = os.path.join(path, 'cache')
                if not os.path.isdir(cache_dir) and os.access(path, os.R_OK|os.W_OK|os.X_OK):
                    os.mkdir(cache_dir)
    else:
        # use default here
        home = get_home_dir()
        path = os.path.join(home,'.EXOSIMS')
        if not os.path.isdir(path) and os.access(home, os.R_OK|os.W_OK|os.X_OK):
            os.mkdir(path)
        cache_dir = os.path.join(path, 'cache')
        if not os.path.isdir(cache_dir) and os.access(path, os.R_OK|os.W_OK|os.X_OK):
            os.mkdir(cache_dir)

    # ensure everything worked out
    assert os.access(cache_dir, os.F_OK), "Cache directory {} does not exist".format(cache_dir)
    assert os.access(cache_dir, os.R_OK), "Cannot read from cache directory {}".format(cache_dir)
    assert os.access(cache_dir, os.W_OK), "Cannot write to cache directory {}".format(cache_dir)
    assert os.access(cache_dir, os.X_OK), "Cannot execute from cache directory {}".format(cache_dir)

    return cache_dir

def get_downloads_dir():
    """
    Finds the EXOSIMS downloads directory.

    Returns:
        downloads_dir (str):
            Path to EXOSIMS downloads directory
    """

    home = get_home_dir()
    path = os.path.join(home, '.EXOSIMS')
    # create .EXOSIMS directory if it does not already exist
    if not os.path.isdir(path) and os.access(home, os.R_OK|os.W_OK|os.X_OK):
        os.mkdir(path)
    downloads_dir = os.path.join(path, 'downloads')
    # create .EXOSIMS/downloads directory if it does not already exist
    if not os.path.isdir(downloads_dir) and os.access(path, os.R_OK|os.W_OK|os.X_OK):
        os.mkdir(downloads_dir)

    # ensure everything worked out
    assert os.access(downloads_dir, os.F_OK), "Downloads directory {} does not exist".format(downloads_dir)
    assert os.access(downloads_dir, os.R_OK), "Cannot read from downloads directory {}".format(downloads_dir)
    assert os.access(downloads_dir, os.W_OK), "Cannot write to downloads directory {}".format(downloads_dir)
    assert os.access(downloads_dir, os.X_OK), "Cannot execute from downloads directory {}".format(downloads_dir)

    return downloads_dir

def get_paths(qFile=None,specs=None,qFargs=None):
    """
    Technically, this function is used in 2 distinct separate places, at the top of runQueue and the top of SurveySimulation
    This function gets EXOSIMS paths
    In Priority Order
    1. Argument specified path (runQueue argument)
    2. Queue file specified path
    3. *.json specified path
    4. Environment Variable
    5. Current working directory

    -*Used by TimeKeeping to search for Observing Block Schedule Files
    -Used by runQueue to get Script Paths, specify run output dir, and runLog.csv location
    -All ENVIRONMENT set keys must contain the keyword 'EXOSIMS'
    
    Args:
        qFile (string) - 
        specs (dict) - fields from a json script
        qFargs (passed args) - arguments from the queue JSON file

    Returns:
        paths (dict) - dictionary containing paths to folders where each of these are located

    """
    pathNames = ['EXOSIMS_SCRIPTS_PATH', # folder location where script files are stored
    'EXOSIMS_OBSERVING_BLOCK_CSV_PATH', # folder location where Observing Block CSV files are saved
    'EXOSIMS_FIT_FILES_FOLDER_PATH', # folder location where fit files are stored
    'EXOSIMS_PLOT_OUTPUT_PATH', # folder location where plots are to be output
    'EXOSIMS_RUN_SAVE_PATH', # folder location where analyzed data is output
    'EXOSIMS_RUN_LOG_PATH', # folder location where runLog.csv is saved
    'EXOSIMS_QUEUE_FILE_PATH'] # full file path to queue file
    paths = dict()

    #### 1. Set current working directory for all paths
    for p in pathNames:
        paths[p] = os.getcwd()
    paths['EXOSIMS_RUN_LOG_PATH'] = get_cache_dir(None) # specify defauly for runLog.csv to be cache dir

    #### 2. Grab Environment Set Paths and overwrite
    for key in os.environ.keys():
        if 'EXOSIMS' in key:
            paths[key] = os.environment.get(key)

    #### 3. Use JSON script specified path
    if not specs == None:
        keysInSpecs = [key for key in specs['paths'].keys() if key in pathNames]
        for key in keysInSpecs:
            paths[key] = specs['paths'][key]

    #### 4. Use queue file script specified path
    if not qFile == None:
        keysInQFile = [key for key in qFile['paths'].keys() if key in pathNames]
        for key in keysInQFile:
            paths[key] = qFile['paths'][key]

    #### 5. Use argument specified path from runQueue specifications
    if not qFargs == None:
        keysPassedInRunQ = [key for key in qFargs.keys() if key in pathNames]
        for key in keysPassedInRunQ:
            paths[key] = qFargs[key]


    #add checks here

    return paths