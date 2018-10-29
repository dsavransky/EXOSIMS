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
