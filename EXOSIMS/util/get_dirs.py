import os
import requests

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
        if os.path.isdir(cachedir):
            cache_dir = cachedir
        else:
            try:
                os.mkdir(cachedir)
                cache_dir = cachedir
            except Exception:
                # use default here
                home = get_home_dir()
                path = os.path.join(home,'.EXOSIMS')
                if not os.path.isdir(path):
                    os.mkdir(path)
                cache_dir = os.path.join(path, 'cache')
                if not os.path.isdir(cache_dir):
                    os.mkdir(cache_dir)
    else:
        # use default here
        home = get_home_dir()
        path = os.path.join(home,'.EXOSIMS')
        if not os.path.isdir(path):
            os.mkdir(path)
        cache_dir = os.path.join(path, 'cache')
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)

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
    if not os.path.isdir(path):
        os.mkdir(path)
    downloads_dir = os.path.join(path, 'downloads')
    if not os.path.isdir(downloads_dir):
        os.mkdir(downloads_dir)

    return downloads_dir

def get_file_from_url(URL, filename):
    """
    Downloads a file from the given URL and saves to .EXOSIMS/downloads

    Args:
        URL (str):
            URL for file to download
        filename (str):
            name of file saved

    Returns:
        success (bool):
            Boolean for successful download
    """

    downloads_dir = get_downloads_dir()
    path = os.path.join(downloads_dir, filename)
    success = False

    r = requests.get(URL, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        if os.path.exists(path):
            success = True
    else:
        print('could not download file at {}'.format(URL))

    return success
