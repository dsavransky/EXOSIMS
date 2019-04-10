# -*- coding: utf-8 -*-
import os.path
import inspect
import importlib
import pkgutil
import imp
import sys

# local indicator of verbosity: False is the typical non-debug setting
_verbose = False

#########################################
# 
# Helper functions
# 
#########################################

def modules_below_matching(pkg, name):
    r'''Return a list of modules, below the named package, matching a given name.

    Example usage: ::

        pkgs = modules_below_matching('EXOSIMS', 'Nemati')
    
    which would find the unique module ``EXOSIMS.OpticalSystem.Nemati``
    and return a length-1 list of that string.  It matches recursively,
    so intervening modules (in the above, ``OpticalSystem``) do not matter.
    '''
    
    # import the top-level package (e.g., "EXOSIMS")
    try:
        root_pkg = importlib.import_module(pkg)
    except ImportError:
        # mis-specification is a hard error
        sys.stderr.write('Error: Could not import root module "%s" within recursive search.\n' % pkg)
        raise
    # search below root_pkg
    prefix = root_pkg.__name__ + '.'
    modules = []
    for _, modname, is_pkg in pkgutil.walk_packages(root_pkg.__path__, prefix):
        # skip packages: they are one up from the level we want
        if is_pkg: continue
        if modname.endswith(name):
            modules.append(modname)
    return modules


def wildcard_expand(pattern):
    r'''Expand a pattern like pkg.*.module into a full package name like pkg.subpkg.module.

    The full package name, which is returned, must be unique, or an error is raised.
    Example usage: ::
    
        module = wildcard_expand('EXOSIMS.*.Nemati')
    
    which would find the unique module named ``EXOSIMS.OpticalSystem.Nemati``
    
    The returned value is a single string.
    '''
    # a.b.*.y.z -> a.b., .y.z
    head, tail = pattern.split('*')
    # a.b, y.z
    root_pkg, mod_name = head.rstrip('.'), tail.lstrip('.')
    # under a.b, find y.z
    sub_pkgs = modules_below_matching(root_pkg, mod_name)
    # it's an error if it's not unique
    #    Note: we do not suppress the error and try the next possibility because
    #    we want mis-specification to be a hard error
    if len(sub_pkgs) > 1:
        sys.stderr.write('Error: multiple modules with name matching "%s".  Specify folder.\n' % pattern)
        raise ValueError('Could not find unique "%s" in EXOSIMS.*' % pattern)
    if len(sub_pkgs) == 0:
        raise ValueError('Could not find any "%s" in EXOSIMS.*' % pattern)
    # return the unique match -- a string
    return sub_pkgs[0]


def get_module_chain(names):
    r"""Attempt to load a module from an ordered list of module names until one succeeds.
    
    Module names may be given fully as:
        EXOSIMS.OpticalSystem.Nemati
    or as wildcards like:
        EXOSIMS.*.Nemati

    Wildcards, if given, must match only one module.
    """
    for name in names:
        # replace name by its expansion if needed
        if '*' in name:
            name = wildcard_expand(name)
        try:
            full_module = importlib.import_module(name)
            #module_name = package
            #source = package
            return full_module
        except ImportError:
            continue
    return None
        

def get_module_in_package(name, folder):
    r"""Get an EXOSIMS module from a given package: handles most requests for modules.

    Return value is a Python package matching the given name."""

    # Helper function: make a qualified module name
    #    e.g., ('EXOSIMS', 'Prototypes', 'OpticalSystem') -> 'EXOSIMS.Prototypes.OpticalSystem'
    def make_module_name(*tup): return '.'.join(tup)

    # Method: for each case, define a list of names to search for.
    if '.' in name:
        # Case 3: module name is given as a qualified module name
        #    -- the name can be given as, e.g., Local.PlanetPopulation.MyPlanets, if "Local" is a package
        #       on sys.path.
        #    -- the name can also be given as, e.g., ".MyPlanets", and it is converted to just "MyPlanets"
        #       to allow loading from a "MyPlanets.py" module on sys.path
        #       i.e., the leading dot allows selection of this case, for very flat local module hierarchies,
        #       but the dot is removed before searching.
        if _verbose:
            print('get_module: case 3: attempting to load <%s>' % name)
        # kill leading ., if any
        module_names = [ name.lstrip('.') ]
    elif folder is not None:
        # Case 2a: folder given
        #    -- first: EXOSIMS.Prototypes.name
        #    -- fallback: EXOSIMS.folder.name
        if _verbose:
            print('get_module: case 2a: attempting to load <%s> from <%s>' % (name, folder))

        # load from Prototype, using asked-for module type, if name is empty or just blanks
        if len(name.strip(' ')) == 0:
            module_names = [
                make_module_name('EXOSIMS', 'Prototypes', folder)
                ]
        else:
            module_names = [
                make_module_name('EXOSIMS', folder, name),
                make_module_name('EXOSIMS', 'Prototypes', name)
                ]
    else:
        if _verbose:
            print('get_module: case 2b: attempting to load <%s>' % name)
        # Case 2b: folder NOT given
        #   -- first: EXOSIMS.Prototypes.name
        #   -- fallback: EXOSIMS.*.name
        module_names = [
            make_module_name('EXOSIMS', 'Prototypes', name),
            make_module_name('EXOSIMS', '*', name)
            ]
    # in all cases, use the list of module names to retrieve the first available module
    full_module = get_module_chain(module_names)
    if not full_module:
        # we can provide a good message now
        sys.stderr.write('Error: No module on paths: %s.\n' %
                         ', '.join(['"%s"' % n for n in module_names]))
    return full_module


def shorten_name(filename):
    r"""Produce a shortened version of a file or package path for one-line printing."""
    ellipsis = '[...]'
    (n_head, n_tail) = (15, 25)
    if len(filename) < n_head + n_tail + len(ellipsis):
        return filename
    else:
        return filename[:n_head] + ellipsis + filename[-n_tail:]

#########################################
# 
# Recommended functions for external callers are below
# 
#########################################

def get_module(name, folder = None):
    """Import specific or Prototype class module.
    
    There are three ways to use the name argument:
    
    Case 1: Applies when name ends in .py: it is interpreted as the name of
        a python source file implementing the stated module type.  For example,
        $HOME/EXOSIMS_local/MyObservatory.py which would be a module that
        implements a MyObservatory class.  Shell variables are expanded.
    Case 2: Default case.  The name is interpreted as an EXOSIMS module, which
        must be loaded.  If folder is given (best practice), we look first
        at EXOSIMS.folder.name for the module ("specific module"), and then
        at EXOSIMS.Prototypes.name ("prototype module").
        As a shortcut (provided that folder is given) if the name is empty or 
        all-blanks, we look at EXOSIMS.Prototypes.folder.
    Case 3: Used when a . (a Python module separator) is part of the name.  The
        name is interpreted as a loadable Python module, and is loaded.  This
        mechanism supports loading of modules from anywhere on the PYTHONPATH.
        For example, as in case 1, if $HOME/EXOSIMS_local/__init__.py exists
        (making EXOSIMS_local a valid Python package), and $HOME is on $PYTHONPATH,
        then giving name as EXOSIMS_local.MyObservatory will load the MyObservatory
        class within the given file.  This supports locally-customized module sets.

    Args:
        name (str):
            string containing desired class name (cases 2, 3 above)
            OR full path to desired module (case 1).  
            The class name must be the same as the file name.
        folder (str):
            The module type (e.g., Observatory or TimeKeeping), which is
            validated against the _modtype attribute of the loaded class.
            For specific modules, this is the same as the name of the
            folder housing the desired class.
    
    Returns:
        desired_module (object):
            module (class) that was requested 
    """

    # Divide into two top-level cases:
    #   1: load from python source file (.py ending), vs.
    #   2, 3: load module from a python package on sys.path (everything else)
    # Both branches below must set up these variables:
    #   -- full_module, as a package import
    #   -- source, indicating where the import came from (file or package name)
    #   -- note, as a string indicating the type of the source of the import
    if name.endswith('.py'):
        # Case 1: module name is given as a path
        if _verbose:
            print('get_module: Case 1: attempting to load <%s>' % name)
        # expand ~/..., $HOME/..., etc.
        path = os.path.normpath(os.path.expandvars(os.path.expanduser(name)))
        if not os.path.isfile(path):
            raise ValueError('Could not load module from path "%s".' % path)
        # the module is loaded under this name
        module_name_to_use = os.path.splitext(os.path.basename(path))[0]
        full_module = imp.load_source(module_name_to_use, path)
        source = path
        note = 'named file'
    else:
        # Cases 2, 3: module name is loadable from the module path (sys.path)
        full_module = get_module_in_package(name, folder)
        if not full_module:
            raise ValueError('Could not import module "%s" (path issue?).' % name)
        source = full_module.__name__
        note = 'prototype' if 'EXOSIMS.Prototypes' in full_module.__name__ else 'specific'
        
    # extract the tail end of the module name -- e.g., TimeKeeping, or Nemati
    module_name = full_module.__name__.split('.')[-1]

    # ensure that, if the module is named BetterTimeKeeping, the class within has this name also
    assert hasattr(full_module, module_name), \
            "Module name %s is incorrect.  This is not a valid EXOSIMS class." % (module_name)
    # extract the particular class from the full module's namespace
    desired_module = getattr(full_module, module_name)
    # ensure the extracted object is a class
    assert inspect.isclass(desired_module), \
      "Module contains an attribute %s but it is not a class." % module_name
    print('Imported %s (%s module) from %s' % (module_name, note, shorten_name(source)))
    # validate the _modtype property of the module we just loaded
    assert hasattr(desired_module, '_modtype'), \
            "Module lacks attribute _modtype.  This is not a valid EXOSIMS class."
    if folder:
        assert (desired_module._modtype == folder), \
            "Module type as loaded (%s) does not match request (%s)." % (desired_module._modtype, folder)

    return desired_module

def get_module_from_specs(specs, modtype):
    """Import specific or Prototype class module using specs dictionary.
    
    The universal idiom for initializing an EXOSIMS module follows the pattern: ::

        get_module(specs['modules']['TimeKeeping'], 'TimeKeeping')(**specs)
    
    Here, get_module loads the module, and the invocation with ``**specs`` runs its
    __init__ method.
    
    The present function abstracts the first half of the idiom by returning: ::

        get_module(specs['modules'][modtype], modtype)

    thus the above idiom may be replaced by: ::

        get_module_from_specs(specs, 'TimeKeeping')(**specs)

    which is shorter, and avoids the duplication of the module type.
    Note: we do not abstract the specs as well, because many callers might wish
    to modify the given ``**specs`` with keywords, or to separate getting the class
    from initializing it.
    
    Args:
        specs (dict):
            specs dictionary with a 'modules' key present.
        modtype (str):
            The module type (e.g., Observatory or TimeKeeping) of the 
            EXOSIMS module to fetch.
    
    Returns:
        desired_module (object):
            module (class) that was requested
        
    """
    return get_module(specs['modules'][modtype], modtype)


