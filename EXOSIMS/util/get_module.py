# -*- coding: utf-8 -*-
import os.path

def get_module(name, folder = None):
    """Imports specific or Prototype class module
    
    Args:
        name (str):
            string containing desired class name
            OR full path to desired module.  
            The class name must be the same as the file name.
        folder (str):
            string containing folder housing desired class
            Equivalently the module type (i.e., Observatory)
    
    Returns:
        desiredmodule (object):
            desired module 
        
    """
    name = os.path.normpath(os.path.expandvars(os.path.expanduser(name)))
    if os.path.isfile(name) and name[-3:]=='.py':
        import imp
        path = name
        name = os.path.basename(os.path.split(path)[1][:-3])
        b = imp.load_source(name,path)
        desiredmodule = getattr(b, name)
        print 'Imported module %s from %s' % (name,path)
    else:
        import importlib
        if folder != None:
            package = 'EXOSIMS.'+folder+'.'+name
            try :
                b = importlib.import_module(package)
                desiredmodule = getattr(b, name)
                if folder=='Prototypes':
                    print 'Imported %s prototype module' % name
                else:
                    print 'Imported specific %s module %s' % (folder, name)
            except ImportError:
                package = 'EXOSIMS.Prototypes.'+folder
                b = importlib.import_module(package)
                desiredmodule = getattr(b, folder)
                print 'Imported %s prototype module' % folder
        
        # else, if the folder name is not given.         
        else:
            package = 'EXOSIMS.Prototypes.'+name
            try :
                b = importlib.import_module(package)
                desiredmodule = getattr(b, name)
                print 'Imported %s prototype module' % name
            except ImportError:
                import inspect
                paths = []
                count = 0
                root = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(get_module))))
                for path, subdirs, files in os.walk(root):
                    for filename in files:
                        if filename == name+'.py':
                            folder = os.path.basename(path)
                            paths.append(path)
                            count += 1
                if count == 1:
                    package = 'EXOSIMS.'+folder+'.'+name
                elif count > 1:
                    print 'Error: multiple modules with same name. Specify folder.', paths
                b = importlib.import_module(package)
                desiredmodule = getattr(b, name)
                print 'Imported specific %s module %s' % (folder, name)

    #validate what you've loaded
    assert hasattr(desiredmodule,'_modtype'),\
            "Module lacks attribute _modtype.  This is not a valid EXOSIMS class."
    if folder:
        assert (desiredmodule._modtype == folder), \
            "Module type (%s) does not match input (%s)."%(desiredmodule._modtype,folder)

    return desiredmodule
    
