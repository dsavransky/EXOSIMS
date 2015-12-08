# -*- coding: utf-8 -*-
import importlib

def get_module(name, folder = None):
    """Imports specific or Prototype class module
    
    Args:
        name (str):
            string containing desired class name
            OR full path to desired module
        folder (str):
            string containing folder housing desired class
    
    Returns:
        desiredmodule (object):
            desired module 
        
    """
    from os.path import isfile, normpath, basename
    if isfile(name) and name[-3:]=='.py':
        path = name
        name = basename(normpath(path[:-3]))
        folder = basename(normpath(path+'/..'))
    
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
            print 'Imported %s prototype module' % folder
            desiredmodule = getattr(b, folder)
            
    else:
        package = 'EXOSIMS.Prototypes.'+name
        try :
            b = importlib.import_module(package)
            desiredmodule = getattr(b, name)
            print 'Imported %s prototype module' % name
        except ImportError:
	    from os.path import abspath, dirname
            import os, inspect
            paths = []
            count = 0
            root = dirname(abspath(inspect.getfile(get_module)))+'/..'
            for path, subdirs, files in os.walk(root):
                for filename in files:
                    if filename == name+'.py':
                        folder = os.path.basename(path)
                        paths.append(path)
                        count += 1
            if count == 1:
                package = 'EXOSIMS.'+folder+'.'+name
                print 'Imported specific %s module %s' % (folder, name)
            elif count > 1:
                print 'Error: multiple modules with same name. Specify folder.', paths
            b = importlib.import_module(package)
            desiredmodule = getattr(b, name)

    return desiredmodule
    
