# -*- coding: utf-8 -*-
import importlib

def get_module(name, folder):
    """Imports specific or Prototype class module
    
    Args:
        name (str):
            string containing desired class name
        folder (str):
            string containing folder housing desired class
    
    Returns:
        desiredmodule (object):
            desired module 
        
    """
                
    try:
        package = 'EXOSIMS.'+folder+'.'+name
        b = importlib.import_module(package)
        print 'Imported specific %s module %s' % (folder, name)
        desiredmodule = getattr(b, name)
    except ImportError:
        package = 'EXOSIMS.Prototypes.'+folder
        b = importlib.import_module(package)
        print 'Imported %s prototype module' % folder
        desiredmodule = getattr(b, folder)
        
    return desiredmodule