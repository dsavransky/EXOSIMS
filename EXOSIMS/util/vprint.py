# -*- coding: utf-8 -*-


def vprint(verbose):
    """This function is equivalent to the python print function, with an extra boolean
    parameter that toggles the print when it is required.
    
    Args:
        verbose (bool):
            If True (default), the function will print the toprint string. 
            If False, the function won't print anything. 
    
    Return:
        f (function):
            The new print function with one argument, the string to be printed (toprint).
    
    """
    if verbose is True:
        def f(toprint):
            print(toprint)
    else:   
        def f(toprint):
            pass
    
    return f
