# -*- coding: utf-8 -*-


class vprint:
    """This function is equivalent to the python print function, with an extra boolean
    parameter that toggles the print when it is required.

    Args:
        verbose (bool):
            If True (default), the function will print the toprint string.
            If False, the function won't print anything.

    Return:
        f (function):
            The new print function with one argument, the string to be printed (toprint)

    """

    def __init__(self, verbose):
        self.verbose = verbose
        # if verbose:
        #     self.verbose = True
        # else:

    def __call__(self, toprint):
        if self.verbose:
            print(toprint)

    # if verbose is True:
    #     def f(toprint):
    #         print(toprint)
    # else:
    #     def f(toprint):
    #         pass
    # return f
