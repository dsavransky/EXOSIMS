# -*- coding: utf-8 -*-
try:
    import collections.abc as collections  # for python 3
except ImportError:
    import collections  # for python 2
import functools


class memoize(object):
    """
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    Code taken directly from the PythonDecoratorLibrary:
        https://wiki.python.org/moin/PythonDecoratorLibrary

    Args:
        func (function):
            Function or instance method to memoize

    Attributes:
        func (function):
            The memoized function or instance method
        cache (dict):
            Dictionary with key-value pairs consisting of function arguments-
            function evaluations
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        """Call method

        Args:
            *args (list):
                Arguments for function

        """
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """
        Return the function's docstring.
        """

        return self.func.__doc__

    def __get__(self, obj, objtype):
        """
        Support instance methods.

        Args:
            obj (object):
                Object of interest
            objtype (type):
                Object's type

        """

        return functools.partial(self.__call__, obj)
