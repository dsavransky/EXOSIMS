#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>
# You may need to put EXOSIMS in your $PYTHONPATH, e.g.,
#   % PYTHONPATH=/path/to/exomissionsim <this_file.py>

r"""Support for testing.

Michael Turmon, JPL, Apr. 2016
"""

import os


def resource_path(p=()):
    r"""Return the path to shared testing resources.  A supplied string or tuple is appended.

    Arguments:
      p (string or tuple of strings):
        defaults to empty; if given, is appended to the testing-system resource path.
    Returns:
      rp: a string containing the path to a resource

    Usage:
      The following are equivalent, but the first is less desirable:
        rp = resource_path() + '/test-scripts/toy-catalog.json'
        rp = resource_path('test-scripts/toy-catalog.json')
        rp = resource_path(('test-scripts', 'toy-catalog.json'))
    """
    # map a string to a tuple containing the string to provide the obvious shortcut
    if isinstance(p, str):
        p = (p,)
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), *p)


def main():
    # might as well do something useful
    print("resource_path is", resource_path())


if __name__ == "__main__":
    main()
