#!/usr/bin/env python
#

r"""Generally useful utilities for test suites

Michael Turmon, JPL, April/May 2016
"""

import sys
import os
import csv
from collections import defaultdict


class RedirectStreams(object):
    r"""Set stdout and stderr to redirect to the named streams.

    Used for eliminating chatter to stdout upon module creation."""

    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class assertMethodIsCalled(object):
    r"""Mock object to instrument calling of a named `method' within a given `obj'.

    Used in a with-clause.  Requires the named method, obj.method, to be called at
    least once.  Also: check the returned object's method_args attribute to see
    how the method was called.
    """

    def __init__(self, obj, method):
        self.obj = obj
        self.method = method
        self.method_args = []
        self.method_kwargs = []

    def wrapper(self, *args, **kwargs):
        self.method_called = True
        # print '** Calling', self.method, 'with', args
        self.method_args.append(args)
        self.method_kwargs.append(kwargs)
        return self.orig_method(*args, **kwargs)

    def __enter__(self):
        self.orig_method = getattr(self.obj, self.method)  # save method
        setattr(self.obj, self.method, self.wrapper)  # put in our wrapper
        self.method_called = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert getattr(self.obj, self.method) == self.wrapper, (
            "method %s was modified during assertMethodIsCalled" % self.method
        )
        # put method back
        setattr(self.obj, self.method, self.orig_method)
        # If an exception was thrown within the block, we've already failed.
        if traceback is None:
            assert self.method_called, "method %s of %s was not called" % (
                self.method,
                self.obj,
            )


# example of how to map key names to converter-functions for the csv loader below.
# each key in the dictionary is the
# column-name within the CSV file, and each value is a function mapping the
# string from the CSV file (which is typically a number, but not always)
# to a numerical value.  The value should be translated into the right units
# by this function.
#
# example for a few fields relevant to exoplanets --
example_unit_map = dict(
    pl_hostname=str,
    pl_orbeccen=float,
    pl_orbsmax=lambda x: float(x) * u.au,
    pl_bmasse=lambda x: (float(x) * const.M_earth),
)


def load_vo_csvfile(filename, unit_map):
    r"""Reads a CSV file and returns a two-level dict mapping hostnames + fields to values.

    Skips initial lines beginning with `#'.
    Map to a value using a construction like, for example:
       d['Wolf 1061'][1]['pl_orbeccen']
    This is the orbital eccentricity of a Wolf 1061 planet, the second one in the catalog.
    If the value was not given in the catalog, the value in the dictionary structure
    is set up as None.
    """

    # basic data structure: a dictionary containing lists (lists of further
    # dictionaries, to be precise).
    d = defaultdict(list)
    with open(filename, "r") as csvfile:
        # skip lines starting with #
        pos = csvfile.tell()
        while csvfile.readline().startswith("#"):
            pos = csvfile.tell()
        csvfile.seek(pos)
        # read the file sequentially by row
        reader = csv.DictReader(csvfile)
        for row in reader:
            # remap the each "value" v of row, which is a dict, through the appropriate
            # function in unit_map above -- but map empty strings to None
            row_remap = {k: (unit_map[k](v) if v else None) for (k, v) in row.items()}
            # Append an entry to the list held within one slot of "d",
            # using the "hostname" as key.
            d[row["pl_hostname"]].append(row_remap)
    return d


if __name__ == "__main__":
    pass
