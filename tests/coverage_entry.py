
r"""Coverage entry point."""

# Example usage:
# (1) Run code-under-test to generate coverage data:
#  $ coverage run ./coverage_entry.py discover 
# (2) Summarize data to html:
#  $ coverage html --include=$EXOSIMS_ROOT/EXOSIMS/*
# (3) Data is in .coverage, html report in htmlcov/index.html and below.
#
# Source: copied directly from python 2.7's unittest/__main__.py 
# because the coverage command-line script can't do -m,
# and we want to load the unittest module with 
# auto-discovery turned on.

import sys

if sys.argv[0].endswith("__main__.py"):
    sys.argv[0] = "python -m unittest"

__unittest = True

from unittest.main import main, TestProgram, USAGE_AS_MAIN
TestProgram.USAGE = USAGE_AS_MAIN

main(module=None)

