# EXOSIMS testing setup file
#
# Usage:
#   [1] Define the EXOSIMS_ROOT variable in your shell session, .bashrc, or equivalent
#   [2] Source this in your shell with:
#       $ source setup.sh
#
# Note: You have to source this file, because it modifies PYTHONPATH and PATH
#
# You should set the EXOSIMS_ROOT shell variable to the head of the EXOSIMS code
# you're currently using.  The file:
#
#   $EXOSIMS_ROOT/EXOSIMS/MissionSim.py
#
# should exist.

# Michael Turmon, JPL, May 2016


## Make a variable to point to EXOSIMS source code
if [ -z "$EXOSIMS_ROOT" ]; then
    echo "Error: Did not find EXOSIMS_ROOT.  Set the variable in the shell or in .bashrc" >&2
    exit 1
    # EXOSIMS_ROOT=/proj/exep/turmon/exomissionsim
fi
# sanity check
if [ ! -r "$EXOSIMS_ROOT/EXOSIMS/MissionSim.py" ]; then
    echo "Warning: EXOSIMS_ROOT does not point to valid EXOSIMS source." >&2
fi
    
## Python executable and utils
##   expect a python 2.7 with a few packages
if [ -d /Users ]; then
    # on MacOS
    py_prog=python
    path_add=
elif [ -d /proj/exep ]; then
    # section 383 servers
    py_prog=python2.7
    # this is where "coverage" tool lives
    path_add=/usr/local/python2/bin
else
    # default
    py_prog=python
    path_add=
fi

## Define this because it might be useful
EXOSIMS_TEST_ROOT=$(cd .. && pwd)

# but only export PYTHONPATH
export PYTHONPATH="${EXOSIMS_ROOT}:${EXOSIMS_TEST_ROOT}:${PYTHONPATH:-}"

# add to main PATH if needed
if [ -n "$path_add" ]; then
  # for coverage program
  PATH="${PATH}:$path_add"
fi

# suggest what to do
cat <<EOF

For unit tests:
  $py_prog -m unittest discover

For coverage:
  coverage run ./coverage_entry.py discover
EOF

# for hygiene, unset these, because this file is sourced
unset py_prog path_add
