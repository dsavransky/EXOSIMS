#!/bin/bash
# Script to fully regenerate, compile and update html docs
# must be run directly from documentation directory

if [ ! -d "../EXOSIMS/Prototypes" ] || [ `basename $PWD` != "documentation" ] ; then
    echo "This script must be run from the documentation directory in the EXOSIMS parent directory."
    exit 1
fi

sphinx-apidoc -f -o . ../EXOSIMS/

rm modules.rst

make html
make html

rsync -uav --delete ./_build/html .

exit 0


