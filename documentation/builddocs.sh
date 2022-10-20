#!/bin/bash
# Script to fully regenerate, compile and update html docs
# must be run directly from documentation directory

if [ ! -d "../EXOSIMS/Prototypes" ] || [ `basename $PWD` != "documentation" ] ; then
    echo "This script must be run from the documentation directory in the EXOSIMS parent directory."
    exit 1
fi

#generate args page
python buildargdoc.py

#sphinx-apidoc -f -o . ../EXOSIMS/
sphinx-apidoc -M -f -o . ../EXOSIMS/ ../EXOSIMS/util/runPostProcessing.py ../EXOSIMS/util/plotConvergencevsNumberofRuns.py ../EXOSIMS/util/plotTimeline.py ../EXOSIMS/util/evenlyDistributePointsOnSphere.py ../EXOSIMS/util/KeplerSTM_C/CyKeplerSTM_setup.py ../EXOSIMS/util/plotKeepoutMap.py ../EXOSIMS/util/depthOfSearch.py

rm modules.rst

make html
make html

exit 0


