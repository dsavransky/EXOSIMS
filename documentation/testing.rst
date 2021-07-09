.. _testing:

Testing
################

EXOSIMS provides both unit tests for component level input/output and functionality validation as well as an end-to-end test suite for full scale operational tests.

End to End Testing
=====================

End to end testing is provided by the ``e2eTests.py`` method in the main EXOSIMS directory. This utility is intended to be executed directly (i.e. ``python e2eTests.py``) and looks for test scripts in the ``Scripts/TestScripts/`` directory in the EXOSIMS folder hierarchy.

For each test script, ``e2eTests`` will:

* Create a ``MissionSim`` object
* Execute a simulation via ``MissionSim.run_sim()``
* Reset the simulation via ``MissionSim.reset_sim()``
* Execute a second simulation from the same object via ``MissionSim.run_sim()``

The test suite will record a ``PASS`` or ``FAIL`` condition for each individual step, and will print a summary of results for all scripts at the end of execution. 


Unit Testing
====================

Unit tests are implemented using Python's ``unittest`` framework (see https://docs.python.org/2/library/unittest.html). The unit tests are in the ``tests`` directory under the EXOSIMS root directory.  The ``tests`` directory contains the same folder hierarchy as the EXOSIMS directory, with a separate folder for each module type, a folder for module prototypes, a folder for utility method unit tests and a ``TestSupport`` folder for test-specific code and input scripts. There are three types of unit tests:

* In folder ``ModuleName`` the ``test_ModuleName.py`` object is used to test basic functionality of all implementations of that module (including the prototype).  Test coverage in these objects is limited only to methods found the prototype, and is only executed on specifically overloaded methods in module implementations.
* In folder ``ModuleName``, any ``test_ImplementationName.py`` object is used to test specific functionality of _only_ that module implementation.  These tests are only needed to test specific expected behavior of implementations, or methods in implementations that are not overloading a prototype method.
* In folder ``Prototypes``, the ``test_ModuleName.py`` object is used to test specific functionality of prototype implementations. 

Unit tests can be executed by running:
:: 

    python -m unittest discover -v

from the EXOSIMS root directory.  This will execute all available unit tests (but not the end to end tests). Individual tests can be executed by running:
::

    python -m unittest -v tests.ModuleName.testName

to run all tests within that object or:
::

    python -m unittest -v tests.ModuleName.testName.testMethodName
    

to run a single individual test.

Unit-testing is performed on CircleCI with every pull request/push made to EXOSIMS. CircleCI's parallelization is utilized with 4 sets of tests being run at once, although this could be upgraded arbitrarily based on the CircleCI plan being used. CircleCI feeds tests names into runtests.py, which then returns:
* an XML file which CircleCI uses to split the timinings 
* The coverage results of the primary EXOSIMS which are sent to coveralls.io.

In case one wishes to create a new CircleCI testing setup, use the follow instructions:

* Create a fork of EXOSIMS on github.
* Link your github account to CircleCI if not done already (i.e "make a CircleCI account.")
* Navigate to the project page of CircleCI, and find the associated github repository. Set up a new project. A config.yml file is already supplied, so press "use existing config." The CircleCI component should be working properly, although coveralls isn't set up yet. 
* Link up your github account to coveralls if not done already. Link EXOSIMS to coveralls. Copy the repository token. 
* Next, navigate back to the project page for your EXOSIMS. Click on "Project Settings", "Environment Variables". 
* Add the following environment variable and value and save: 

Variable name: COVERALLS_REPO_TOKEN
Value: [The repository ticket string that was copied before.]

At this point, CircleCI should be working properly. 

Configuration for CircleCI can be found in config.yml file inside of .circleci. 

runtests.py can also be used standalone to create XML files and coverage file via the following command: 
 ::

    python runtests.py $TESTFILES

This will generate xml files inside of test-report and a .coverage file. 

