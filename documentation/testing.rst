.. _testing:

Testing
################

EXOSIMS provides both unit tests for component level input/output and functionality validation as well as an end-to-end test suite for full scale operational tests. Unit tests are intended to be run as part of EXOSIMS's continuous integration process, while the end to-end tests are typically deployed prior to major merges and releases to ensure continuity of baseline functionality. 

End-to-End Testing
=====================

End-to-end testing is provided by the :py:mod:`~EXOSIMS.e2eTests` script in the main EXOSIMS directory. This utility is intended to be executed directly (i.e. ``python e2eTests.py``) and looks for test scripts in the ``Scripts/TestScripts/`` directory in the EXOSIMS folder hierarchy.  New scripts can be added at any time and will be automatically used the next time the test suite is executed. 

For each test script, ``e2eTests`` will:

* Create a ``MissionSim`` object
* Execute a simulation via ``MissionSim.run_sim()``
* Reset the simulation via ``MissionSim.reset_sim()``
* Execute a second simulation from the same object via ``MissionSim.run_sim()``

The test suite will record a ``PASS`` or ``FAIL`` condition for each individual step, and will print a summary of results for all scripts at the end of execution. 


Unit Testing
====================

Unit tests are implemented using Python's ``unittest`` framework (see https://docs.python.org/3/library/unittest.html). The unit tests are in the ``tests`` directory under the EXOSIMS root directory (see :ref:`exosimsdirs`).  The ``tests`` directory contains the same folder hierarchy as the EXOSIMS directory, with a separate folder for each module type, a folder for module prototypes, a folder for utility method unit tests.  Additionally, there is a ``TestSupport`` folder for test-specific code and input scripts, and a ``TestModules`` folder containing test-specific module implementations (typically for testing error condition checks in the prototypes). 

There are three types of unit tests:

* In folder ``ModuleName`` the ``test_ModuleName.py`` object is used to test basic functionality of all implementations of that module (including the prototype).  Test coverage in these objects is limited only to methods found the prototype, and is only executed on specifically overloaded methods in module implementations.
* In folder ``ModuleName``, any ``test_ImplementationName.py`` object is used to test specific functionality of _only_ that module implementation.  These tests are only needed to test specific expected behavior of implementations, or methods in implementations that are not overloading a prototype method.
* In folder ``Prototypes``, the ``test_ModuleName.py`` object is used to test specific functionality of prototype implementations. 

Unit tests can be executed (from the command line) by running:
:: 

    python -m unittest discover -v

from the EXOSIMS root directory.  This will execute all available unit tests (but not the end-to-end tests). Individual tests can be executed by running:
::

    python -m unittest -v tests.ModuleName.testName

to run all tests within that object or:
::

    python -m unittest -v tests.ModuleName.testName.testMethodName

to run a single individual test.

Unit-testing is performed via GitHub actions on the main repository, as configured by the ``ci.yml`` file.  Alternatively, CircleCI can be used. CircleCI feeds tests names into ``runtests.py``, which then returns:

* an XML file which CircleCI uses to split the timings 
* The coverage results of the primary EXOSIMS which are sent to coveralls.io.

The basic steps to create a CircleCI testing setup for a fork of the repository are:

* Create a fork of EXOSIMS on GitHub.
* Link your GitHub account to CircleCI 
* Navigate to the project page of CircleCI, and find the associated GitHub repository. Set up a new project. A config.yml sample file is provided below.
* Link up your github account to coveralls if not done already. Link EXOSIMS to coveralls. Copy the repository token. 
* In the CircleCI project, navigate to "Project Settings">"Environment Variables" and add a new variable with name ``COVERALLS_REPO_TOKEN`` and value set to the repsitory token.


``runtests.py`` can also be used standalone to create XML files and coverage file by running: 
::

    python runtests.py $TESTFILES

This will generate xml files inside of test-report and a .coverage file. 


Sample ``.circlci/config.yml``:

.. code-block:: 

    version: 2.1
    jobs:
      test:
        parallelism: 4
        #run 4 test runs at the same time
        working_directory: ~/circleci-python
        docker:
          - image: "circleci/python:3.9.5"
        # uses python 3.9.5 
        steps:
          - checkout
          - run: 
              name: Grant local/bin permission
              command: sudo chown -R circleci:circleci /usr/local/bin
          - run: 
              name: Grant lib/python3.9 permission
              command: sudo chown -R circleci:circleci /usr/local/lib/python3.9/site-packages
          - restore_cache:
              name: install dependencies start (install cache if one exists)
              key: -v3-{{ checksum "requirements.txt" }}
          - run: pip install --upgrade pip
          - run: pip install --upgrade -r requirements.txt
          - run: pip install coverage
          - run: pip install unittest-xml-reporting
          - run: pip install -e .
          - save_cache: 
              key: -v3-{{ checksum "requirements.txt" }}
              name: install dependencies end (generate cache if one doesn't exist)
              paths: 
                - ".venv"
                - "/usr/local/lib/python3.9/site-packages"
                - "/usr/local/bin"
          # build the environment. the caching steps restore the dependencies from CircleCI's servers from past runs to make things a bit faster 
          - run: mkdir -p coverages
          # coverage files for each parallel run is stored in here
          - run: 
              name: find and run tests and generate coverage files
              no_output_timeout: 60m
              # some tests take quite a long time to finish so the default time has to be increased, particularly when circleci first generates the timing data for the tests 
              command: |
                TESTFILES=$(circleci tests glob "tests/**/test_*.py" | circleci tests split --split-by=timings)
                python runtests.py $TESTFILES
          - store_test_results:
              path: test-reports
          # store test results for circleci's test splitting 
          - store_artifacts:
              path: test-reports 
          - persist_to_workspace:
              root: coverages
              paths: 
                - ./*
          # save the .coverage to the project's workspace to be combined in the done step

      done:
        working_directory: ~/circleci-python
        docker:
        - image: "circleci/python:3.9.5"
        steps:
          - checkout
          - run: pip install coverage
          - run: pip install coveralls
          - attach_workspace: 
              at: ~/
          - run: coverage combine ~/
          # combine the coverage files from the workspace in the home directory
          - run: coveralls 

    workflows:
      test_then_upload:
        jobs:
          - test
          - done: 
              requires: [test]


