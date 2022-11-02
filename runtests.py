import unittest
import xmlrunner
from coverage import Coverage
import sys

"""
Unittest-compatible test-running script

Sonny Rappaport, Cornell June 7/6/2021. Written in conjunction with .circleci/config.yml
to work with circleci's parallelization, coverage.py, and coveralls.
"""


def format_path(file_path):
    """Converts from the default bash /directory/test.py format to directory.test
    format (as unittest only works with )

    Args:
        file_path (String): The name of the file to be formatted. Should be in a
        /.../..../test.py format.

    """

    no_py = file_path.replace(".py", "")
    no_slash = no_py.replace("/", ".")

    return no_slash


if __name__ == "__main__":
    """When called via bash with a list of file names, creates and runs a unittest
    suite over all the test files passed into this method, generating both a XML
    file and a .coverage file for each parallel run on circleci. The XML file is placed
    in exosims/test-reports and the XML file is placed in the exosims root folder.

    Command line usage example:

    python runtests.py [List of testfile names]
    """
    cov = Coverage()
    cov.start()
    loader = unittest.TestLoader()
    tests_format = []
    for x in sys.argv:
        tests_format.append(format_path(x))
    # sys.argv (argument from bash) should contain a list of file names
    suites = [loader.loadTestsFromName(str) for str in tests_format]
    combosuite = unittest.TestSuite(suites)
    # create suite of all tests
    runner = xmlrunner.XMLTestRunner(output="test-reports")
    # generate XML files containing the times of each test for circle-ci's
    # test-splitting via time
    runner.run(combosuite)
    cov.stop()
    cov.save()
