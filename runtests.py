import unittest
import xmlrunner
from coverage import Coverage
import sys

def format_path(file_path):
    """Converts from the default bash /directory/test.py format to directory.test
    format"""

    no_py = file_path.replace(".py","")

    no_slash = no_py.replace("/",".")

    return no_slash

if __name__ == "__main__":
    """Creates and runs a unittest suite over all the test files passed into
    this method, generating both a XML file and a .coverage file for each 
    parallel run on circleci"""

    cov = Coverage()
    loader = unittest.TestLoader()
    tests = []
    for x in sys.argv: 
      tests.append(format_path(x))
    suites = [loader.loadTestsFromName(str) for str in tests]
    combosuite = unittest.TestSuite(suites)
    runner = xmlrunner.XMLTestRunner(output='test-reports')
    cov.start()
    runner.run(combosuite)
    cov.stop() 
    cov.save() 
