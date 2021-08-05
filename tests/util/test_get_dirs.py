import unittest
import EXOSIMS.util.get_dirs as gd
import os
from unittest.mock import * 

class TestGetDirs(unittest.TestCase):
    """
    Tests the get_dir tool.

    Sonny Rappaport, Cornell, July 2021 
    """

    def test_get_home_dir(self): 
        """
        Tests that get_home_dir works in muiltiple OS environments.

        Test method: Uses unittest's mock library to create fake OS environment
        and paths to see if get_dirs returns the correct home directory. 
        """

        

    def test_get_paths(self): 

        """
        Tests that get_paths returns the proper (relative) paths. 

        Test method: Calls the method and tests to see if the path dictionary 
        matches expectations for various trivial inputs. 
        """

        #test no parameter output 
        dict_paths = gd.get_paths()
        outputs = dict_paths.values()
        outputs_rel = []
        for x in outputs: 
            outputs_rel.append(os.path.relpath(x))

        #test doesn't work on windows for now
        self.assertEqual(outputs_rel,['.', '.', '.', '.', '.', '../.EXOSIMS/cache', '.'])