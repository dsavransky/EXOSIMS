import unittest
import EXOSIMS.util.get_dirs as gd
import os

class TestGetDirs(unittest.TestCase):
    """
    Tests the get_dir tool.

    Sonny Rappaport, Cornell, July 2021 
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

        self.assertEqual(outputs_rel,['.', '.', '.', '.', '.', '..\\.EXOSIMS\\cache', '.'])