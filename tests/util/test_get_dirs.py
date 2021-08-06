import unittest
import EXOSIMS.util.get_dirs as gd
import os
from unittest.mock import * 
import numpy as np

class TestGetDirs(unittest.TestCase):
    """
    Tests the get_dir tool.

    Sonny Rappaport, Cornell, July 2021 
    """

    def setUp(self): 
        self.assertErrors = []

    def tearDown(self):

        #Strings are added in the order the assertions carrying them
        #should be created below. 

        exp_homes = ['posixhome','myshome','sharehome','userhome', 'otherOShome']

        exp_asrt = []

        for s in exp_homes: 
            exp_asrt.append("Identified "+s+ " as home directory, but it does" +
            " not exist or is not accessible/writeable" )
        np.testing.assert_array_equal(self.assertErrors,exp_asrt)

    def test_get_home_dir(self): 
        """
        Tests that get_home_dir works in muiltiple OS environments.

        Test method: Uses unittest's mock library to create fake OS environment
        and paths to see if get_dirs returns the correct home directory. setUp
        and tearDown are used to ignore the assertion errors that will be caused
        by os.environ. This is going to assume that os.environ actually contains
        the correct paths. 
        """

        directories = \
        [
            {'HOME':'posixhome'},
            {},
            {'HOME':'myshome','MSYSTEM':'test'},
            {'HOMESHARE':'sharehome'},
            {'USERPROFILE':'userhome'},
            {'HOME': 'otherOShome'},
            {}
        ]

        os_name = ['posix','posix','nt','nt','nt','door','door']
        
        home_names = ['posixhome','none','myshome','sharehome','userhome',
                'otherOShome' 'none']

        for i, dic in enumerate(directories):
            with patch.dict(os.environ,dic,clear=True), \
            patch.object(os,'name',os_name[i]):
                if i == 1 or i == 6: 
                    with self.assertRaises(OSError):
                        gd.get_home_dir()
                else: 
                    try: self.assertEqual(gd.get_home_dir(),home_names[i])
                    except AssertionError as e: 
                        self.assertErrors.append(str(e))

                

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