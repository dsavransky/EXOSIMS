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

    def test_get_home_dir(self): 
        """
        Tests that get_home_dir works in muiltiple OS environments.

        Test method: Uses unittest's mock library to create fake OS environment
        and paths to see if get_dirs returns the correct home directory. 
        This assumes that the os library does its job correctly as the mocking
        library will overwrite whatever os has stored for testing purposes.

        This method also assumes that winreg works as expected. 
        """
        #collect assertion errors and verify at the end that we only get the 
        #expected assertion errors.
        #this tests the assertion error as well- it should be called for all
        #of these cases as I use imaginary pathnames 
        assertErrors = [] 

        #mock directories 
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

        #mock os names
        os_name = ['posix','posix','nt','nt','nt','door','door']
        
        #names for home directory- 'none' shouldn't show up 
        home_names = ['posixhome','none','myshome','sharehome','userhome',
                'otherOShome', 'none']

        #test all paths except for winreg 
        for i, dic in enumerate(directories):
            with patch.dict(os.environ,dic,clear=True), \
            patch.object(os,'name',os_name[i]):
                #i==1 and i==6 correspond to where homedir isn't in environ
                if i == 1 or i == 6: 
                    with self.assertRaises(OSError):
                        gd.get_home_dir()
                else: 
                    try: self.assertEqual(gd.get_home_dir(),"hot dog")



# have to fix this

                    except AssertionError as e: 
                        assertErrors.append(str(e))
        #add all assertion errors so far to the expected list of assertion 
        #errors 
        exp_asrt = []
        for s in home_names: 
            if s == 'none': 
                continue
            exp_asrt.append("Identified "+s+ " as home directory, but it does" +
            " not exist or is not accessible/writeable")

        #test winreg branch 

        #first, test that if winreg doesn't except, homedir is set 
        #(mock a key: make keys do nothing. mock queryvalueex: return test homedir)

        with patch.dict(os.environ,{},clear=True), \
            patch.object(os,'name','nt'), \
            patch('winreg.OpenKey'), \
            patch('winreg.QueryValueEx') as mockquery:
                mockquery.return_value= ['winregHome']
                try: self.assertEqual(gd.get_home_dir(), 'winrefklsdfskafjgHome')
                except AssertionError as e: 
                    assertErrors.append(str(e))

        exp_asrt.append("Identified "+"winregHome"+ " as home directory, but it does" +
                " not exist or is not accessible/writeable")
        
        np.testing.assert_array_equal(assertErrors ,exp_asrt)
                

            


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

        if os.name == 'nt':
            self.assertEqual(first = outputs_rel,
                second = ['.', '.', '.', '.', '.', '..\\.EXOSIMS\\cache', '.'])
        else: 
            self.assertEqual(first = outputs_rel,
                second = ['.', '.', '.', '.', '.', '../.EXOSIMS/cache', '.'])