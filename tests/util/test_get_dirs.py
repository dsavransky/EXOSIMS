import unittest
import EXOSIMS.util.get_dirs as gd
import os
from unittest.mock import *
import numpy as np
import sys


class TestGetDirs(unittest.TestCase):
    """
    Tests the get_dir tool.

    Sonny Rappaport, Cornell, July 2021
    """

    def test_get_home_dir(self):
        """
        Tests that get_home_dir works in muiltiple OS environments.

        Test method: Uses unittest's mock library to create fake OS environment
        and paths to see if get_dirs returns the correct home directory. Because
        get_dirs returns assertionerrors when the homedir isn't real, use the
        assertion message itself to check that the homedir is correct.

        This assumes that the os library does its job correctly as the mocking
        library will overwrite whatever os has stored for testing purposes.

        This method also assumes that winreg works as expected.
        """
        # collect assertion errors and verify at the end that we only get the
        # expected assertion errors.
        # this tests the assertion error as well- it should be called for all
        # of these cases as I use imaginary pathnames
        assertErrors = []

        # mock directories
        directories = [
            {"HOME": "posixhome"},
            {},
            {"HOME": "myshome", "MSYSTEM": "test"},
            {"HOMESHARE": "sharehome"},
            {"USERPROFILE": "userhome"},
            {"HOME": "otherOShome"},
            {},
        ]

        # mock os names
        os_name = ["posix", "posix", "nt", "nt", "nt", "door", "door"]

        # names for home directory- 'none' shouldn't show up
        home_names = [
            "posixhome",
            "none",
            "myshome",
            "sharehome",
            "userhome",
            "otherOShome",
            "none",
        ]

        # test all paths except for winreg
        for i, dic in enumerate(directories):
            with (
                patch.dict(os.environ, dic, clear=True),
                patch.object(os, "name", os_name[i]),
            ):
                # i==1 and i==6 correspond to where homedir isn't in environ
                if i == 1 or i == 6:
                    with self.assertRaises(OSError):
                        gd.get_home_dir()
                else:
                    try:
                        gd.get_home_dir()
                    except AssertionError as e:
                        assertErrors.append(str(e))
        # add all assertion errors so far to the expected list of assertion
        # errors
        exp_asrt = []
        for s in home_names:
            if s == "none":
                continue
            exp_asrt.append(
                "Identified "
                + s
                + " as home directory, but it does"
                + " not exist or is not accessible/writeable"
            )

        # test winreg branch

        # first, test that if winreg doesn't except, homedir is set
        # (mock a key: make key functions do nothing.
        # mock queryvalueex: return test homedir)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(os, "name", "nt"),
            patch.dict(sys.modules, {"winreg": MagicMock()}),
            patch("winreg.OpenKey"),
            patch("winreg.QueryValueEx") as mockquery,
        ):
            mockquery.return_value = ["winregHome"]
            try:
                gd.get_home_dir()
            except AssertionError as e:
                assertErrors.append(str(e))

        # second, test that home is tried if an exception is raised and attempt
        # at homedir setting is made

        with (
            patch.dict(os.environ, {"HOME": "winreghome2"}, clear=True),
            patch.object(os, "name", "nt"),
            patch.dict(sys.modules, {"winreg": MagicMock()}),
            patch("winreg.OpenKey"),
            patch("winreg.QueryValueEx") as mockquery,
        ):
            mockquery.side_effect = Exception
            try:
                gd.get_home_dir()
            except AssertionError as e:
                assertErrors.append(str(e))

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(os, "name", "nt"),
            patch.dict(sys.modules, {"winreg": MagicMock()}),
            patch("winreg.OpenKey"),
            patch("winreg.QueryValueEx") as mockquery,
        ):
            mockquery.side_effect = Exception
            with self.assertRaises(OSError):
                gd.get_home_dir()

        exp_asrt.append(
            "Identified "
            + "winregHome"
            + " as home directory, but it does"
            + " not exist or is not accessible/writeable"
        )

        exp_asrt.append(
            "Identified "
            + "winreghome2"
            + " as home directory, but it does"
            + " not exist or is not accessible/writeable"
        )

        np.testing.assert_array_equal(assertErrors, exp_asrt)

    def test_get_paths(self):
        """
        Tests that get_paths returns the proper (relative) paths.

        Test method: Calls the method and tests to see if the path dictionary
        matches expectations for various trivial inputs. For some cases, use the
        python mock library to simplify testing

        For the JSON, queue file, and and runqueue branches, just use a simple
        dictionary (*although this should probably be changed to the respective
        datatype. )
        """

        # test no parameter output, testing branch #1.
        # mock current working directory
        dict_paths = gd.get_paths()
        outputs = dict_paths.values()
        outputs_rel = []
        for x in outputs:
            outputs_rel.append(os.path.relpath(x))

        # test environment output, testing branch #2. mock environment dictionary
        with patch.dict(
            os.environ, {"EXOSIMS1": "exosims_path", "EXOSIMS2": "exosims_path2"}
        ):

            # only keep the key/values i seek to test for each branch
            test_dict = dict()
            dict_paths = gd.get_paths()
            for key in dict_paths:
                if key == "EXOSIMS1" or key == "EXOSIMS2":
                    test_dict[key] = dict_paths[key]

            self.assertDictEqual(
                test_dict, {"EXOSIMS1": "exosims_path", "EXOSIMS2": "exosims_path2"}
            )

        # test JSON script output, branch #3. mock
        paths = {
            "EXOSIMS_SCRIPTS_PATH": "scriptspath",
            "EXOSIMS_OBSERVING_BLOCK_CSV_PATH": "csvpath",
            "EXOSIMS_FIT_FILES_FOLDER_PATH": "folderpath",
            "EXOSIMS_PLOT_OUTPUT_PATH": "outputpath",
            "EXOSIMS_RUN_SAVE_PATH": "savepath",
            "EXOSIMS_RUN_LOG_PATH": "logpath",
            "EXOSIMS_QUEUE_FILE_PATH": "filepath",
        }
        paths_test = {"paths": paths}
        self.assertDictEqual(paths, gd.get_paths(specs=paths_test))

        # test qFile script specified path, branch #4
        self.assertDictEqual(paths, gd.get_paths(qFile=paths_test))

        # test runQueue specified path, branch #5
        self.assertDictEqual(paths, gd.get_paths(qFargs=paths))
