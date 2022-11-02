#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>

r"""Utility unit tests -- get_module

Michael Turmon, JPL, Apr. 2016
"""

import unittest
import os
import inspect
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.get_module import get_module_from_specs


EXO_SOURCE = os.path.dirname(os.path.dirname(inspect.getfile(get_module)))


class TestUtilityMethods(unittest.TestCase):
    r"""Test utility functions."""

    def validate_module(self, m, modtype=None):
        self.assertTrue(inspect.isclass(m))
        self.assertTrue(hasattr(m, "_modtype"))
        if modtype:
            self.assertEqual(m._modtype, modtype)

    ##############
    ## Tests for  case 1: source is explicitly named
    ##############

    def test_get_module_source_1(self):
        r"""Test init-from-source -- prototype.

        Approach: Load module from a file we know is present."""
        source = os.path.join(EXO_SOURCE, "Prototypes", "TimeKeeping.py")
        # basic test
        m = get_module(source, "TimeKeeping")
        self.validate_module(m, "TimeKeeping")
        # again -- unspecified type
        m = get_module(source)
        self.validate_module(m, "TimeKeeping")

    def test_get_module_source_2(self):
        r"""Test init-from-source -- not a prototype.

        Approach: Load module from a file we know is present."""
        source = os.path.join(EXO_SOURCE, "PlanetPopulation", "KnownRVPlanets.py")
        # basic test
        m = get_module(source)
        self.validate_module(m, "PlanetPopulation")
        # again -- unspecified type
        m = get_module(source)
        self.validate_module(m, "PlanetPopulation")

    def test_get_module_source_fail_not_there(self):
        r"""Test init-from-source -- failure

        Approach: Load module from a file we know is not present."""
        # basic test
        source = os.path.join(EXO_SOURCE, "NotPresent", "NotPresent.py")
        with self.assertRaises(ValueError):
            m = get_module(source, "Not_used")

    def test_get_module_source_fail_wrong_type(self):
        r"""Test init-from-source -- failure.

        Approach: Load module, but ask for the wrong type."""
        source = os.path.join(EXO_SOURCE, "Prototypes", "TimeKeeping.py")
        #
        with self.assertRaises(AssertionError):
            m = get_module(source, "WrongModuleType")

    ##############
    ## Tests for case 2A: named module, named folder
    ##############

    def test_get_module_Mname_Fname_1(self):
        r"""Test get-from-named-folder -- prototype.

        Approach: Load module from a file we know is present."""
        module = "TimeKeeping"
        # basic test
        m = get_module(module, module)
        self.validate_module(m, module)

    def test_get_module_Mname_Fname_2(self):
        r"""Test get-from-named-folder -- non-prototype.

        Approach: Load module from a file we know is present."""
        module = "KnownRVPlanets"
        folder = "PlanetPopulation"
        # basic test
        m = get_module(module, folder)
        self.validate_module(m, folder)

    def test_get_module_Mname_Fname_default(self):
        r"""Test get-from-named-folder -- prototype, shortcut using blanks.

        Approach: Load module from a file we know is present."""
        module = " "
        folder = "PlanetPopulation"
        # basic test
        m = get_module(module, folder)
        self.validate_module(m, folder)

    def test_get_module_Mname_Fname_fail_not_there(self):
        r"""Test get-from-named-folder -- failure

        Approach: Load module from a source we know is not present."""
        module = "KnownRVPlanets_not_there"
        folder = "PlanetPopulation"
        # fails with ValueError: not found
        with self.assertRaises(ValueError):
            m = get_module(module, folder)

    ##############
    ## Tests for case 2B: named module, un-named folder
    ##############

    def test_get_module_Mname_ok_1(self):
        r"""Test get-no-folder -- Prototype

        Approach: Load module from a source we know is present."""
        module = "TimeKeeping"
        m = get_module(module)
        self.validate_module(m, module)

    def test_get_module_Mname_ok_2(self):
        r"""Test get-no-folder -- Prototype

        Approach: Load module from a source we know is present."""
        module = "PlanetPopulation"
        m = get_module(module)
        self.validate_module(m, module)

    def test_get_module_Mname_ok_star(self):
        r"""Test get-no-folder -- found by searching below EXOSIMS for a named non-Prototype module.

        Approach: Load module from a *non-Prototype* source we know is present."""
        module = "KnownRVPlanets"
        m = get_module(module)
        # this is a non-Prototype module of type PlanetPopulation
        self.validate_module(m, "PlanetPopulation")

    def test_get_module_Mname_fail_not_there(self):
        r"""Test get-no-folder -- failure

        Approach: Load module from a source we know is not present."""
        module = "KnownRVPlanets_not_there"
        with self.assertRaises(ValueError):
            m = get_module(module)

    ##############
    ## Tests for case 3: non-EXOSIMS module
    ##############

    def test_get_module_non_exosims_ok_1(self):
        r"""Test non-EXOSIMS loading -- simple source.

        Approach: Load module from a source we know is present."""
        module = "tests.TestModules.TimeKeeping.TimeKeepingModified"
        modtype = "TimeKeeping"
        m = get_module(module, modtype)
        self.validate_module(m, modtype)

    def test_get_module_non_exosims_ok_2(self):
        r"""Test non-EXOSIMS loading -- simple source.

        Approach: Load module from a source we know is present."""
        module = "tests.TestModules.PlanetPopulation.PlanetPopulationModified"
        modtype = "PlanetPopulation"
        m = get_module(module, modtype)
        self.validate_module(m, modtype)

    def test_get_module_non_exosims_notthere_package(self):
        r"""Test non-EXOSIMS loading -- failure due to absent package

        Approach: Load module from a package we know is not present."""
        module = "tests_not_there"
        modtype = "PlanetPopulation"
        with self.assertRaises(ValueError):
            m = get_module(module, modtype)

    def test_get_module_non_exosims_notthere_module(self):
        r"""Test non-EXOSIMS loading -- failure due to absent module

        Approach: Load a module we know is not present."""
        module = "tests.TestModules.TimeKeepingNotThere"
        modtype = "TimeKeeping"
        with self.assertRaises(ValueError):
            m = get_module(module, modtype)

    # the following several tests can be done, using a local module, because
    # we can make the local module in-valid by putting in the wrong _modtype
    # value to the class, etc.  Of course, real EXOSIMS modules can't have
    # such errors.
    def test_get_module_non_exosims_error_modtype_mismatch(self):
        r"""Test non-EXOSIMS loading -- _modtype attribute mismatch.

        Approach: Load module, ask for the wrong modtype."""
        module = "tests.TestModules.PlanetPopulation.PlanetPopulationModified"
        modtype = "TimeKeeping"  # note: TimeKeeping vs. PlanetPopulation
        with self.assertRaises(AssertionError):
            m = get_module(module, modtype)

    def test_get_module_non_exosims_error_modtype_wrong(self):
        r"""Test non-EXOSIMS loading -- _modtype attribute wrong.

        Approach: Load module having the wrong _modtype attribute."""
        module = "tests.TestModules.PlanetPopulation.PlanetPopulationErrorModtype"
        modtype = "PlanetPopulation"
        with self.assertRaises(AssertionError):
            m = get_module(module, modtype)

    def test_get_module_non_exosims_error_no_modtype(self):
        r"""Test non-EXOSIMS loading -- no _modtype attribute at all

        Approach: Load module having no _modtype attribute."""
        module = (
            "tests.TestModules.PlanetPopulation.PlanetPopulationErrorMissingModtype"
        )
        modtype = "PlanetPopulation"
        with self.assertRaises(AssertionError):
            m = get_module(module, modtype)

    def test_get_module_non_exosims_error_naming(self):
        r"""Test non-EXOSIMS loading -- class name wrong.

        Approach: Load module which defines a class with the wrong name."""
        module = "tests.TestModules.PlanetPopulation.PlanetPopulationErrorNaming"
        modtype = "PlanetPopulation"
        with self.assertRaises(AssertionError):
            m = get_module(module, modtype)

    ##############
    ## Tests for get_module_from_specs
    ##############

    def test_get_module_from_specs_1(self):
        r"""Test get-from-specs -- prototype.

        Approach: Load module from a file we know is present."""
        module = "TimeKeeping"
        modtype = module
        # a dictionary
        specs = {"modules": {modtype: module}}
        # basic test
        m = get_module_from_specs(specs, modtype)
        self.validate_module(m, modtype)

    def test_get_module_from_specs_2(self):
        r"""Test get-from-specs -- prototype, again.

        Approach: Load module from a file we know is present."""
        module = "PlanetPopulation"
        modtype = module
        # a dictionary
        specs = {"modules": {modtype: module}}
        # basic test
        m = get_module_from_specs(specs, modtype)
        self.validate_module(m, modtype)

    def test_get_module_from_specs_3(self):
        r"""Test get-from-specs -- non-prototype.

        Approach: Load module from a file we know is present."""
        module = "KnownRVPlanets"
        modtype = "PlanetPopulation"
        # a dictionary
        specs = {"modules": {modtype: module}}
        # basic test
        m = get_module_from_specs(specs, modtype)
        self.validate_module(m, modtype)

    def test_get_module_from_specs_4(self):
        r"""Test get-from-specs -- non-EXOSIMS.

        Approach: Load module from a file we know is present."""
        module = "tests.TestModules.TimeKeeping.TimeKeepingModified"
        modtype = "TimeKeeping"
        # a dictionary
        specs = {"modules": {modtype: module}}
        # basic test
        m = get_module_from_specs(specs, modtype)
        self.validate_module(m, modtype)

    # test to be sure errors propagate up
    def test_get_module_from_specs_error(self):
        r"""Test get-from-specs -- error.

        Approach: Load module from a file we know is present."""
        module = "KnownRVPlanetsNotThere"
        modtype = "PlanetPopulation"
        # a dictionary
        specs = {"modules": {modtype: module}}
        # basic test
        with self.assertRaises(ValueError):
            m = get_module_from_specs(specs, modtype)


if __name__ == "__main__":
    unittest.main()
