import unittest
from EXOSIMS.Prototypes.TargetList import TargetList
from tests.TestSupport.Info import resource_path
import os
import astropy
import json
import copy
import numpy as np


class TestMLR(unittest.TestCase):
    """

    Test updated mass-luminosity relationship functionality in
    TargetList.stellar_mass()

    """

    def setUp(self):
        self.dev_null = open(os.devnull, "w")
        self.script = resource_path("test-scripts/test-mlr.json")
        with open(self.script) as f:
            self.spec = json.loads(f.read())
        # self.spec["StarCatalog"] = "HWOMissionStars"
        # Using HWOMissionStars because Fernandes2021 only works on FGK stars

    def tearDown(self):
        self.dev_null.close()

    def test_format(self):
        """
        Tests that each mass model outputs proper type and size, and tests
        to make sure there are no negative masses or overly large masses.
        ** Test only runs with FGK stars.
        """

        TList = TargetList(**copy.deepcopy(self.spec))

        # Henry 1993
        # check that values are within reasonable range and are not zero
        TList.massLuminosityRelationship = "Henry1993"
        TargetList.stellar_mass(TList)
        ind = 0
        for mass in TList.MsTrue:
            val = TList.MsTrue[ind].value.item()
            assert 0.0 <= val < 20.0
            val = TList.MsEst[ind].value.item()
            assert 0.0 <= val < 20.0
            ind += 1
        ind = 0
        # check that output format is correct
        self.assertIsInstance(TList.MsEst, astropy.units.quantity.Quantity)
        self.assertIsInstance(TList.MsTrue, astropy.units.quantity.Quantity)
        self.assertEqual(len(TList.MsEst), len(TList.Name), len(TList.MsTrue))

        # Fernandes 2021
        TList.massLuminosityRelationship = "Fernandes2021"
        TargetList.stellar_mass(TList)
        ind = 0
        for mass in TList.MsTrue:
            val = TList.MsTrue[ind].value.item()
            assert 0.0 <= val < 20.0
            val = TList.MsEst[ind].value.item()
            assert 0.0 <= val < 20.0
        self.assertIsInstance(TList.MsEst, astropy.units.quantity.Quantity)
        self.assertIsInstance(TList.MsTrue, astropy.units.quantity.Quantity)
        self.assertEqual(len(TList.MsEst), len(TList.Name), len(TList.MsTrue))

        # Henry 1993 + 1999
        TList.massLuminosityRelationship = "Henry1993+1999"
        # TargetList.stellar_mass(TList)
        ind = 0
        for mass in TList.MsTrue:
            val = TList.MsTrue[ind].value.item()
            assert 0.0 <= val < 20.0
            val = TList.MsEst[ind].value.item()
            assert 0.0 <= val < 20.0
        self.assertIsInstance(TList.MsEst, astropy.units.quantity.Quantity)
        self.assertIsInstance(TList.MsTrue, astropy.units.quantity.Quantity)
        self.assertEqual(len(TList.MsEst), len(TList.Name), len(TList.MsTrue))

        # Fang 2010
        TList.massLuminosityRelationship = "Fang2010"
        # TargetList.stellar_mass(TList)
        ind = 0
        for mass in TList.MsTrue:
            val = TList.MsTrue[ind].value.item()
            assert 0.0 <= val < 20.0
            val = TList.MsEst[ind].value.item()
            assert 0.0 <= val < 20.0
        self.assertIsInstance(TList.MsEst, astropy.units.quantity.Quantity)
        self.assertIsInstance(TList.MsTrue, astropy.units.quantity.Quantity)
        self.assertEqual(len(TList.MsEst), len(TList.Name), len(TList.MsTrue))

    def test_input(self):
        """
        Checks that valid inputs for MLR are accepted and invalid ones are not.
        """

        # valid inputs
        self.spec = dict(self.spec)
        self.spec["massLuminosityRelationship"] = "Henry1993"
        TList = TargetList(**copy.deepcopy(self.spec))
        self.assertEqual(TList.massLuminosityRelationship, "Henry1993")

        self.spec["massLuminosityRelationship"] = "Fernandes2021"
        TList = TargetList(**copy.deepcopy(self.spec))
        self.assertEqual(TList.massLuminosityRelationship, "Fernandes2021")

        self.spec["massLuminosityRelationship"] = "Henry1993+1999"
        TList = TargetList(**copy.deepcopy(self.spec))
        self.assertEqual(TList.massLuminosityRelationship, "Henry1993+1999")

        self.spec["massLuminosityRelationship"] = "Fang2010"
        TList = TargetList(**copy.deepcopy(self.spec))
        self.assertEqual(TList.massLuminosityRelationship, "Fang2010")

    def test_calc(self):
        """
        Checks a calculated values for each model against expected result
        """

        TList = TargetList(**copy.deepcopy(self.spec))

        testMV = TList.MV
        testL = TList.L

        # checking each MLR
        # Henry 1993
        TList.massLuminosityRelationship = "Henry1993"
        TargetList.stellar_mass(TList)
        Henry1993 = TList.MsEst.value
        ind = 0
        for i in testMV:
            testmass = 10.0 ** (
                0.002456 * testMV[ind] ** 2 - 0.09711 * testMV[ind] + 0.4365
            )
            # done in solar masses
            self.assertEqual(testmass, Henry1993[ind])
            ind += 1

        # Fernandes 2021
        TList.massLuminosityRelationship = "Fernandes2021"
        TargetList.stellar_mass(TList)
        Fernandes2021 = TList.MsEst.value
        ind = 0
        for i in testL:
            testmass = 10 ** (
                (0.219 * np.log10(testL[ind]))
                + (0.063 * ((np.log10(testL[ind])) ** 2))
                - (0.119 * ((np.log10(testL[ind])) ** 3))
            )
            Fval = Fernandes2021[ind]
            self.assertEqual(testmass, Fval)
            ind += 1

        # Henry 1993 extended
        TList.massLuminosityRelationship = "Henry1993+1999"
        TargetList.stellar_mass(TList)
        self.MV = TList.MV
        count = 0
        # initialize MsEst attribute
        self.MsEst = np.array([])
        # start index count for multiple relations
        count = 0
        for MV in self.MV:
            if 0.50 <= self.MV[count] <= 2.0:
                mass = (
                    10.0
                    ** (
                        0.002456 * self.MV[count] ** 2
                        - 0.09711 * self.MV[count]
                        + 0.4365
                    )
                ).item()
                self.MsEst = np.append(self.MsEst, mass)
                err = (np.random.random(len(self.MV)) * 2.0 - 1.0) * 0.07
                count += 1
            elif 0.18 <= self.MV[count] < 0.50:
                mass = (10.0 ** (-0.1681 * self.MV[count] + 1.4217)).item()
                self.MsEst = np.append(self.MsEst, mass)
                count += 1
            elif 0.08 <= self.MV[count] < 0.18:
                mass = (
                    10
                    ** (
                        0.005239 * self.MV[count] ** 2
                        - 0.2326 * self.MV[count]
                        + 1.3785
                    )
                ).item()
                self.MsEst = np.append(self.MsEst, mass)
                count += 1
            else:
                # default to Henry 1993
                mass = (
                    10.0
                    ** (
                        0.002456 * self.MV[count] ** 2
                        - 0.09711 * self.MV[count]
                        + 0.4365
                    )
                ).item()
                self.MsEst = np.append(self.MsEst, mass)
                count += 1
        ind = 0
        for i in self.MV:
            self.assertEqual(self.MsEst[ind], TList.MsEst.value[ind])
            ind += 1

        # Fang 2010
        TList.massLuminosityRelationship = "Fang2010"
        TargetList.stellar_mass(TList)
        self.MV = TList.MV
        count = 0
        # initialize MsEst attribute
        self.MsEst = np.array([])
        # start index count for multiple relations
        count = 0
        for MV in self.MV:
            if self.MV[count] <= 1.05:
                mass = (10 ** (0.558 - 0.182 * MV - 0.0028 * MV**2)).item()
                self.MsEst = np.append(self.MsEst, mass)
                err = (np.random.random(1) * 2.0 - 1.0) * 0.05
                self.MsTrue = (1.0 + err) * self.MsEst
                count += 1
            else:
                mass = (
                    10
                    ** (0.489 - 0.125 * self.MV[count] + 0.00511 * self.MV[count] ** 2)
                ).item()
                self.MsEst = np.append(self.MsEst, mass)
                err = (np.random.random(1) * 2.0 - 1.0) * 0.07
                self.MsTrue = (1.0 + err) * self.MsEst
                count += 1
        ind = 0
        for i in self.MV:
            self.assertEqual(self.MsEst[ind], TList.MsEst.value[ind])
            ind += 1

    def test_piecewise_bounds(self):
        """
        Ensures that the boundary values for Henry1993+1999 and
        Fang2010 are properly covered.
        """

        # Henry1993+1999
        self.spec["massLuminosityRelationship"] = "Henry1993+1999"
        TList = TargetList(**copy.deepcopy(self.spec))
        MV = np.array([0.08, 0.18, 0.50, 2.0])
        TList.MV = MV
        TargetList.stellar_mass(TList)
        # 0.08
        test1 = 10 ** (0.005239 * TList.MV[0] ** 2 - 0.2326 * TList.MV[0] + 1.3785)
        # 0.18
        test2 = 10.0 ** (-0.1681 * TList.MV[1] + 1.4217)
        # 0.50
        test3 = 10.0 ** (0.002456 * TList.MV[2] ** 2 - 0.09711 * TList.MV[2] + 0.4365)
        # 2.00
        test4 = 10.0 ** (0.002456 * TList.MV[3] ** 2 - 0.09711 * TList.MV[3] + 0.4365)
        testEst = np.array([test1, test2, test3, test4])
        ind = 0
        for mass in testEst:
            self.assertEqual(testEst[ind], TList.MsEst.value[ind])
            ind += 1

        # Fang 2010
        self.spec["massLuminosityRelationship"] = "Fang2010"
        TList = TargetList(**copy.deepcopy(self.spec))
        MV = np.array([1.05])
        TList.MV = MV
        TargetList.stellar_mass(TList)
        # 1.05
        test1 = 10 ** (0.558 - 0.182 * TList.MV[0] - 0.0028 * TList.MV[0] ** 2)
        testEst = np.array([test1])
        ind = 0
        self.assertEqual(testEst[0], TList.MsEst.value[0])
