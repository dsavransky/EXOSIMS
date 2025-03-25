import unittest
from tests.TestSupport.Utilities import RedirectStreams
import EXOSIMS.StarCatalog
from EXOSIMS.StarCatalog.HIPfromSimbad import HIPfromSimbad
from EXOSIMS.util.get_module import get_module
import os, sys
import pkgutil
from io import StringIO
import astropy.units as u
from EXOSIMS.util.get_dirs import get_downloads_dir
import shutil
import csv
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.exceptions import TableParseError


class TestHIPfromSimbad(unittest.TestCase):
    """
    Sonny Rappaport, July 2021, Cornell

    This class tests HIPfromSimbad.

    """

    def setUp(self):
        """

        Set up HIPfromSimbad modules via both a text file and a list, with the
        particular stars being used arbitrarily.

        """
        # list of HIP numbers to be used, chosen (arbitrarily) from min to max
        hip_list = [37279, 97649, 32349]

        self.list_fixture = HIPfromSimbad(catalogpath=hip_list)

        path = "tests/TestSupport/test-scripts/HIPfromSimbadTestText.txt"
        self.text_fixture = HIPfromSimbad(catalogpath=path)

    def test_init(self):
        """
        Test of initialization and __init__.

        Test method: Manually place data from the Simbad database into an CSV
        file (called "HIPFromSimbaadTestCSV", placed in the test-scripts folder)
        and load this into python. Check that HIPfromSimbad correctly gathers
        this information using astroquery via numpy tests.

        Note: Some data isn't taken from the Simbad database. They are noted
        with comments.

        Check both loading filenames from a text file (called
        "HIPfromSimbadinput.txt" and from a list of identification HIP #'s.)

        """

        # nickname for the overall object
        HIP_list = self.list_fixture
        HIP_text = self.text_fixture

        # same data from before, just in CSV format for comparison
        path = "tests/TestSupport/test-scripts/HIPFromSimbadTestCSV.csv"
        expected = np.genfromtxt(
            path, delimiter=",", names=True, dtype=None, encoding=None
        )

        # prepare expected lists
        expected_names = np.array(expected["name"].astype("str"))

        # not compared with HIPfromSimbad- just used to compute distances
        expected_parallax = expected["parallax"].astype("float") * u.mas

        expected_distance = expected_parallax.to("pc", equivalencies=u.parallax())

        expected_coord_str = expected["COORDS"].astype("str")

        expected_coords = SkyCoord(
            expected_coord_str, unit=(u.hourangle, u.deg, u.arcsec)
        )

        expected_spec = expected["spec"].astype("str")

        # note- the vmag here isn't the visual magnitude (Vmag)  found in simbad.
        # instead, as per the code's specifications, it's the HPmag, the median
        # magnitude. this is following how the code works in HIPfromSimbad.
        # this is taken from the Gaia catalog, hipparcos_newreduction.
        expected_vmag = expected["vmag"].astype("float")

        expected_hmag = expected["hmag"].astype("float")

        expected_imag = expected["imag"].astype("float")

        expected_bmag = expected["bmag"].astype("float")

        expected_kmag = expected["kmag"].astype("float")

        # this BV value is taken from the Gaia catalog, hippercos_newreduction.
        expected_BV = expected["BV"].astype("float")

        expected_MV = expected_vmag - 5.0 * (np.log10(expected_distance.value) - 1.0)

        np.testing.assert_equal(expected_names, HIP_list.Name)
        np.testing.assert_equal(expected_names, HIP_text.Name)

        # prepare skycoord arrays for testing.
        exp_coords_array = []
        list_coords_array = []
        text_coords_array = []

        for i in range(len(expected_coords)):
            exp_coords_array.append(
                [expected_coords[i].ra.degree, expected_coords[i].dec.degree]
            )

            list_coords_array.append(
                [HIP_list.coords[i].ra.degree, HIP_list.coords[i].dec.degree]
            )

            text_coords_array.append(
                [HIP_list.coords[i].ra.degree, HIP_list.coords[i].dec.degree]
            )

        # test skycoords RA and DEC
        np.testing.assert_allclose(exp_coords_array, list_coords_array)
        np.testing.assert_allclose(exp_coords_array, text_coords_array)

        # test spectra ID. assume we are ignoring the "C" part of classifications
        np.testing.assert_equal(expected_spec, HIP_list.Spec)
        np.testing.assert_equal(expected_spec, HIP_text.Spec)

        # test distance
        np.testing.assert_allclose(expected_distance, HIP_list.dist)
        np.testing.assert_allclose(expected_distance, HIP_text.dist)

        # test vmag
        np.testing.assert_allclose(expected_vmag, HIP_list.Vmag)
        np.testing.assert_allclose(expected_vmag, HIP_text.Vmag)

        # test mv
        np.testing.assert_allclose(expected_MV, HIP_list.MV)
        np.testing.assert_allclose(expected_MV, HIP_text.MV)
