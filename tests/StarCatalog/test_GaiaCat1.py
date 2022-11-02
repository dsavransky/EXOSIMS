import unittest
from tests.TestSupport.Utilities import RedirectStreams
import EXOSIMS.StarCatalog
from EXOSIMS.StarCatalog.GaiaCat1 import GaiaCat1
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


class TestGaiaCat1(unittest.TestCase):

    """
    Sonny Rappaport, July 2021, Cornell

    This class tests GaiaCat1's initialization.

    """

    def setUp(self):

        """
        test data source file is from the Gaia 2nd data release,
        with the following ADQL query used to gather the data:

        SELECT TOP 1000 gaia_source.source_id,gaia_source.ra,gaia_source.ra_error,
        gaia_source.dec,gaia_source.dec_error,gaia_source.parallax,
        gaia_source.parallax_error,
        gaia_source.astrometric_matched_observations,
        gaia_source.visibility_periods_used,gaia_source.phot_g_mean_mag,
        gaia_source.phot_bp_mean_mag,gaia_source.phot_rp_mean_mag,
        gaia_source.teff_val
        FROM gaiadr2.gaia_source
        WHERE gaia_source.source_id IS NOT NULL
                 AND gaia_source.ra IS NOT NULL
                 AND gaia_source.ra_error IS NOT NULL
                 AND gaia_source.dec IS NOT NULL
                 AND gaia_source.dec_error IS NOT NULL
                 AND gaia_source.parallax IS NOT NULL
                 AND gaia_source.parallax_error IS NOT NULL
                 AND gaia_source.astrometric_matched_observations IS NOT NULL
                 AND gaia_source.visibility_periods_used IS NOT NULL
                 AND gaia_source.phot_g_mean_mag IS NOT NULL
                 AND gaia_source.phot_bp_mean_mag IS NOT NULL
                 AND gaia_source.phot_rp_mean_mag IS NOT NULL
                 AND gaia_source.teff_val IS NOT NULL
        ORDER by gaia_source.source_id;

        copy the gaia sample datafile from test-scripts to the downloads folder,
        (if the gaia sample datafile isn't there already)
        """
        downloads_path = get_downloads_dir()
        if not os.path.exists(downloads_path + "/GaiaCatGVTest.gz"):
            shutil.copy(
                "tests/TestSupport/test-scripts/GaiaCatGVTest.gz", downloads_path
            )

        self.fixture = GaiaCat1(catalogfile="GaiaCatGVTest.gz")

    def test_init(self):

        """
        Test of initialization and __init__.

        Test method: Use the same dataset, but as a CSV file instead, and check
        that the GaiaCat1 object has stored the data correctly.
        """

        # nickname for the overall object
        gaia = self.fixture

        # same raw data from before, just in CSV format.
        expected = np.genfromtxt(
            "tests/TestSupport/test-scripts/GaiaCatCSVTest.csv",
            delimiter=",",
            names=True,
        )

        # test all prototype attributes
        np.testing.assert_allclose(expected["source_id"], gaia.Name)
        np.testing.assert_allclose(expected["teff_val"], gaia.Teff)
        np.testing.assert_allclose(expected["phot_g_mean_mag"], gaia.Gmag)
        np.testing.assert_allclose(expected["phot_bp_mean_mag"], gaia.BPmag)
        np.testing.assert_allclose(expected["phot_rp_mean_mag"], gaia.RPmag)
        np.testing.assert_allclose(expected["ra_error"], gaia.RAerr)
        np.testing.assert_allclose(expected["dec_error"], gaia.DECerr)
        np.testing.assert_allclose(expected["parallax_error"], gaia.parxerr)
        np.testing.assert_allclose(
            expected["astrometric_matched_observations"],
            gaia.astrometric_matched_observations,
        )
        np.testing.assert_allclose(
            expected["visibility_periods_used"], gaia.visibility_periods_used
        )
        exp_para_units = expected["parallax"] * u.mas
        np.testing.assert_allclose(exp_para_units, gaia.parx)
        exp_dist = exp_para_units.to("pc", equivalencies=u.parallax())
        np.testing.assert_allclose(exp_dist, gaia.dist)
        exp_coords = SkyCoord(
            ra=expected["ra"] * u.deg, dec=expected["dec"] * u.deg, distance=exp_dist
        )

        # prepare skycoord arrays for testing.
        exp_coords_array = []
        gaia_coords_array = []

        for i in range(len(exp_coords)):
            exp_coords_array.append(
                [
                    exp_coords[i].ra.degree,
                    exp_coords[i].dec.degree,
                    exp_coords[i].distance.pc,
                ]
            )
            gaia_coords_array.append(
                [
                    gaia.coords[i].ra.degree,
                    gaia.coords[i].dec.degree,
                    gaia.coords[i].distance.pc,
                ]
            )

        np.testing.assert_allclose(exp_coords_array, gaia_coords_array)

        # expected versions of these three parameters
        eGmag = expected["phot_g_mean_mag"]
        eBPmag = expected["phot_bp_mean_mag"]
        eRPmag = expected["phot_rp_mean_mag"]

        expected_vmag = eGmag - (
            -0.01760 - 0.006860 * (eBPmag - eRPmag) - 0.1732 * (eBPmag - eRPmag) ** 2
        )
        expected_rmag = eGmag - (
            -0.003226 + 0.3833 * (eBPmag - eRPmag) - 0.1345 * (eBPmag - eRPmag) ** 2
        )
        expected_Imag = eGmag - (
            0.02085 + 0.7419 * (eBPmag - eRPmag) - 0.09631 * (eBPmag - eRPmag) ** 2
        )

        # test these three parameters. seems to be small rounding imprecision,
        # so upped rtolerance slightly

        np.testing.assert_allclose(expected_vmag, gaia.Vmag, rtol=2e-7)
        np.testing.assert_allclose(expected_rmag, gaia.Rmag, rtol=2e-7)
        np.testing.assert_allclose(expected_Imag, gaia.Imag, rtol=2e-7)
