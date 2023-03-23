# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from EXOSIMS.Prototypes.StarCatalog import StarCatalog
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

v = Vizier(columns=["Plx", "B-V", "Hpmag"], catalog="I/311/hip2")
# Simbad.reset_votable_fields()
Simbad.add_votable_fields(
    "typed_id",  # queries value (i.e. HP)
    "flux(V)",  # m_V
    "flux(B)",  # m_B
    "flux(R)",  # m_R
    "flux(I)",  # m_I
    "flux(H)",  # m_K
    "flux(J)",  # m_K
    "flux(K)",  # m_K
    "distance",  # parsecs
    "flux_bibcode(V)",  # flux citation
    "flux_error(V)",  # v-band uncertainty
    "sp",  # spectral type
)


class HIPfromSimbad(StarCatalog):
    """
    Catalog generator class that uses astroquery to get stellar properties from SIMBAD

    Sonny Rappaport, August 2021: Fixed several typos.

    """

    def __init__(self, catalogpath=None, **specs):
        """

        Args:
            HIP (list or string):
                List of Hipparcos identifiers (HIP numbers) or path to text file.

        Example file format:

            ```HIP 37279```
            ```HIP 97649```

        """

        if catalogpath is None:
            raise ValueError("catalogpath keyword must be specified for HIPfromSimbad")
            # classpath = os.path.split(inspect.getfile(self.__class__))[0]
            # filename = 'hip.csv'
            # catalogpath = os.path.join(classpath, filename)

        if isinstance(catalogpath, str):
            HIP = np.loadtxt(catalogpath, delimiter=",", dtype="str")

            if HIP[0][:3] != "HIP":
                raise ValueError(
                    "First value in list is not explicitly an HIP Identifier"
                )
            HIP_names = [HIP[i] for i in range(len(HIP))]
        elif isinstance(catalogpath, list):
            HIP_names = ["HIP " + str(catalogpath[i]) for i in range(len(catalogpath))]
        else:
            raise ValueError(
                (
                    "Input is neither a list of integers nor a path to a list of "
                    "HIP identifier strings"
                )
            )
        print(HIP_names)
        # catalogpath = pkg_resources.resource_filename('EXOSIMS.StarCatalog',
        #                                                catalogfile)
        #
        # if not os.path.exists(catalogpath):
        #    raise IOError('Catalog File %s Not Found.'%catalogpath)

        StarCatalog.__init__(self, ntargs=len(HIP_names), **specs)
        simbad_list = Simbad.query_objects(HIP_names)
        BV = []

        # fill in distances
        for i, targ in enumerate(simbad_list["Distance_distance"]):
            try:
                result = v.query_object(simbad_list["TYPED_ID"][i])["I/311/hip2"]
                d = 1000 / result["Plx"]
                simbad_list["Distance_distance"][i] = d.data.data[0]
                simbad_list["Distance_method"][i] = "hip2"
                BV.append(result["B-V"].data.data[0])
                simbad_list["FLUX_V"][i] = result["Hpmag"].data.data[0]
            except Exception as err:
                print("simbad_list" + simbad_list["TYPED_ID"][i])
                print("Exception returned in Vizier Query for query:")
                print(err)
                d = np.nan

        data = simbad_list
        # Distance to the planetary system in units of parsecs
        self.dist = simbad_list["Distance_distance"].data.data * u.pc
        # print(simbad_list['RA'].data.data)
        self.coords = SkyCoord(
            ra=simbad_list["RA"].data.data,
            dec=simbad_list["DEC"].data.data,
            distance=self.dist,
            unit=(u.hourangle, u.deg, u.pc),
        )
        # Right Ascension of the planetary system in decimal degrees
        # Declination of the planetary system in decimal degrees
        # self.pmra = data['st_pmra'].data*u.mas/u.yr
        # Angular change in right ascension over time as seen from the center of
        # mass of the Solar System, units (mas/yr)
        # self.pmdec = data['st_pmdec'].data*u.mas/u.yr #Angular change in declination
        # over time as seen from the center of mass of the Solar System, units (mas/yr)
        self.L = np.empty(data["SP_TYPE"].size)
        self.L[:] = np.nan
        # data['st_lbol'].data
        # Amount of energy emitted by a star per unit time, measured in units of solar
        # luminosities. The bolometric corrections are derived from V-K or B-V colors,
        # units [log(solar)]

        # list of non-astropy attributes
        self.Name = np.array(
            HIP_names
        )  # Name of the star as given by the Hipparcos Catalog.
        self.Spec = np.array(data["SP_TYPE"]).astype(str)
        # Classification of the star based on their spectral characteristics following
        # the Morgan-Keenan system
        self.Vmag = np.array(data["FLUX_V"].data.data)  # V mag
        self.Jmag = np.array(data["FLUX_J"].data.data)  # Stellar J Magnitude Value
        self.Hmag = np.array(data["FLUX_H"].data.data)  # Stellar H  Magnitude Value
        self.Imag = np.array(data["FLUX_I"].data.data)  # Stellar I Magnitude Value
        self.Bmag = np.array(data["FLUX_B"].data.data)
        self.Kmag = np.array(data["FLUX_K"].data.data)
        self.BV = np.array(BV)
        # data['BV'] #Color of the star as measured by the difference between B and V
        # bands, units of [mag]

        # absolute V mag
        self.MV = self.Vmag - 5.0 * (np.log10(self.dist.to("pc").value) - 1.0)
        # self.Teff =  data['st_teff']
        # st_mbol Apparent magnitude of the star at a distance of 10 parsec
        # units of [mag]
        # self.BC = -self.Vmag + data['st_mbol'] # bolometric correction
        # self.stellar_diameters = data['st_rad']*2.*R_sun # stellar_diameters
        # in solar diameters
        # self.Binary_Cut = ~data['wds_sep'].mask #WDS (Washington Double Star) C
        # atalog separation (arcsecs)
        # save original data
        self.data = np.array(data)
