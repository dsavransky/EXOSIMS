# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Distance
from EXOSIMS.Prototypes.StarCatalog import StarCatalog
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

v = Vizier(columns=["B-V", "Hpmag"], catalog="I/311/hip2")
simbad = Simbad()
simbad.add_votable_fields(
    "pmra", "pmdec", "plx_value", "rvz_radvel", "V", "B", "sp_type"
)


class HIPfromSimbad(StarCatalog):
    """
    Catalog generator class that uses astroquery to get stellar properties from SIMBAD

    Args:
        HIP (list or string):
            List of Hipparcos identifiers (HIP numbers) or path to text file.
        **specs:
            :ref:`sec:inputspec`

    Example file format:

        ```HIP 37279```
        ```HIP 97649```


    """

    def __init__(self, catalogpath=None, **specs):

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

        StarCatalog.__init__(self, ntargs=len(HIP_names), **specs)
        try:
            simbad_query = simbad.query_objects(HIP_names)
        except:  # noqa
            print("Initial query failed. Trying again.")
            simbad_query = simbad.query_objects(HIP_names)
        simbad_list = simbad_query.to_pandas()

        assert np.all(
            simbad_list["user_specified_id"].str.strip().values == np.array(HIP_names)
        ), "Simbad query returned unexpected number of objects."

        # fill in photometry
        missing_photom = simbad_list.loc[
            (simbad_list["B"].isna() | simbad_list["V"].isna())
        ]
        if len(missing_photom) > 0:
            for j, row in missing_photom.iterrows():
                try:
                    result = v.query_object(row["main_id"])["I/311/hip2"]
                    simbad_list.loc[simbad_list["main_id"] == row.main_id, "V"] = (
                        result["Hpmag"].data.data[0]
                    )
                    simbad_list.loc[simbad_list["main_id"] == row.main_id, "B"] = (
                        result["Hpmag"].data.data[0] + result["B-V"].data.data[0]
                    )
                except TypeError:
                    self.vprint(f"Vizier query failed for {row.user_specified_id}")

        # Distance and coordinates
        dist = Distance(
            parallax=simbad_list["plx_value"].values * simbad_query["plx_value"].unit
        )
        self.dist = (dist.value * dist.unit).to(u.pc)

        self.coords = SkyCoord(
            ra=simbad_list["ra"].values,
            dec=simbad_list["dec"].values,
            unit=(simbad_query["ra"].unit, simbad_query["dec"].unit),
            distance=self.dist,
        )

        # Proper motions
        self.pmra = simbad_list["pmra"].values * simbad_query["pmra"].unit
        self.pmdec = simbad_list["pmdec"].values * simbad_query["pmdec"].unit

        # allow TargetList to fill in luminosities
        self.L = np.zeros(len(HIP_names)) * np.nan

        # target Name
        self.Name = np.array(HIP_names)
        self.Spec = simbad_list["sp_type"].values.astype(str)
        self.Vmag = simbad_list["V"].values
        self.Bmag = simbad_list["B"].values
        self.BV = self.Bmag - self.Vmag

        # absolute V mag
        self.MV = self.Vmag - 5.0 * (np.log10(self.dist.to("pc").value) - 1.0)

        # save original data
        self.data = simbad_list
