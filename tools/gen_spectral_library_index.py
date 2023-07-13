"""
This script generates the index of Pickles and BPGS spectral files for all available
spectral types. It shoudl not be needed for normal EXOSIMS operations, but is available
if this index ever needs to be regenerated in the future.
"""
import pkg_resources
from astropy.io import fits
from MeanStars import MeanStars
import os
import numpy as np
import pandas
import json
from EXOSIMS.Prototypes.TargetList import TargetList

ms = MeanStars()
specdict = {"O": 0, "B": 1, "A": 2, "F": 3, "G": 4, "K": 5, "M": 6}

pickles_path = pkg_resources.resource_filename("EXOSIMS.TargetList", "dat_uvk")
bpgs_path = pkg_resources.resource_filename("EXOSIMS.TargetList", "bpgs")


# start with pickles atlas
with fits.open(os.path.join(pickles_path, "pickles_uk.fits")) as hdulist:
    pickles_spectra = hdulist[1].data

all_spectral_files = {}
used_files = []
for j, spec in enumerate(pickles_spectra["SPTYPE"]):
    # this atlas contains two redundant spectra that we don't need
    if spec in ["M2.5V", "M10III"]:
        continue
    tmp = ms.specregex.match(spec)
    if tmp:
        specclass = list(tmp.groups())

        # files with two character subclasses actually represent ranges
        if len(specclass[1]) > 1:
            subclass = range(int(specclass[1][0]), int(specclass[1][1]) + 1)
        else:
            subclass = [int(specclass[1])]

        # write all required entries to dict
        for sc in subclass:
            all_spectral_files[f"{specclass[0]}{sc}{specclass[2]}"] = {
                "file": f"{pickles_spectra['FILENAME'][j]}.fits",
                "specclass": [
                    specclass[0],
                    int(sc),
                    specclass[2],
                    specdict[specclass[0]] * 10 + sc,
                ],
            }
        used_files.append(f"{pickles_spectra['FILENAME'][j]}.fits")

# identify files that can be removed
unused_files = [
    os.path.join(pickles_path, fn)
    for fn in list(
        set([f"{f}.fits" for f in pickles_spectra["FILENAME"]]) - set(used_files)
    )
]

# now bpgs
bpgs_spectra = pandas.read_table(
    os.path.join(bpgs_path, "BPGS_README"),
    delim_whitespace=True,
    skiprows=3,
    names=["file", "target", "type"],
)
# throw away those without spectral types
unused_files += [
    os.path.join(
        bpgs_path,
        f"{fn.split('.')[0]}.fits",
    )
    for fn in bpgs_spectra.loc[bpgs_spectra["type"].isnull(), "file"].values
]
bpgs_spectra = bpgs_spectra.loc[bpgs_spectra["type"].notnull()]

# find new spectral types
tmp = []
for t in bpgs_spectra["type"].values:
    if ms.specregex.match(t):
        tmp.append(t)
newspecs = list(set(tmp) - set(all_spectral_files.keys()))
# remove any non-standard or supergiants types
newspecs = [n for n in newspecs if (n.endswith("I") or n.endswith("V"))]
tmp = np.array(tmp)

used_files2 = []
for s in newspecs:
    # in cases where there are multiple matches (types G6IV and A1V), pick the last
    # available entry
    row = bpgs_spectra.loc[bpgs_spectra["type"] == s].iloc[-1]
    specclass = list(ms.specregex.match(row.type).groups())
    specclass[1] = int(specclass[1])

    all_spectral_files[s] = {
        "file": f"{row.file.split('.')[0]}.fits",
        "specclass": specclass
        + [
            specdict[specclass[0]] * 10 + specclass[1],
        ],
    }
    used_files2.append(row.file)

unused_files += [
    os.path.join(
        bpgs_path,
        f"{fn.split('.')[0]}.fits",
    )
    for fn in list(set(bpgs_spectra["file"].values) - set(used_files2))
]

# # sort final dict
# all_spectral_files = dict(sorted(all_spectral_files.items(), key=lambda item: item[0]))

# write final index to disk
with open(
    os.path.join(
        pkg_resources.resource_filename("EXOSIMS", "TargetList"),
        "spectral_catalog_index.json",
    ),
    "w",
) as outfile:
    json.dump(all_spectral_files, outfile)

# remove unused files from directories
for f in unused_files:
    if os.path.exists(f):
        print(f"Removing {f}")
        os.remove(f)

# some final consistency checking
with open(
    os.path.join(
        pkg_resources.resource_filename("EXOSIMS", "TargetList"),
        "spectral_catalog_index.json",
    ),
    "r",
) as f:
    specindex = json.load(f)

for s in specindex:
    if specindex[s]["file"].startswith("pickles"):
        filepath = os.path.join(pickles_path, specindex[s]["file"])
    else:
        filepath = os.path.join(bpgs_path, specindex[s]["file"])
    assert os.path.exists(filepath)

specclasses = np.zeros((len(specindex), 4), dtype=object)
for j, s in enumerate(specindex):
    specclasses[j] = specindex[s]["specclass"]

# check missing from EXOCAT
specs0 = {
    "modules": {
        "PlanetPopulation": " ",
        "StarCatalog": "EXOCAT1",
        "OpticalSystem": " ",
        "ZodiacalLight": " ",
        "BackgroundSources": " ",
        "PlanetPhysicalModel": " ",
        "Observatory": " ",
        "TimeKeeping": " ",
        "PostProcessing": " ",
        "Completeness": " ",
        "TargetList": " ",
        "SimulatedUniverse": " ",
        "SurveySimulation": " ",
        "SurveyEnsemble": " ",
    },
    "scienceInstruments": [{"name": "imager"}],
    "starlightSuppressionSystems": [{"name": "coronagraph"}],
}

TL = TargetList(**specs0)
uspecs = np.unique(TL.Spec)
missing_specs = [us for us in uspecs if us not in specindex]

# show closest match
for s in missing_specs:
    tmp = ms.specregex.match(s).groups()
    tmp2 = specdict[tmp[0]] * 10 + float(tmp[1])
    # filter by luminosity class
    tmp3 = specclasses[specclasses[:, 2] == tmp[2]]
    # next get the closest numerical spectral class representation
    row = tmp3[np.argmin(np.abs(tmp3[:,3] - tmp2))]
    print(f"{s} matched to {row[0]}{row[1]}{row[2]}")
