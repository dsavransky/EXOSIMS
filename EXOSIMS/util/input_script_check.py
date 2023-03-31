from EXOSIMS.util.get_module import get_module
import EXOSIMS.MissionSim
import json
from typing import List, Dict, Union, Any, Tuple
from EXOSIMS.util.keyword_fun import get_all_mod_kws, check_opticalsystem_kws
import argparse
import copy


def parse_mods(specs: Dict[str, Any]) -> Dict[str, type]:
    """Check for presence of all required modules in input specs and return list of
    module class types.

    Args:
        specs (str or dict):
            Either full path to JSON script or an :ref:`sec:inputspec` dict

    Returns:
        dict:
            dict of all module classes along with MissionSim
    """

    req_mods = [
        "StarCatalog",
        "PlanetPopulation",
        "PlanetPhysicalModel",
        "OpticalSystem",
        "ZodiacalLight",
        "BackgroundSources",
        "PostProcessing",
        "Completeness",
        "TargetList",
        "SimulatedUniverse",
        "Observatory",
        "TimeKeeping",
        "SurveySimulation",
        "SurveyEnsemble",
    ]

    assert "modules" in specs, "Input specs missing modules keyword."
    mods = {}
    for k in req_mods:
        assert k in specs["modules"], f"Module {k} missing from input specs."
        mods[k] = get_module(specs["modules"][k], k, silent=True)

    mods["MissionSim"] = EXOSIMS.MissionSim.MissionSim

    return mods


def check_for_unused_kws(
    specs: Union[Dict[str, Any], str]
) -> Tuple[List[str], Dict[str, Any]]:
    """Check input specification for consistency with module inputs

    Args:
        specs (str or dict):
            Either full path to JSON script or an :ref:`sec:inputspec` dict

    Returns:
        tuple:
            unused (list):
                List of unused keywords
            specs (dict):
                Original input (useful if read from disk)
    """

    if isinstance(specs, str):
        with open(specs, "r") as f:
            specs = json.loads(f.read())

    mods = parse_mods(specs)
    allkws, allkwmods, ukws, ukwcounts = get_all_mod_kws(mods)

    unused = list(set(specs.keys()) - set(ukws))
    if "modules" in unused:
        unused.remove("modules")

    return unused, specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check an input script for spurious entries."
    )
    parser.add_argument("path", nargs=1, type=str, help="Input script path (str).")
    args = parser.parse_args()

    # look for unused kws
    unused, specs = check_for_unused_kws(args.path[0])
    if len(unused) > 0:
        print(
            "\nThe following input keywords were not used in any "
            "module init:\n\t{}".format("\n\t".join(unused))
        )

    # now check the optical system
    try:
        OS = get_module(
            specs["modules"]["OpticalSystem"], "OpticalSystem", silent=True
        )(**copy.deepcopy(specs))
        out = check_opticalsystem_kws(specs, OS)
        if out != "":
            print(f"\n{out}")
    except:  # noqa: E722
        print(
            "Could not instantiate OpticalSystem with this script, "
            "likely due to missing files."
        )
