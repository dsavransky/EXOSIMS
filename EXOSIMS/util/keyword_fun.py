import inspect
from typing import List, Dict, Tuple, Any
import re
import numpy as np


def get_all_args(mod: type) -> List[str]:
    """Return list of all arguments to inits of every base of input class

    Args:
        mod (type):
            Class of object of interest

    Returns:
        list:
            List of all arguments to mod.__init__()

    """

    kws = []
    bases = inspect.getmro(mod)
    for b in bases:
        if (b is not object) and hasattr(b, "__init__"):
            kws += inspect.getfullargspec(b.__init__).args  # type: ignore

    return kws


def get_all_mod_kws(mods: Dict[str, type]) -> Tuple[List[str], List[str]]:
    """Collect all keywords from all modules

    Args:
        mods (dict):
            dict of all module classes along with MissionSim

    Returns:
        tuple:
            allkws (list):
                All keywords
            allkwmods (list):
                The names of the modules the keywords belong to.
            ukws (~numpy.ndarray(str)):
                Unique keywords (excluding self and scriptfile)
            ukwcounts (~numpy.ndarray(int)):
                Unique keyword counts

    """

    # collect keywords
    allkws = []
    allkwmods = []
    for modname in mods:
        tmp = get_all_args(mods[modname])
        allkws += tmp
        allkwmods += [mods[modname].__name__] * len(tmp)

    # pop things in whitelist and get unique args
    ukws = np.array(allkws)
    whitelist = ["self", "scriptfile"]
    for w in whitelist:
        ukws = ukws[ukws != w]
    ukws, ukwcounts = np.unique(ukws, return_counts=True)

    return allkws, allkwmods, ukws, ukwcounts


def get_allmod_args(sim) -> Dict[str, List[str]]:
    """Return list of all arguments to all inits of all modules in a MissionSim Object

    Args:
        sim (:py:class:`~EXOSIMS.MissionSim`):
            MissionSim object

    Returns:
        dict:
            Dictionary of all input arguments and the modules in which they appear
    """

    classp = re.compile(r"<class '(\S+)'>")

    allkws: Dict[str, List[str]] = {}
    allmods = [mod.__class__ for mod in sim.modules.values()]
    allmods.append(sim.__class__)
    if sim.TargetList.keepStarCatalog:
        allmods.append(sim.TargetList.StarCatalog.__class__)
    else:
        allmods.append(sim.TargetList.StarCatalog)

    for mod in allmods:
        classname = classp.match(str(mod)).group(1)  # type: ignore
        kws = get_all_args(mod)
        if "self" in kws:
            kws.remove("self")
        if "scriptfile" in kws:
            kws.remove("scriptfile")
        for kw in kws:
            if kw in allkws:
                allkws[kw].append(classname)
            else:
                allkws[kw] = [
                    classname,
                ]

    return allkws


def check_opticalsystem_kws(specs: Dict[str, Any], OS: Any):
    """Check input specification against an optical system object

    Args:
        specs (str or dict):
            Either full path to JSON script or an :ref:`sec:inputspec` dict
        OS (:ref:`OpticalSystem`):
            OpticalSystem object

    Returns:
        str:
            Description of what's wrong with the input (blank if its all good).

    """
    out = ""

    if "scienceInstruments" in specs:
        for inst in specs["scienceInstruments"]:
            extra_keys = list(set(inst.keys()) - set(OS.allowed_scienceInstrument_kws))
            if len(extra_keys) > 0:
                out += (
                    f"Unknown key(s): {', '.join(extra_keys)} for "
                    f"science instrument {inst['name']}\n"
                )

    if "starlightSuppressionSystems" in specs:
        for syst in specs["starlightSuppressionSystems"]:
            extra_keys = list(
                set(syst.keys()) - set(OS.allowed_starlightSuppressionSystem_kws)
            )
            if len(extra_keys) > 0:
                out += (
                    f"Unknown key(s): {', '.join(extra_keys)} for "
                    f"system {syst['name']}\n"
                )
            if ("core_mean_intensity" in syst) and ("core_platescale" not in syst):
                out += (
                    f"core_mean_intensity requires core_platescale to be set in "
                    f"system {syst['name']}\n"
                )

    if "observingModes" in specs:
        for nmode, mode in enumerate(specs["observingModes"]):
            extra_keys = list(set(mode.keys()) - set(OS.allowed_observingMode_kws))
            if "description" in extra_keys:
                extra_keys.remove("description")
            if len(extra_keys) > 0:
                out += (
                    f"Unknown key(s): {', '.join(extra_keys)} for "
                    f"observing mode #{nmode}\n"
                )

    return out
