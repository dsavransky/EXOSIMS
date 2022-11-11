import inspect
from typing import List, Dict
import re


def get_all_args(mod: type) -> List[str]:
    """Return list of all arguments to inits of every base of input class

    Args:
        mod (type):
            Class of object of interest

    Returns
        list:
            List of all arguments to mod.__init__()

    """

    kws = []
    bases = inspect.getmro(mod)
    for b in bases:
        if (b != object) and hasattr(b, "__init__"):
            kws += inspect.getfullargspec(b.__init__).args  # type: ignore

    return kws


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
