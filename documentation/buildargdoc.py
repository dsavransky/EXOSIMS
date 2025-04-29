"""
Construct a list of every prototype argument with appropriate links to their
documentation.

This utility may only be used when EXOSIMS is installed from the git repo in editable
mode.
"""

from EXOSIMS.util.keyword_fun import get_allmod_args
import EXOSIMS.MissionSim
import os

if __name__ == "__main__":

    specs = {
        "modules": {
            "PlanetPopulation": " ",
            "StarCatalog": " ",
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

    sim = EXOSIMS.MissionSim.MissionSim(**specs)
    argdict = get_allmod_args(sim)
    args = sorted(list(argdict.keys()), key=str.casefold)

    docpath = os.path.abspath(
        os.path.join(
            os.path.split(EXOSIMS.MissionSim.__file__)[0],
            "..",
            "documentation",
        )
    )
    with open(os.path.join(docpath, "arglistpreamble.txt"), "r") as f:
        preamble = f.readlines()

    with open(os.path.join(docpath, "arglist.rst"), "w") as f:
        f.writelines(preamble)

        for arg in args:
            f.write("    * - ``{}``\n".format(arg))
            f.write(
                "      - {}\n".format(
                    ", ".join([":py:class:`~{}`".format(a) for a in argdict[arg]])
                )
            )
