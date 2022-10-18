""""
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

    preamble = """.. _arglist:

    EXOSIMS Prototype Inputs
    ##########################

    EXOSIMS contains a large number of user-settable parameters (either via the
    input JSON script or passed directly to the constructors of various modules at
    instantiation).  All inputs have associated defaults that are automatically
    filled in if not set by the user.

    The table below includes a list of all Prototype module inputs.

    .. list-table:: Prototype Arguments
        :widths: 25 75
        :header-rows: 1

        * - Argument
          - Modules
    """

    fname = os.path.abspath(
        os.path.join(
            os.path.split(EXOSIMS.MissionSim.__file__)[0],
            "..",
            "documentation",
            "arglist.rst",
        )
    )
    with open(fname, "w") as f:
        f.write(preamble)

        for arg in args:
            f.write("    * - ``{}``\n".format(arg))
            f.write(
                "      - {}\n".format(
                    ", ".join([":py:class:`~{}`".format(a) for a in argdict[arg]])
                )
            )
