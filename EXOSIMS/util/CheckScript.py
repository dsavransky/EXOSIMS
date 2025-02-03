import json
import os.path
import numpy as np
from EXOSIMS.util.vprint import vprint
import sys


class CheckScript(object):
    """
    Class that facilitates the comparison of the input script fiel for EXOSIMS and the
    outspec for a simulation. CheckScript highlights any differences between the two.

    Args:
        scriptfile (str):
            Full path to scriptfile
        outspec (dict):
            outspec dictionary

    """

    def __init__(self, scriptfile, outspec):
        self.outspec = outspec
        if scriptfile is not None:
            assert os.path.isfile(scriptfile), "%s is not a file." % scriptfile
            try:
                with open(scriptfile, "r") as f:
                    script = f.read()
                self.specs_from_file = json.loads(script)
            except ValueError as err:
                vprint(
                    "Error in CheckScript: Input file `%s' improperly formatted."
                    % (scriptfile)
                )
                vprint("Error: JSON error was: %s" % (err))
                # re-raise here to suppress the rest of the backtrace.
                # it is only confusing details about the bowels of json.loads()
                raise ValueError(err)
            except:  # noqa: E722
                vprint("Error in CheckScript: %s" % (sys.exc_info()[0]))
                raise
        else:
            self.specs_from_file = {}

    def recurse(self, json1, json2, pretty_print=False, recurse_level=0, outtext=""):
        """
        This function iterates recursively through the JSON structures of the script
        file and the simulation outspec, checking them against one another. Outputs the
        following warnings:

        #. Catches parameters that are never used in the sim or are not in the outspec
        #. Catches parameters that are unspecified in the script file and notes default
           value used
        #. Catches mismatches in the modules being imported
        #. Catches cases where the value in the script file does not match the value in
           the outspec

        Args:
            json1 (dict):
                The scriptfile json input.
            json2 (dict):
                The outspec json input
            pretty_print (boolean):
                Write output to a single return string rather than sequentially
            recurse_level (int):
                The current level of recursion
            outtext (string):
                The concatinated output text
        Returns:
            outtext (string):
                The concatinated output text
        """

        unused = np.setdiff1d(list(json1), list(json2))
        unspecified = np.setdiff1d(list(json2), list(json1))
        both_use = np.intersect1d(list(json1), list(json2))
        text_buffer = "  " * recurse_level

        # Check for unused fields
        for spec in unused:
            out = text_buffer + "WARNING 1: {} is not used in simulation".format(spec)
            if pretty_print:
                vprint(out)
            outtext += out + "\n"

        # Check for unspecified specs
        for spec in unspecified:
            out = (
                f"{text_buffer}WARNING 2: {spec} is unspecified in script, "
                f"using default value: {json2[spec]}"
            )
            if pretty_print:
                vprint(out)
            outtext += out + "\n"

        # Loop through full json structure
        for jkey in both_use:
            items = json1[jkey]
            # Check if there is more depth to JSON
            if jkey == "modules":
                out = "NOTE: Moving down a level from key: {}".format(jkey)
                if pretty_print:
                    vprint(out)
                outtext += out + "\n"
                for mkey in json2[jkey]:
                    if (
                        json1[jkey][mkey] != json2[jkey][mkey]
                        and json1[jkey][mkey] != " "
                        and json1[jkey][mkey] != ""
                    ):
                        out = (
                            "  WARNING 3: module {} from script file does not match "
                            "module {} from simulation"
                        ).format([json1[jkey][mkey]], [json2[jkey][mkey]])
                        if pretty_print:
                            vprint(out)
                        outtext += out + "\n"
                    elif json1[jkey][mkey] == " " or json1[jkey][mkey] == "":
                        out = (
                            "  NOTE: Script file does not specify module, "
                            "using default: {}"
                        ).format([json2[jkey][mkey]])
                        if pretty_print:
                            vprint(out)
                        outtext += out + "\n"
            elif isinstance(json1[jkey], dict) and isinstance(json2[jkey], dict):
                if "name" in json1[jkey]:
                    out = "NOTE: Moving down a level from key: {} to {}".format(
                        jkey, json1[jkey]["name"]
                    )
                else:
                    out = "NOTE: Moving down a level from key: {}".format(jkey)
                if pretty_print:
                    vprint(out)
                outtext += out + "\n"
                outtext = self.recurse(
                    json1[jkey],
                    json2[jkey],
                    pretty_print=pretty_print,
                    recurse_level=recurse_level + 1,
                    outtext=outtext,
                )
            else:
                try:
                    for i in range(len(items)):
                        if json1[jkey][i] != json2[jkey][i]:
                            if isinstance(json1[jkey][i], dict) and isinstance(
                                json2[jkey][i], dict
                            ):
                                if "name" in json1[jkey][i]:
                                    out = (
                                        "NOTE: Moving down a level from key: "
                                        "{} to {}"
                                    ).format(jkey, json1[jkey][i]["name"])
                                else:
                                    out = (
                                        "NOTE: Moving down a level from key: {}".format(
                                            jkey
                                        )
                                    )
                                if pretty_print:
                                    vprint(out)
                                outtext += out + "\n"
                                outtext = self.recurse(
                                    json1[jkey][i],
                                    json2[jkey][i],
                                    pretty_print=pretty_print,
                                    recurse_level=recurse_level + 1,
                                    outtext=outtext,
                                )
                            else:
                                out = (
                                    f"{text_buffer}WARNING 4: {jkey} in script file "
                                    "does not match spec in simulation: "
                                    f"(Script {jkey}:{json1[jkey]}, "
                                    f"Simulation {jkey}:{json2[jkey]})"
                                )
                                if pretty_print:
                                    vprint(out)
                                outtext += out + "\n"
                # Make sure script file matches with sim
                except TypeError:
                    if json1[jkey] != json2[jkey]:
                        out = (
                            f"{text_buffer}WARNING 4: {jkey} in script file does not "
                            "match spec in simulation: "
                            f"(Script {jkey}:{json1[jkey]}, "
                            "Simulation {jkey}:{json2[jkey]})"
                        )
                        if pretty_print:
                            vprint(out)
                        outtext += out + "\n"
        return outtext

    def write_file(self, filename):
        """Write output to disk

        Args:
            filename (str):
                Full path to output file

        """
        outtext = self.recurse(self.specs_from_file, self.outspec, pretty_print=True)
        with open(filename, "w") as outfile:
            outfile.write(outtext)
