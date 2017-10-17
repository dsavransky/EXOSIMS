.. _utils:

Utilities
####################

EXOSIMS provides multiple utilities for mission construction and ensemble analysis.  Some of these are accessible via methods in the ``MissionSim`` object, and some as standalone modules.

MissionSim Utilities
========================


genWaypoint
--------------

Generates a ballpark estimate of the expected number of star visits and
the total completeness of these visits for a given mission duration
        
    Args:
        duration (int):
            The length of time in days allowed for the waypoint calculation, defaults to 365
        tofile (string):
            Name of the file containing a plot of total completeness over mission time,
            by default genWaypoint does not create this plot

    Returns:
        out (dictionary):
            Output dictionary containing the number of stars visited, the total completness
            achieved, and the amount of time spent integrating.

genWaypoint is intended to be run prior to a simulation to provide a general idea of what to expect
within the simulation. By default, genWaypoint outputs a structure that looks like::

    {'numStars': 191, 'Total Completeness': 88.439895817937568, 'Total intTime': <Quantity 362.4756365032982 d>}

containing the number of stars visited, the total completeness of all the stars visited, and the total 
time spent integrating, which is bounded by the duration.

If "tofile" is specified, genWaypoint also generates a graph of total completeness over total integration time.

To run genWaypoint::

    sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
    sim.genWaypoint(tofile="mygraph.png")

    OR

    sim.genWaypoint(duration=730, tofile="mygraph.png")


checkScript
--------------------

Calls CheckScript and checks the script file against the mission outspec.
        
    Args:
        scriptfile (string):
            The path to the scriptfile being used by the sim
        prettyprint (boolean):
            Outputs the results of Checkscript in a readable format.
        tofile (string):
            Name of the file containing all output specifications (outspecs).
            Default to None.
            
    Returns:
        out (String):
            Output string containing the results of the check.

checkScript takes in a scriptfile and examines it in comparison to the mission outspec. It identifies any 
inconsitancies it finds between the two. The possible warnings are::

    WARNING 1: Catches parameters that are never used in the sim or are not in the outspec
    WARNING 2: Catches parameters that are unspecified in the script file and notes default value used
    WARNING 3: Catches mismatches in the modules being imported
    WARNING 4: Catches cases where the value in the script file does not match the value in the outspec

checkScript has several output options. By default, it will retunr a string containing all information. If 
"prettyprint" is specified, then checkscript will output this information to the commandline in a readble format. 
if "tofile" is specified, then the method will save this information to a file.

To run checkScript::

    sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
    sim.checkScript(scriptfile)

    OR

    sim.checkScript(scriptfile, prettyprint=True)

    OR

    sim.checkScript(scriptfile, tofile="check.txt")


