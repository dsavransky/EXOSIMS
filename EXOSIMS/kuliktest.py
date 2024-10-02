import EXOSIMS
import EXOSIMS.MissionSim
import os.path

scriptfile = os.path.join(
    EXOSIMS.__path__[0], "Scripts", "TestScripts/kulikscript.json"
)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

sim.run_sim()
