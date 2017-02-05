def run_one():

    sim.run_sim()
    res = sim.DRM[:]
    sim.DRM = []
    sim.TimeKeeping.__init__(**sim.TimeKeeping._outspec)
    sim.SimulatedUniverse.gen_physical_properties(**sim.SimulatedUniverse._outspec)
    sim.SimulatedUniverse.init_systems()

    return res


