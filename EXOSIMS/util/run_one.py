def run_one():

    sim.run_sim()
    res = sim.DRM[:]
    sim.DRM = []
    sim.TimeKeeping.__init__(**sim.TimeKeeping._outspec)

    return res


