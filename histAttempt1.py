#!/usr/bin/env python
# coding: utf-8

# In[1]:


import EXOSIMS
import EXOSIMS.MissionSim
import os
import matplotlib.pyplot as plt

def run_simulation(scriptfile):
    sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
    sim.run_sim()
    DRM = sim.SurveySimulation.DRM
    
    # Count number of unique detections in this run
    detections = 0
    for obs in DRM:
        if 'det_status' in obs:
            detections += sum(obs['det_status'])  # Number of detections in this obs
    return detections

def run_ensemble(scriptfile, N=50):
    yields = []
    for i in range(N):
        print(f"Running simulation {i+1}/{N}...")
        try:
            y = run_simulation(scriptfile)
            yields.append(y)
        except Exception as e:
            print(f"Simulation {i+1} failed: {e}")
    return yields

def plot_yield_histogram(yields):
    plt.figure(figsize=(8, 5))
    plt.hist(yields, bins=range(min(yields), max(yields)+2), edgecolor='black', align='left')
    plt.xlabel("Number of Detections")
    plt.ylabel("Number of Simulations")
    plt.title("Histogram of Exoplanet Detection Yields")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import os.path
    scriptfile = os.path.join(EXOSIMS.__path__[0], 'Scripts', 'sampleScript_coron.json')
    
    # Run ensemble
    yields = run_ensemble(scriptfile, N=50)  # Adjust N as needed

    # Plot
    plot_yield_histogram(yields)

