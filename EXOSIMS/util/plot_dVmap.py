import EXOSIMS,os.path
#use whichever scriptfile you'd like
scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','template_SotoStarshade.json')
import EXOSIMS.MissionSim
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

obs = sim.Observatory

try:
    obs.__class__.__name__ == 'SotoStarshade'
except ValueError:
    sim.vprint('SotoStarshade not selected as Observatory in given script.')

#values of slew time and angular sep. stored in the 2D interpolant
input_dt  = obs.dV_interp.x

#making new arrays for the two independent variables
dtValues  = np.linspace( input_dt[0]  , input_dt[-1]  , 1000 )
angValues = np.linspace( -180         , 180           , 1000 )

#interpolated dV values
dVin  = obs.dV_interp(dtValues , angValues)
logN = np.log(dVin)

#creating colormap and figure, plotting log of dV
cMap = cm.hot_r
fig = plt.figure()
ax  = fig.gca()
img = plt.imshow(logN.T,cmap=cMap,interpolation='none',extent=[-180,180,input_dt[0],input_dt[-1]],origin='lower',aspect=4)

#making new logarithmic tick marks, kind of arbitrary
rng = np.arange(1.2,6.3,0.6)
rngTicks = np.e**rng
newTicks = ['%.2f' % round(rngTicks[x] * 1000 / 1000,3) for x in range(len(rngTicks))]

#labels and stuff
sm = plt.cm.ScalarMappable(cmap=cMap)
sm.set_array(logN)
cbar = plt.colorbar(sm)
cbar.set_label('$\Delta$v (m/s)')
cbar.ax.set_yticklabels(newTicks)
ax.set_yticks(np.arange(input_dt[0],input_dt[-1],10))
ax.set_xticks(np.arange(-180,181,60))
plt.show()   

#bold axes, or else it's not a real plot
fontsize=13
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
ax.set_xlabel('Star Angular Separation $\psi$ (deg)',fontsize=13,fontweight='bold')
ax.set_ylabel('Slew Time $\Delta$t (day)',fontsize=13,fontweight='bold')
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(13)
cbar.set_label('$\Delta$v (m/s)',fontsize=13,fontweight='bold')
