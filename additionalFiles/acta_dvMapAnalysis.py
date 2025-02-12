import os.path
import numpy as np
import pickle
import astropy.units as u
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=0, vmax=24)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)

# Toggle lines 13-15 for a different orbit type
#T = np.array(["8.3257", "7.8471", "8.208", "8.8263", "9.4806", "10.1505", "11.5073", "12.0652", "12.0442", "11.9228"])  # L1
T = np.array(["7.0958","9.1699","11.241","13.0486","15.1067","17.1515","19.1719","21.1419","23.2099","25.183"])     # DRO
#T = np.array(["6.0066", "6.9808", "7.9736", "8.8975", "10.0002", "11.0244", "12.05", "13.0656", "14.0171", "14.6949"])  # L2

# distances for comparing initial target star
d = np.array(["7500", "10000"])

# distances for comparing minimum delta-v at different separation distances
dd = np.array(["0.001", "1", "100", "1000", "7500", "20000", "40000", "60000", "80000", "100000", "120000"])

# distances for comparing reachable targets at different distances
ddd = np.array(["7500", "20000", "40000", "60000", "80000", "100000", "120000"])


# vary orbit period at two different target stars for a single separation distance
ind = 0
factor = 2
fig, axs = plt.subplots(1,2)
for ii in T:
    ctr = 1
    nArray1 = np.array([])
    nArray2 = np.array([])

    width = 1/12
    for jj in d:
        path_str1 ="$HOME/.EXOSIMS/cache_lunarSSDRO_12172024/dVMap_" + jj + ".0_DRO_" + ii + "_days.dVmap"
#        path_str1 ="$HOME/.EXOSIMS/cache_lunarSSL1S_12172024/dVMap_" + jj + ".0_L1_S_" + ii + "_days.dVmap"
#        path_str1 ="$HOME/.EXOSIMS/cache_lunarSSL2N_12172024/dVMap_" + jj + ".0_L2_N_" + ii + "_days.dVmap"
        path_f1 = os.path.normpath(os.path.expandvars(path_str1))
        f1 = open(path_f1, "rb")
        tmp1 = pickle.load(f1)
        dVMap1 = tmp1["dVMap"]

        loc20_1 = np.argwhere(dVMap1 < 20)
        nInds1 = len(np.unique(loc20_1[:,1]))

        nArray1 = np.append(nArray1,nInds1)
        
        path_str2 ="$HOME/.EXOSIMS/cache_lunarSSDRO_12172024_72/dVMap_" + jj + ".0_DRO_" + ii + "_days.dVmap"
#        path_str2 ="$HOME/.EXOSIMS/cache_lunarSSL1S_12172024_72/dVMap_" + jj + ".0_L1_S_" + ii + "_days.dVmap"
#        path_str2 ="$HOME/.EXOSIMS/cache_lunarSSL2N_12172024_72/dVMap_" + jj + ".0_L2_N_" + ii + "_days.dVmap"
        path_f2 = os.path.normpath(os.path.expandvars(path_str2))
        f2 = open(path_f2, "rb")
        tmp2 = pickle.load(f2)
        dVMap2 = tmp2["dVMap"]

        loc20_2 = np.argwhere(dVMap2 < 20)
        nInds2 = len(np.unique(loc20_2[:,1]))

        nArray2 = np.append(nArray2,nInds2)

    axs[0].bar(np.arange(len(nArray1)) + width*ind, nArray1, width, color = cmap.to_rgba((ind+1)*factor), label=ii + " days")
    axs[1].bar(np.arange(len(nArray2)) + width*ind, nArray2, width, color = cmap.to_rgba((ind+1)*factor), label=ii + " days")
    ind = ind + 1

axs[0].set_title("0$^\circ$ Lat, -70$^\circ$ Lon")
axs[0].set_xlabel("Separation distance [km]")
axs[0].set_ylabel("Number of unique targets")
axs[0].set_xticks(np.arange(len(d))+(5/12-1/24))
axs[0].set_xticklabels(d)

axs[1].set_title("0$^\circ$ Lat, 10$^\circ$ Lon")
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Orbit Period")
axs[1].set_xlabel("Separation distance [km]")
axs[1].set_xticks(np.arange(len(d))+(5/12-1/24))
axs[1].set_xticklabels(d)

# vary orbit period for two separation distances at a single initial target
ind = 0
factor = 2
fig = plt.figure()
ax = plt.subplot(111)
for ii in T:
    ctr = 1
    nArray = np.array([])

    width = 1/12
    for jj in ddd:
#        path_str ="$HOME/.EXOSIMS/cache_lunarSSL1S_10182024/dVMap_" + jj + ".0_L1_S_" + ii + "_days.dVmap"
        path_str ="$HOME/.EXOSIMS/cache_lunarSSDRO_10182024/dVMap_" + jj + ".0_DRO_" + ii + "_days.dVmap"
#        path_str ="$HOME/.EXOSIMS/cache_lunarSSL2N_12172024/dVMap_" + jj + ".0_L2_N_" + ii + "_days.dVmap"
        path_f1 = os.path.normpath(os.path.expandvars(path_str))
        f1 = open(path_f1, "rb")
        tmp = pickle.load(f1)
        dVMap = tmp["dVMap"]

        loc20 = np.argwhere(dVMap < 200)
        nInds = len(np.unique(loc20[:,1]))

        nArray = np.append(nArray,nInds)
    print(nArray)
    ax.bar(np.arange(len(nArray)) + width*ind, nArray, width, color = cmap.to_rgba((ind+1)*factor), label=ii + " days")
    ind = ind + 1

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Orbit Period")

plt.xlabel("Separation distance [km]")
plt.ylabel("Number of unique targets")
plt.xticks(np.arange(len(ddd))+(5/12-1/24), ddd)
ax.xaxis.set_ticks_position('none')

## vary separation distance for a single orbit
#plt.figure()
#markers = ["o" , "v" , "^" , "<", ">", "8", "s", "p", "P", "*"]
#ctr = 0
#for ii in T:
#    dvMins = np.array([])
#    
#    path_str ="$HOME/.EXOSIMS/cache_lunarSSL2S_12172024/dVMap_0.001_L2_S_12.05_days.dVmap"
#    path_f1 = os.path.normpath(os.path.expandvars(path_str))
#    f1 = open(path_f1, "rb")
#    tmp = pickle.load(f1)
#    dVMap = tmp["dVMap"]
#    dVflat = dVMap.flatten()
#    dVmin = min(dVflat)
#    
#    dvMins = np.append(dvMins, dVmin)
#    for jj in dd[1:]:
#        path_str ="$HOME/.EXOSIMS/cache_lunarSSL2S_12172024/dVMap_" + jj + ".0_L2_S_12.05_days.dVmap"
#        path_f1 = os.path.normpath(os.path.expandvars(path_str))
#        f1 = open(path_f1, "rb")
#        tmp = pickle.load(f1)
#        dVMap = tmp["dVMap"]
#        dVflat = dVMap.flatten()
#        dVmin = min(dVflat)
#        dvMins = np.append(dvMins, dVmin)
#plt.plot(dd.astype(float), dvMins, color = cmap.to_rgba(ctr+20), linewidth=2)
#plt.scatter(dd.astype(float), dvMins, color = cmap.to_rgba(ctr+20), linewidth=2, s = 100)
#ctr = ctr + 1
#
#xx = np.array([dd[0].astype(float), dd[-1].astype(float)])
#yy = np.array([20, 20])
#plt.plot(xx, yy, 'k--')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel("Separation distance [km]")
#plt.ylabel("Minimum Delta-v [m/s]")

plt.show()
breakpoint()
