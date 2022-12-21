import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# Creating plots to look at observations visually
def plot_obs(SS, systemParams, mode, sInd, pInds, SNR, detected):
    """
    Plots the planet location in separation-dMag space over the completeness PDF.
    Useful for checking scaling and visualizing why a target observation failed. The
    plots are saved in the according to the simulation's seed in the plot_obs folder.

    Args:
        SS (SurveySimulation module):
            SurveySimulation class object
        systemParams (dict):
            Dictionary of time-dependant planet properties averaged over the
            duration of the integration
        mode (dict):
            Selected observing mode for detection
        sInd (integer):
            Integer index of the star of interest
        pInds (list of integers):
            Index values of the planets of interest
        SNR (float ndarray):
            Detection signal-to-noise ratio of the observable planets
        detected (integer ndarray):
            Detection status for each planet orbiting the observed target star:
            1 is detection, 0 missed detection, -1 below IWA, and -2 beyond OWA
    """
    try:
        SS.counter += 1
    except:
        SS.counter = 0
    TL = SS.TargetList
    Comp = SS.Completeness
    SU = SS.SimulatedUniverse
    dMag = systemParams["dMag"]
    WA = systemParams["WA"]
    dMag_range = np.linspace(16, 35, 5)
    # Making custom colormap for unnormalized values
    dMag_norm = mpl.colors.Normalize(vmin=dMag_range[0], vmax=dMag_range[-1])
    IWA = mode["IWA"]
    OWA = mode["OWA"]
    FR = 10 ** (dMag / (-2.5))

    x_Hrange = Comp.xnew
    y_Hrange = Comp.ynew
    H = Comp.Cpdf
    distance = TL.dist[sInd]
    int_comp = TL.int_comp[sInd]
    intCutoff_comp = TL.intCutoff_comp[sInd]
    saturation_comp = TL.saturation_comp[sInd]
    L = TL.L[sInd]

    smin = np.tan(IWA.to(u.rad)) * distance.to(u.AU) / np.sqrt(L)
    smax = np.tan(OWA.to(u.rad)) * distance.to(u.AU) / np.sqrt(L)
    WA = WA / np.sqrt(L)
    dMag -= 2.5 * np.log10(L)

    int_dMag = TL.int_dMag[sInd]
    scaled_int_dMag = int_dMag - 2.5 * np.log10(L)
    int_WA = TL.int_WA[sInd]
    scaled_int_WA = int_WA / np.sqrt(L)
    s_int = np.tan(scaled_int_WA.to(u.rad)) * distance.to(u.AU)
    # x_Hrange = x_Hrange * np.sqrt(L)
    # y_Hrange = y_Hrange + np.log10(L)

    my_cmap = plt.get_cmap("viridis")
    edge_cmap = plt.get_cmap("plasma")
    fig, ax = plt.subplots(figsize=[9, 9])
    extent = [x_Hrange[0], x_Hrange[-1], y_Hrange[0], y_Hrange[-1]]
    levels = np.logspace(-6, -1, num=30)
    H_scaled = H / 10000
    ax.contourf(
        H_scaled,
        levels=levels,
        cmap=my_cmap,
        origin="lower",
        extent=extent,
        norm=mpl.colors.LogNorm(),
    )
    # ax.plot(wa_range, dmaglims, c='r') # Contrast curve line
    FR_norm = mpl.colors.LogNorm()
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=FR_norm)
    sm._A = []
    sm.set_array(np.logspace(-6, -1))
    fig.subplots_adjust(left=0.15, right=0.85)
    cbar_ax = fig.add_axes([0.865, 0.125, 0.02, 0.75])
    cbar = fig.colorbar(sm, cax=cbar_ax, label=r"Normalized Density")
    # ax.plot(wa_range, dmaglims, c='r')
    # populate detection status array
    # 1:detected, 0:missed, -1:below IWA, -2:beyond OWA
    det_dict = {1: "detected", 0: "Missed", -1: "below_IWA", -2: "beyond_OWA"}
    WA = WA.flatten()
    det_str = ""
    ax.scatter(
        s_int.to(u.AU).value,
        scaled_int_dMag,
        color=edge_cmap(0),
        s=50,
        label="Value used to calculate integration time",
    )
    for i, pInd in enumerate(pInds):
        # color = my_cmap(dMag_norm(dMag[i]))
        s_i = np.tan(WA[i].to(u.rad)) * distance.to(u.AU)
        s_nom = np.tan(SU.WA[pInd].to(u.rad)) * distance.to(u.AU)
        detection_status = det_dict[detected[i]]
        det_str += str(i) + "_" + detection_status
        color = edge_cmap((i + 1) / (len(pInds) + 1))
        ax.scatter(
            s_i.to(u.AU).value,
            dMag[i],
            s=100,
            label=f"Planet: {pInd},\
                   SNR: {SNR[i]:.2f}",
            color=color,
        )
        # ax.axvline(s_nom.to(u.AU).value, color='k', label=f'Nominal s for planet {pInd}')
    ax.set_title(
        f"int_comp: {int_comp:.2f}, intCutoff_comp: {intCutoff_comp:.2f},\
                 saturation_comp: {saturation_comp:.2f}"
    )
    ax.set_xlim(0, 3)
    ax.set_ylim(dMag_range[0], dMag_range[-1])
    ax.set_xlabel("s (AU)")
    ax.set_ylabel("dMag")
    ax.axvline(x=smin.to(u.AU).value, color="k", label="Min s (IWA)")
    ax.axvline(x=smax.to(u.AU).value, color="k", label="Max s (OWA)")
    ax.axhline(
        y=TL.saturation_dMag[sInd] - 2.5 * np.log10(L),
        color=my_cmap(0),
        label="saturation_dMag",
    )
    ax.axhline(
        y=TL.intCutoff_dMag[sInd] - 2.5 * np.log10(L),
        color=my_cmap(0.5),
        label="intCutoff_dMag",
    )
    ax.axhline(
        y=TL.int_dMag[sInd] - 2.5 * np.log10(L), color=my_cmap(1), label="int_dMag"
    )
    # pos = ax.get_position()
    # ax.set_position([pos.x0, pos.y0, pos.width*0.2, pos.height])
    # ax.legend(loc='center right', bbox_to_anchor=(-.25, 0.5))
    ax.legend()
    folder = Path(f"plot_obs/{SS.seed:.0f}/")
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(folder, f"sInd_{sInd}_obs_{SS.counter}_status_{det_str}.png"))
    plt.close()
