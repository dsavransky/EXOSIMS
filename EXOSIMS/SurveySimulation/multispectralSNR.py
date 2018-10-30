from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from EXOSIMS.SurveySimulation import *
import EXOSIMS, os
import astropy.units as u
import astropy.constants as const
import numpy as np
import itertools
from scipy import interpolate
try:
    import cPickle as pickle
except:
    import pickle
import time
import copy
from EXOSIMS.util.deltaMag import deltaMag

class multispectralSNR(SurveySimulation):

    def __init__(self, **specs):


    def calc_SNR(self, SNR, mode):
        # SNR CALCULATION:
        # first, calculate SNR for observable planets (without false alarm)
        planinds = pIndsChar[:-1] if pIndsChar[-1] == -1 else pIndsChar
        SNRplans = np.zeros(len(planinds))

        # then, calculate number of bins
        lam0 = mode['lam']
        R = mode['R']       # spectral resolution
        deltaLam = lam0/R
        BW0_min = lam0 * (1 - mode['BW']/2)
        BW0_max = lam0 * (1 + mode['BW']/2)
        lam = BW0_min + deltaLam/2
        num_SNR_bins = int((BW0_max - BW0_min)/deltaLam)
        minimodes = []

        for j in  range(num_SNR_bins):
            minimode = copy.deepcopy(mode)
            BW = (1 - ((lam - deltaLam/2)/lam)) * 2
            minimode['lam'] = lam
            minimode['BW'] = BW
            minimodes.append(minimode)

        if len(planinds) > 0:
            # initialize arrays for SNR integration
            fZs = np.zeros((self.ntFlux, num_SNR_bins))/u.arcsec**2
            systemParamss = np.empty(self.ntFlux, dtype='object')
            Ss = np.zeros((self.ntFlux, len(planinds), num_SNR_bins))
            Ns = np.zeros((self.ntFlux, len(planinds), num_SNR_bins))
            # integrate the signal (planet flux) and noise
            dt = intTime/self.ntFlux
            for i in range(self.ntFlux):
                TK.allocate_time(dt/2.)
                for j, minimode in minimodes:
                    # allocate first half of dt
                    # calculate current zodiacal light brightness
                    fZs[i] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, minimode)[0]
                    # propagate the system to match up with current time
                    SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
                    self.propagTimes[sInd] = TK.currentTimeNorm
                    # save planet parameters
                    systemParamss[i] = SU.dump_system_params(sInd)
                    # calculate signal and noise (electron count rates)
                    Ss[i,:,j], Ns[i,:,j] = self.calc_signal_noise(sInd, planinds, dt, minimode, 
                            fZ=fZs[i])
                    # allocate second half of dt
                    lam += deltaLam
                TK.allocate_time(dt/2.)
            
            # average output parameters
            fZ = np.mean(fZs)
            systemParams = {key: sum([systemParamss[x][key]
                    for x in range(self.ntFlux)])/float(self.ntFlux)
                    for key in sorted(systemParamss[0])}
            # calculate planets SNR
            S = Ss.sum(0)
            N = Ns.sum(0)
            SNRplans[N > 0] = S[N > 0]/N[N > 0]
            # allocate extra time for timeMultiplier
            extraTime = intTime*(mode['timeMultiplier'] - 1)
            TK.allocate_time(extraTime)

        # if only a FA, just save zodiacal brightness in the middle of the integration
        else:
            totTime = intTime*(mode['timeMultiplier'])
            TK.allocate_time(totTime/2.)
            fZ = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
            TK.allocate_time(totTime/2.)

        # calculate the false alarm SNR (if any)
        # XXX currenty broken
        SNRfa = []
        if pIndsChar[-1] == -1:
            fEZ = fEZs[-1]/u.arcsec**2
            dMag = dMags[-1]
            WA = WAs[-1]*u.arcsec
            for minimode in minimodes:
                C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, minimode)
            S = (C_p*intTime).decompose().value
            N = np.sqrt((C_b*intTime + (C_sp*intTime)**2).decompose().value)
            SNRfa = S/N if N > 0 else 0.

        # save all SNRs (planets and FA) to one array
        SNRinds = np.where(det)[0][tochar]
        SNR[SNRinds] = np.append(SNRplans, SNRfa)

        # now, store characterization status: 1 for full spectrum, 
        # -1 for partial spectrum, 0 for not characterized
        char = (SNR >= mode['SNR'])


