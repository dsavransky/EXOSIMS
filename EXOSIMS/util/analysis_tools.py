import EXOSIMS,EXOSIMS.MissionSim
import os.path,json
from scipy.stats import norm
from matplotlib.pyplot import *
import pickle
import sys

# Python 3 compatibility:
if sys.version_info[0] > 2:
    xrange = range

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def genOutSpec_ensemble(scriptfile, savefolder, nb_run_sim=1, **specs):
    """ Run an ensemble of simulations, and store results in save folder. 
    
    Args:
        scriptfile (boolean):
            Path to scripfile
        savefolder (boolean):
            Path to save folder
        nb_run_sim (integer):
            Number of simulations to run
    
    """
    
    for j in xrange(int(nb_run_sim)):
        print('\nSurvey simulation number %s/%s' %(j+1, int(nb_run_sim)))
        reload(EXOSIMS)
        reload(EXOSIMS.MissionSim)
        sim = EXOSIMS.MissionSim.MissionSim(scriptfile, **specs)
        SS = sim.SurveySimulation
        SS.run_sim()
        SS._outspec['DRM'] = SS.DRM
        sim.genOutSpec(savefolder + '/' + str(sim._outspec['seed']))
    
    return



def obs_ensemble(savefolder):
    """ Load observation results stored in save folder and build lists of ensemble results.
    
    Args:
        savefolder (boolean):
            Path to save folder
    
    Returns:
        obs (list):
            List of detection results, depending on the detection status
    
    """
    
    obs = []
    names = [x for x in os.listdir(savefolder) if x not in '.DS_Store']
    for name in names:
        specs = json.loads(open(savefolder+'/'+name).read())
        DRM = specs['DRM']
        nobs = len(DRM)
        obs.append(nobs)
        for i in xrange(nobs):
           det.append(len([x for x in DRM[i]['det_status'] if x == det_status]))
    
    return obs


def det_ensemble2(obj, status=None):
    """ Load the detection results stored in save folder and build lists of ensemble results.
    
    Args:
        savefolder (boolean):
            Path to save folder
        status (integer)
            Seleted detection status:
            1:detection, 0:missed detection, -1:outside IWA, -2:outside OWA
    
    Returns:
        det (list):
            List of detection results, depending on the detection status
    
    """
    
    det = []
    for i in range(len(obj)):
        DRM = obj[i]
        nobs = len(DRM)
        ndet = 0
        plan_inds = np.array([],dtype=int)
        for i in xrange(nobs):
            det_status = DRM[i]['det_status']
            if status is None: # unique detections
                mask = [j for j,k in enumerate([x==1 for x in det_status]) if k == True]
                plan_inds = np.append(plan_inds, np.array(DRM[i]['plan_inds'])[mask])
                ndet = np.unique(plan_inds).size
            else:
                ndet += sum([int(x==status) for x in det_status])
        det.append(ndet)
    
    return det



def det_ensemble(savefolder, status=None):
    """ Load the detection results stored in save folder and build lists of ensemble results.
    
    Args:
        savefolder (boolean):
            Path to save folder
        status (integer)
            Seleted detection status:
            1:detection, 0:missed detection, -1:outside IWA, -2:outside OWA
    
    Returns:
        det (list):
            List of detection results, depending on the detection status
    
    """
    
    det = []
    names = [x for x in os.listdir(savefolder) if x not in '.DS_Store']
    for name in names:
        specs = json.loads(open(savefolder+'/'+name).read())
        DRM = specs['DRM']
        nobs = len(DRM)
        ndet = 0
        plan_inds = np.array([],dtype=int)
        for i in xrange(nobs):
            det_status = DRM[i]['det_status']
            if status is None: # unique detections
                mask = [j for j,k in enumerate([x==1 for x in det_status]) if k == True]
                plan_inds = np.append(plan_inds, np.array(DRM[i]['plan_inds'])[mask])
                ndet = np.unique(plan_inds).size
            else:
                ndet += sum([int(x==status) for x in det_status])
        det.append(ndet)
    
    return det


def char_ensemble(savefolder, status=None):
    """ Load the characterization results stored in save folder and build lists of ensemble results.
    
    Args:
        savefolder (boolean):
            Path to save folder
        status (integer)
            Characterization status:
            1:full spectrum, -1:partial spectrum, 0:not characterized
    
    Returns:
        char (list):
            List of characterization results, depending on the characterization status
    
    """
    
    char = []
    names = [x for x in os.listdir(savefolder) if x not in '.DS_Store']
    for name in names:
        specs = json.loads(open(savefolder+'/'+name).read())
        DRM = specs['DRM']
        nobs = len(DRM)
        nchar = 0
        for i in xrange(nobs):
            nchar += len([x for x in DRM[i]['char_status'] if x == status])
        char.append(nchar)
    
    return char


def draw_pdf(var1, var2=None, label1='var1', label2='var2', xlab='variable', \
        ylab='Normalized frequency', linewidth=2, framealpha=0.3, \
        markersize=5, fontsize=18, xmax=18, binfac=1): 
    
    bins = range(xmax+1)
    (mu, sigma) = norm.fit(var1)
    y1 = mlab.normpdf(bins, mu, sigma)
    figure()
    grid('on')
    plot(bins, y1, 'b-o', linewidth=linewidth, markersize=markersize, label=label1)
    if var2 != None:
        (mu, sigma) = norm.fit(var2)
        y2 = mlab.normpdf(bins, mu, sigma)
        plot(bins, y2, 'r-o', linewidth=linewidth, markersize=markersize, label=label2)
    xlabel(xlab, fontsize=fontsize)
    ylabel(ylab, fontsize=fontsize)
    xlim(0,xmax)
    tick_params(axis='both', which='major', labelsize=fontsize)
    if var2 != None:
        legend(fancybox=True, framealpha=framealpha, loc='upper right', fontsize=fontsize)
    
    return


def det_multi_ensemble(savefolder, varvalues, status=None):
    """ Load the detection results stored in save folder and build lists of ensemble results.
    
    Args:
        savefolder (boolean):
            Path to save folder
        varvalues (nparray):
            Variable values
        det_status (integer)
            Detection status, 1:detection, 0:missed detection, -1:outside IWA, -2:outside OWA
    
    Returns:
        det (list):
            List of detection results, depending on the detection status
    
    """
    
    det_mean = []
    det_std = []
    for i,j in enumerate(varvalues):
        subfolder = savefolder + '/' + str(j)
        det = det_ensemble(subfolder, status)
        det_mean.append(mean(det))
        det_std.append(std(det))
    
    return det_mean, det_std

def draw_multi_ensemble(savefolder, varvalues, status=None, label1='var1', label2='var2', xlab='variable', \
        ylab='Normalized frequency', linewidth=2, framealpha=0.3, \
        markersize=5, fontsize=18):
    
    det_mean, det_std = det_multi_ensemble(savefolder, varvalues, status)
    figure()
    grid('on')
    plot(varvalues, det_mean, 'b-o', linewidth=linewidth, markersize=markersize, label=label1)
    plot(varvalues, det_std, 'r-o', linewidth=linewidth, markersize=markersize, label=label2)
    xlabel(xlab, fontsize=fontsize)
    ylabel(ylab, fontsize=fontsize)
#    xlim(0,xmax)
    tick_params(axis='both', which='major', labelsize=fontsize)
    legend(fancybox=True, framealpha=framealpha, loc='best', fontsize=fontsize)
    
    return
    
    
    
    
    