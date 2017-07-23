import os.path
import glob
try:
    import cPickle as pickle
except:
    import pickle
import astropy.units as u
import numpy as np


def gen_summary(run_dir):
    basedir = '/data/extmount/EXOSIMSres'
    pklfiles = glob.glob(os.path.join(run_dir,'*.pkl'))

    out = {'detected':[],
           'fullspectra':[],
           'partspectra':[],
           'Rps':[],
           'Mps':[],
           'tottime':[],
           'starinds':[],
           'smas':[],
           'ps':[],
           'es':[],
           'WAs':[],
           'SNRs':[],
           'fZs':[],
           'fEZs':[]}

    for f in pklfiles:
        print f
        with open(f, 'rb') as g:
            res = pickle.load(g)

        dets = np.hstack([row['plan_inds'][row['det_status'] == 1]  for row in res['DRM']])
        out['detected'].append(dets)
        
        out['WAs'].append(np.hstack([row['det_params']['WA'][row['det_status'] == 1].to('arcsec').value for row in res['DRM']]))
        out['fEZs'].append(np.hstack([row['det_params']['fEZ'][row['det_status'] == 1].value for row in res['DRM']]))
        out['fZs'].append(np.hstack([[row['det_fZ'].value]*len(np.where(row['det_status'] == 1)[0]) for row in res['DRM']]))
        out['fullspectra'].append(np.hstack([row['plan_inds'][row['char_status'] == 1]  for row in res['DRM']]))
        out['partspectra'].append(np.hstack([row['plan_inds'][row['char_status'] == -1]  for row in res['DRM']]))
        out['tottime'].append(np.sum([row['det_time'].value+row['char_time'].value for row in res['DRM']]))
        out['SNRs'].append(np.hstack([row['det_SNR'][row['det_status'] == 1]  for row in res['DRM']]))
        out['Rps'].append((res['systems']['Rp'][dets]/u.R_earth).decompose().value)
        out['smas'].append(res['systems']['a'][dets].to(u.AU).value)
        out['ps'].append(res['systems']['p'][dets])
        out['es'].append(res['systems']['e'][dets])
        out['Mps'].append((res['systems']['Mp'][dets]/u.M_earth).decompose())
        out['starinds'].append(np.hstack([[row['star_ind']]*len(np.where(row['det_status'] == 1)[0]) for row in res['DRM']]))

    return out


