"""
Plot Yield Plot Histograms
Written by: Dean Keithly
Written on: 11/28/2018
"""
from EXOSIMS.util.read_ipcluster_ensemble import gen_summary
import os
if not 'DISPLAY' in os.environ.keys(): #Check environment for keys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import math
import datetime
import re

class yieldPlotHistograms(object):
    """Designed to plot yield plot histograms
    """
    _modtype = 'util'

    def __init__(self, args=None):
        """
        Args:
            args is not used
        """
        pass

    def singleRunPostProcessing(self, PPoutpath, folder):
        """Generates a single yield histogram for the run_type
        Args:
            PPoutpath (string) - output path to place data in
            folder (string) - full filepath to folder containing runs
        """
        res = gen_summary(folder)
        self.res = res
        self.dist_plot([res['detected']],PPoutpath=PPoutpath,folder=folder,legtext=[''])

    def multiRunPostProcessing(self, PPoutpath, folders):
        """Generates a yield histograms for the provided run_types
        Args:
            PPoutpath (string) - output path to place data in
            folders (string) - full filepaths to folders containing runs of each run_type
        """
        resList = list()
        for folder in folders:
            resList.append(gen_summary(folder)['detected'])
        self.resList = resList
        self.dist_plot(resList,PPoutpath=PPoutpath,folder=folder)


    def dist_plot(self,res,uniq = True,fig=None,lstyle='--',plotmeans=True,legtext=None,PPoutpath=os.getcwd(),folder=None):
        """
        Args:
            res (list) - list of gen_summary outputs [res1,res2,res3]
            uniq (boolean) - indicates whether to count unique detections (True) or all detections (False)
            fig (integer)
            lstyle (string)
            plotmeans (boolean)
            legtext (list) - list of strings containing legend labels for each element in res
            PPoutpath (string) -
            folder (string) - name of folder containing runs used to generate this plot (used for naming only)
        """
        #Set linewidth and color cycle
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rc('axes',prop_cycle=(cycler('color',['red','purple','blue','black','darkorange','forestgreen'])))

        rcounts = []
        for el in res:
            if uniq:
                rcounts.append(np.array([np.unique(r).size for r in el]))
            else:
                rcounts.append(np.array([len(r) for r in el]))

        bins = range(np.min(np.hstack(rcounts).astype(int)),np.max(np.hstack(rcounts).astype(int))+2)
        bcents = np.diff(bins)/2. + bins[:-1]

        pdfs = []
        for j in range(len(res)):
            pdfs.append(np.histogram(rcounts[j],bins=bins,density=True)[0].astype(float))

        mx = 1.1*np.max(pdfs) #math.ceil(np.max(pdfs)*10)/10#np.round(np.max(pdfs),decimals=1)
        print(mx)

        syms = 'osp^v<>h'
        if fig is None:
            plt.figure()
        else:
            plt.figure(fig)
            plt.gca().set_prop_cycle(None)

        if legtext is None:
            legtext = [None]*len(res)

        for j in range(len(res)):
            leg = legtext[j]
            #c = plt.gca()._get_lines.prop_cycler.next()['color']#before 3.6
            c = plt.gca()._get_lines.prop_cycler.__next__()['color']#after 3.6
            if plotmeans:
                mn = np.mean(rcounts[j])
                plt.plot([mn]*2,[0,mx],'--',color=c)
                if leg is not None:
                    leg += ' ($\\mu = %2.2f$)'%mn
            plt.plot(bcents, pdfs[j], syms[np.mod(j,len(syms))]+lstyle,color=c,label=leg)

        plt.ylim([0,mx])
        if legtext[0] is not None:
            plt.legend()
        plt.xlabel('Unique Detections',weight='bold')
        plt.ylabel('Normalized Yield Frequency',weight='bold')

        date = datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S_%f")
        #date = unicode(datetime.datetime.now())
        date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
        fname = 'yieldPlotHist_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath,fname+'.png'))
        plt.savefig(os.path.join(PPoutpath,fname+'.svg'))
        plt.savefig(os.path.join(PPoutpath,fname+'.eps'))
        plt.savefig(os.path.join(PPoutpath,fname+'.pdf'))
