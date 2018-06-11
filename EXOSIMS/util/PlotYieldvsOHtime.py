"""
Written By: Dean Keithly
"""

from EXOSIMS.util.read_ipcluster_ensemble import gen_summary
from pylab import *
import numpy as np
from cycler import cycler
import math



#5/14/2018 SLSQP Scheduler static Yield vs Mission Length
pathRuns = '/home/dean/Documents/SIOSlab/'
t = ['Dean15May18RS09CXXfZ01OB22PP01SU01',\
'Dean15May18RS09CXXfZ01OB23PP01SU01',\
'Dean15May18RS09CXXfZ01OB24PP01SU01',\
'Dean15May18RS09CXXfZ01OB25PP01SU01',\
'Dean15May18RS09CXXfZ01OB26PP01SU01',\
'Dean15May18RS09CXXfZ01OB27PP01SU01',\
'Dean15May18RS09CXXfZ01OB28PP01SU01',\
'Dean15May18RS09CXXfZ01OB29PP01SU01',\
'Dean15May18RS09CXXfZ01OB30PP01SU01',\
'Dean15May18RS09CXXfZ01OB31PP01SU01',\
'Dean15May18RS09CXXfZ01OB32PP01SU01',\
'Dean15May18RS09CXXfZ01OB33PP01SU01',\
'Dean15May18RS09CXXfZ01OB34PP01SU01',\
'Dean15May18RS09CXXfZ01OB35PP01SU01',\
'Dean15May18RS09CXXfZ01OB36PP01SU01',\
'Dean15May18RS09CXXfZ01OB37PP01SU01',\
'Dean15May18RS09CXXfZ01OB38PP01SU01',\
'Dean15May18RS09CXXfZ01OB39PP01SU01',\
'Dean15May18RS09CXXfZ01OB40PP01SU01',\
'Dean15May18RS09CXXfZ01OB41PP01SU01',\
'Dean15May18RS09CXXfZ01OB42PP01SU01',\
'Dean15May18RS09CXXfZ01OB43PP01SU01',\
'Dean15May18RS09CXXfZ01OB44PP01SU01',\
'Dean15May18RS09CXXfZ01OB45PP01SU01',\
'Dean15May18RS09CXXfZ01OB46PP01SU01',\
'Dean15May18RS09CXXfZ01OB47PP01SU01',\
'Dean15May18RS09CXXfZ01OB48PP01SU01',\
'Dean15May18RS09CXXfZ01OB49PP01SU01',\
'Dean15May18RS09CXXfZ01OB50PP01SU01',\
'Dean15May18RS09CXXfZ01OB51PP01SU01',\
'Dean15May18RS09CXXfZ01OB52PP01SU01',\
'Dean15May18RS09CXXfZ01OB53PP01SU01',\
'Dean2May18RS09CXXfZ01OB01PP01SU01']#replace with more recent run
res1 = gen_summary(pathRuns + t[0])
res2 = gen_summary(pathRuns + t[1])
res3 = gen_summary(pathRuns + t[2])
res4 = gen_summary(pathRuns + t[3])
res5 = gen_summary(pathRuns + t[4])
res6 = gen_summary(pathRuns + t[5])
res7 = gen_summary(pathRuns + t[6])
res8 = gen_summary(pathRuns + t[7])
res9 = gen_summary(pathRuns + t[8])
res10 = gen_summary(pathRuns + t[9])
res11 = gen_summary(pathRuns + t[10])
res12 = gen_summary(pathRuns + t[11])
res13 = gen_summary(pathRuns + t[12])
res14 = gen_summary(pathRuns + t[13])
res15 = gen_summary(pathRuns + t[14])
res16 = gen_summary(pathRuns + t[15])
res17 = gen_summary(pathRuns + t[16])
res18 = gen_summary(pathRuns + t[17])
res19 = gen_summary(pathRuns + t[18])
res20 = gen_summary(pathRuns + t[19])
res33 = gen_summary(pathRuns + t[32])
res21 = gen_summary(pathRuns + t[20])
res22 = gen_summary(pathRuns + t[21])
res23 = gen_summary(pathRuns + t[22])
res24 = gen_summary(pathRuns + t[23])
res25 = gen_summary(pathRuns + t[24])
res26 = gen_summary(pathRuns + t[25])
res27 = gen_summary(pathRuns + t[26])
res28 = gen_summary(pathRuns + t[27])
res29 = gen_summary(pathRuns + t[28])
res30 = gen_summary(pathRuns + t[29])
res31 = gen_summary(pathRuns + t[30])
res32 = gen_summary(pathRuns + t[31])

#I took these directly from the excel spreadsheet
ohTime = np.asarray([0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,\
    0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8])*2# X 2 because ohTime and settlingTime are the same

#Clump
rcounts = []
#el = res1['detected']
res = [res1['detected'], res2['detected'], res3['detected'], res4['detected'], res5['detected'],\
    res6['detected'], res7['detected'], res8['detected'], res9['detected'], res10['detected'],\
    res11['detected'], res12['detected'], res13['detected'], res14['detected'], res15['detected'],\
    res16['detected'], res17['detected'], res18['detected'], res19['detected'], res20['detected'], res33['detected'],\
    res21['detected'], res22['detected'], res23['detected'], res24['detected'], res25['detected'],\
    res26['detected'], res27['detected'], res28['detected'], res29['detected'], res30['detected'],\
    res31['detected'], res32['detected']]
for el in res:
    rcounts.append(np.array([np.unique(r).size for r in el]).astype(float))#unique detections



#calculate mean detections, standard deviations, and percentiles
meanUniqueDetections = list()
fifthPercentile = list()
twentyfifthPercentile = list()
fiftiethPercentile = list()
seventyfifthPercentile = list()
ninetiethPercentile = list()
nintyfifthPercentile = list()
minNumDetected = list()
percentAtMinimum = list()
maxNumDetected = list()
stdUniqueDetections = list()
for el in rcounts:
    meanUniqueDetections.append(mean(el))
    stdUniqueDetections.append(np.std(el))
    fifthPercentile.append(np.percentile(el,5))
    twentyfifthPercentile.append(np.percentile(el,25))
    fiftiethPercentile.append(np.percentile(el,50))
    seventyfifthPercentile.append(np.percentile(el,75))
    ninetiethPercentile.append(np.percentile(el,90))
    nintyfifthPercentile.append(np.percentile(el,95))
    minNumDetected.append(min(el))
    percentAtMinimum.append(float(el.tolist().count(min(el)))/len(el))
    maxNumDetected.append(max(el))
print meanUniqueDetections

fig = figure(1, figsize=(8.5,4.5))
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rc('axes',prop_cycle=(cycler('color',['purple'])))#,'blue','black','purple'])))
rcParams['axes.linewidth']=2
rc('font',weight='bold') 

B = boxplot(np.transpose(asarray(rcounts)), sym='', whis= 1000., positions=ohTime, widths=0.03125)
xlabel('Overhead Time (days)', weight='bold')
ylabel('Unique Detections', weight='bold')
axes = gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([0,1.1*np.amax(rcounts)])
axes.set_xlim([0.03125, 1.6+0.03125])
show(block=False)

filename = 'UniqDetvsOHTime'
runPath = '/home/dean/Documents/SIOSlab/SPIE2018Journal/'
savefig(runPath + filename + '.png', format='png', dpi=500)
savefig(runPath + filename + '.svg')
savefig(runPath + filename + '.eps', format='eps', dpi=500)
#scatter(months, meanUniqueDetections)

#How to get box plot quartiles means and such
[item.get_ydata() for item in B['boxes']]


#Calculate Percent of Mission Time Wasted
key = 'tottime'
res_detTime = [res1[key], res2[key], res3[key], res4[key], res5[key],\
    res6[key], res7[key], res8[key], res9[key], res10[key],\
    res11[key], res12[key], res13[key], res14[key], res15[key]]

# key = 'tottime'
# res_detTime = [res1[key], res2[key], res3[key], res4[key], res5[key],\
#     res6[key], res7[key], res8[key], res9[key], res10[key],\
#     res11[key], res12[key], res13[key], res14[key], res15[key]]


#pklfiles = glob.glob(os.path.join(run_dir,'*.pkl'))


### Make Dmitry's violin plots
fig2 = figure(2, figsize=(8.5,4.5))
parts = violinplot(np.transpose(asarray(rcounts)), showmeans=False, showmedians=False, showextrema=False, widths=0.03125, positions=ohTime)
for pc in parts['bodies']:
    #pc.set_facecolor('#D43F3A')
    pc.set_facecolor('purple')
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rc('axes',prop_cycle=(cycler('color',['purple'])))#,'blue','black','purple'])))
rcParams['axes.linewidth']=2
rc('font',weight='bold') 
xlabel('Overhead Time (days)', weight='bold')
ylabel('Unique Detections', weight='bold')
axes = gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([0,1.05*np.amax(rcounts)])
axes.set_xlim([0.03125, 1.6+0.03125])


inds = np.arange(len(rcounts))+1
scatter(ohTime, meanUniqueDetections, marker='o', color='k', s=30, zorder=3)
vlines(ohTime, minNumDetected, maxNumDetected, color='k', linestyle='-', lw=1)
vlines(ohTime, twentyfifthPercentile, seventyfifthPercentile, color='silver', linestyle='-', lw=3)

show(block=False)

filename = 'UniqDetvsOHTimeVIOLIN'
runPath = '/home/dean/Documents/SIOSlab/SPIE2018Journal/'
savefig(runPath + filename + '.png', format='png', dpi=500)
savefig(runPath + filename + '.svg')
savefig(runPath + filename + '.eps', format='eps', dpi=500)
show(block=False)


