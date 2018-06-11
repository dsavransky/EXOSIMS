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
t = ['Dean6May18RS09CXXfZ01OB08PP01SU01',\
'Dean6May18RS09CXXfZ01OB09PP01SU01',\
'Dean6May18RS09CXXfZ01OB10PP01SU01',\
'Dean6May18RS09CXXfZ01OB11PP01SU01',\
'Dean6May18RS09CXXfZ01OB12PP01SU01',\
'Dean6May18RS09CXXfZ01OB13PP01SU01',\
'Dean6May18RS09CXXfZ01OB14PP01SU01',\
'Dean6May18RS09CXXfZ01OB15PP01SU01',\
'Dean6May18RS09CXXfZ01OB16PP01SU01',\
'Dean6May18RS09CXXfZ01OB17PP01SU01',\
'Dean6May18RS09CXXfZ01OB18PP01SU01',\
'Dean2May18RS09CXXfZ01OB01PP01SU01',\
'Dean6May18RS09CXXfZ01OB19PP01SU01',\
'Dean6May18RS09CXXfZ01OB20PP01SU01',\
'Dean6May18RS09CXXfZ01OB21PP01SU01']
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

# 1mo, 0.08333yr=08
# 2mo, 0.1666yr=09
# 3mo, 0.25yr=10
# 4mo, 0.33333yr=11
# 5mo, 0.416666yr=12
# 6mo, 0.5yr=13
# 7mo, 0.583333yr=14
# 8mo, 0.6666666yr=15
# 9mo, 0.75yr=16
# 10mo, 0.83333yr=17
# 11mo, 0.91666yr=18
# 13mo, 1.08333yr=19
# 14mo, 1.166666yr=20
# 15mo, 1.25yr=21
months = np.arange(15)+1


#Clump
rcounts = []
#el = res1['detected']
res = [res1['detected'], res2['detected'], res3['detected'], res4['detected'], res5['detected'],\
    res6['detected'], res7['detected'], res8['detected'], res9['detected'], res10['detected'],\
    res11['detected'], res12['detected'], res13['detected'], res14['detected'], res15['detected']]
for el in res:
    rcounts.append(np.array([np.unique(r).size for r in el]).astype(float))#unique detections
    #rcounts.append(np.array([len(r) for r in el]))

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

fig = figure(1)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rc('axes',prop_cycle=(cycler('color',['purple'])))#,'blue','black','purple'])))
rcParams['axes.linewidth']=2
rc('font',weight='bold') 

B = boxplot(np.transpose(asarray(rcounts)), sym='', whis= 1000.)
xlabel('Total Mission Time (months)', weight='bold')
ylabel('Unique Detections', weight='bold')
axes = gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([0,1.1*np.amax(rcounts)])
show(block=False)

filename = 'UniqDetvsMissionLength'
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
parts = violinplot(np.transpose(asarray(rcounts)), showmeans=False, showmedians=False, showextrema=False, widths=0.75)
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
xlabel('Total Mission Time (months)', weight='bold')
ylabel('Unique Detections', weight='bold')
axes = gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([0,1.05*np.amax(rcounts)])


inds = np.arange(len(rcounts))+1
scatter(inds, meanUniqueDetections, marker='o', color='k', s=30, zorder=3)
vlines(inds, minNumDetected, maxNumDetected, color='k', linestyle='-', lw=2)
vlines(inds, twentyfifthPercentile, seventyfifthPercentile, color='silver', linestyle='-', lw=5)

show(block=False)

filename = 'UniqDetvsMissionLengthVIOLIN'
runPath = '/home/dean/Documents/SIOSlab/SPIE2018Journal/'
savefig(runPath + filename + '.png', format='png', dpi=500)
savefig(runPath + filename + '.svg')
savefig(runPath + filename + '.eps', format='eps', dpi=500)
show(block=False)

