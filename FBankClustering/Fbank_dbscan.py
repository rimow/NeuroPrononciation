from time import time
import matplotlib.pyplot as plt
import operator
import  math
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import scipy.io as sio
from sklearn.cluster import KMeans
import numpy as np

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def include_period(p1,p2):
    if((p1[0]<=p2[0]) and (p1[1]>=p2[1])):
        return 1
    elif((p1[0]<=p2[0]) and (p1[1]<p2[1])):
        return 2
    else:
        return 3




filename = './data/Bref80_L4M01.mat'
alignfile = './data/Bref80_L4M01.aligned'
fbank = sio.loadmat(filename)['d1']
n_samples, n_features = fbank.shape
period = np.loadtxt(alignfile,delimiter=' ',usecols=(0,1))
phoneme = np.loadtxt(alignfile,dtype= str ,delimiter=' ',usecols=[2])
n_phoneme = len(np.unique(phoneme))
cut_period = [[round(start,2),round(end,2)] for start,end in zip(drange(0,(n_samples+1)*0.01,0.01),drange(0.02,(n_samples+2)*0.01,0.01))]
#cut_period = [(start,end) for start,end in zip(drange(0,(n_samples-1)/100,0.01),drange(0.02,n_samples/100,0.01))]



pho = [None]*n_samples


i=0
j=0

while (i<n_samples and j<len(period)-1):
    if(include_period(period[j],cut_period[i])==1):
        pho[i] = phoneme[j]
        i+=1
    elif(include_period(period[j],cut_period[i])==2):
        pho[i] = phoneme[j]#+'+'+phoneme[j+1]
        j+=1
        i+=1
    elif (include_period(period[j],cut_period[i])==3):

        pho[i-1]=phoneme[j]
        j+=1
while(i<n_samples):
    pho[i]=phoneme[-1]
    i+=1

print(fbank[0])
print(np.linalg.norm(fbank[0]-fbank[10]))

reduced_data = PCA(n_components=2).fit_transform(fbank)
fbank= reduced_data
db = DBSCAN(eps=0.5, min_samples=30).fit(fbank)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
unique_labels = set(labels)
if -1 in unique_labels:
    labels = [x+2 for x in labels]
else:
    labels = [x+1 for x in labels]


print(set(labels))
# Number of clusters in labels, ignoring noise if present.
n_clusters= len(set(labels)) - (1 if -1 in labels else 0)

cluster_pho = [None]*n_clusters

for label,ph in zip(labels,pho):
    if(cluster_pho[label-1]==None):
        cluster_pho[label-1]= {ph:1}
    elif ph in cluster_pho[label-1].keys():
        cluster_pho[label-1][ph]+=1
    else:
        cluster_pho[label-1][ph]=1



cluster_pho[:]= [sorted(x.items(),key = operator.itemgetter(1),reverse=True)for x in cluster_pho]

#first = cluster_pho[0];
#temp = zip(*first)
nrows = int(round(math.sqrt(n_clusters)))
ncols = int(math.ceil(n_clusters/round(math.sqrt(n_clusters))))

figures,axs  = plt.subplots(nrows = nrows,ncols = ncols )

for ax,data in zip(axs.ravel(),cluster_pho):
    data = zip(*data)
    ax.bar(range(len(data[0])),data[1],width = 0.2)
    ax.set_xticks(np.arange(len(data[0]))+0.1)
    ax.set_xticklabels(data[0],rotation = 0)
#plt.hist(temp[1],bins=temp[0])
plt.show()

print(cluster_pho)




print(unique_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = fbank[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = fbank[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters)
plt.show()
