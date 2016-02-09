from time import time
import matplotlib.pyplot as plt
import operator

from sklearn.cluster import MeanShift, estimate_bandwidth
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
fband = sio.loadmat(filename)['d1']
n_samples, n_features = fband.shape
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


#print(pho)

bandwidth = estimate_bandwidth(fband, quantile=0.1,n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(fband)

centroids = ms.cluster_centers_
labels = ms.labels_
n_clusters = len(np.unique(labels))
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
figures,axs  = plt.subplots(nrows = 2,ncols = 2)

for ax,data in zip(axs.ravel(),cluster_pho):
    data = zip(*data)
    ax.bar(range(len(data[0])),data[1],width = 0.2)
    ax.set_xticks(np.arange(len(data[0]))+0.1)
    ax.set_xticklabels(data[0],rotation = 0)
#plt.hist(temp[1],bins=temp[0])
plt.show()

print(cluster_pho)



reduced_data = PCA(n_components=2).fit_transform(fband)
bandwidth = estimate_bandwidth(reduced_data, quantile=0.2,n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = ms.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = ms.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('Meanshift clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
