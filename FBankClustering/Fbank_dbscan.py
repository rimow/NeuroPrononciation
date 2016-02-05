import matplotlib.pyplot as plt
import operator
import  math
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import scipy.io as sio
import numpy as np
import preprocess

filename = './data/Bref80_L4M01.mat'
alignfile = './data/Bref80_L4M01.aligned'
fbank = sio.loadmat(filename)['d1']

print(np.linalg.norm(fbank[-10]-fbank[6]))
pho = preprocess.create_reference(fbank,alignfile)
print(pho)
reduced_data = PCA(n_components=2).fit_transform(fbank)
#fbank= reduced_data
db = DBSCAN(eps=2.1, min_samples=20).fit(fbank)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
unique_labels = set(labels)
if -1 in unique_labels:
    labels = [x+2 for x in labels]
else:
    labels = [x+1 for x in labels]

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
