
import matplotlib.pyplot as plt
import operator
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.cluster import KMeans
import numpy as np
import math
import preprocess
import gap
import analyse

# Read mat file and align file.
filename = './data/Bref80_L4M01.mat'
alignfile = './data/Bref80_L4M01.aligned'
fbank = sio.loadmat(filename)['d1']
reduced_data = PCA(n_components=2).fit_transform(fbank)
#fbank = reduced_data
pho = preprocess.create_reference(fbank,alignfile)


ks, logWks, logWkbs, sk = gap.gap_statistic(fbank,20,30)


plt.plot(ks,logWks)
plt.xlabel("number of cluster  K")
plt.ylabel("log W")
plt.show()

plt.plot(ks,logWkbs)
plt.xlabel("number of cluster  K")
plt.ylabel("gap")
plt.show()
gap_diff = []
for i in range(0,len(logWkbs)-1):
    gap_diff.append(logWkbs[i]-(logWkbs[i+1]-sk[i+1]))

ax = plt.subplot(111)
ax.bar(range(len(ks[0:-1])),gap_diff,width = 0.6)
ax.set_xticks(np.arange(len(ks[0:-1]))+0.3)
ax.set_xticklabels(ks[0:-1],rotation = 0)
plt.xlabel("number of cluster  K")
plt.ylabel("gap(k)-(gap(k+1)-s(k+1)")
plt.show()

#
# print(gap_diff)
#
# print ks,logWks,logWkbs,sk

# inertias=[]
# #cluster the data to n_clusters class.
# for n_clusters in range(50,60):
#     kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
#     kmeans.fit(fbank)
#     inertias.append(kmeans.inertia_)

# plt.plot(range(50,60),inertias)
# plt.xlabel("cluster number")
# plt.ylabel("inertia")
# plt.title("Inertia -Cluster number")
# plt.show()


n_clusters = 3
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(fbank)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

analyse.pourcentage(pho,n_clusters,labels,"../classement",1)
#Calculate the number of each phoneme in each cluster
cluster_pho = [None]*n_clusters
for label,ph in zip(labels,pho):
    if(cluster_pho[label-1]==None):
        cluster_pho[label-1]= {ph:1}
    elif ph in cluster_pho[label-1].keys():
        cluster_pho[label-1][ph]+=1
    else:
        cluster_pho[label-1][ph]=1
cluster_pho[:]= [sorted(x.items(),key = operator.itemgetter(1),reverse=True)for x in cluster_pho]


# use bar chart to visualize each class
nrows = int(round(math.sqrt(n_clusters)))
ncols = int(math.ceil(n_clusters/round(math.sqrt(n_clusters))))
figures,axs  = plt.subplots(nrows = nrows,ncols = ncols)

for ax,data in zip(axs.ravel(),cluster_pho):
    data = zip(*data)
    ax.bar(range(len(data[0])),data[1],width = 0.2)
    ax.set_xticks(np.arange(len(data[0]))+0.1)
    ax.set_xticklabels(data[0],rotation = 0)
plt.show()

print(cluster_pho)


#Visulization by PCA
reduced_data = PCA(n_components=2).fit_transform(fbank)
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

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
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
