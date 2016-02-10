""" Implement Gap statistics algorithme to choose K for Kmeans"""

import numpy as np
from sklearn.cluster import KMeans


def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2  for i in range(K) for c in clusters[i]])

def bounding_box(X):
    mins = []
    maxs = []
    for i in range(X.shape[1]):
        mins.append(min(X,key=lambda a:a[i])[i])
        maxs.append(max(X,key=lambda a:a[i])[i])
    return mins,maxs

def gap_statistic(X,low,high):
    """Calculate the gap.Choose the number of clusters as the smallest k such that gap(k) > Gap(k+1)-s(k+1).

        Args:
            X: samples data
            low: low boundary of cluster number
            high: high boundary of cluster number
        Returns:
            ks: number of clusters
            Wks:Wk in the algorithem
            Wkbs:gap
            sk: sk
    """
    mins,maxs = bounding_box(X)
    # Dispersion for real distribution
    ks = range(low,high+1)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))

    for indk, k in enumerate(ks):

        clusters = [[]]*k
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=5)
        kmeans.fit(X)
        mu = kmeans.cluster_centers_
        labels = kmeans.labels_
        for i,data in zip(labels,X):
            clusters[i].append(data)
        #print(clusters)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([np.random.uniform(min_i,max_i) for min_i,max_i in zip(mins,maxs)])
            Xb = np.array(Xb)
            clusters = [[]]*k
            kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
            kmeans.fit(Xb)
            mu = kmeans.cluster_centers_
            labels = kmeans.labels_
            for j,data in zip(labels,Xb):
                clusters[j].append(data)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk)
