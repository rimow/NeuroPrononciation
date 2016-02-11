
import scipy.io as sio
from sklearn.cluster import KMeans
import phonemesAnalysis.utiles as utiles
import phonemesAnalysis.analyse as analyse



# Read mat file and align file.
filename = './data/Bref80_L4M01.mat'
alignfile = './data/Bref80_L4M01.aligned'
fbank = sio.loadmat(filename)['d1']
csv = "./resultats/matlabFbank.csv"
classementPath = "./data/classement"
hop_span = 0.01
Y = utiles.getY(fbank,alignfile,hop_span)

#Kmeans without initialization 3 classes (consonnes et voyelles)
n_clusters = 3
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(fbank)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
analyse.pourcentage(Y,n_clusters,labels,classementPath,0,csv)

#Kmeans without initialization 3 classes(voise et non voise)
n_clusters = 3
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(fbank)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
analyse.pourcentage(Y,n_clusters,labels,classementPath,1,csv)

#Kmeans withous initialzation 6 classes

n_clusters = 6
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(fbank)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
analyse.pourcentage(Y,n_clusters,labels,classementPath,2,csv)

analyse.histogrammesPhonemes(n_clusters,labels,Y)


