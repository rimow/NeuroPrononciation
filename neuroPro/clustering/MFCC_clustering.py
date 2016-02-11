from phonemesAnalysis.utiles import *
from phonemesAnalysis.featuresGeneration import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster
from phonemesAnalysis.analyse import pourcentage


##########################################################################################################################
################################################### DONNEES ##############################################################
##########################################################################################################################

#Chemin du fichier ou on souhaite ecrire les resultats, peut s'ouvrir avec Excel
fichier = "../resultats/resultatsClustering/mfccClustering(40Coeffs).csv"

path = "../data/Bref80_L4M01.wav"
path_aligned = "../data/Bref80_L4M01.aligned"
dict_path = "../data/classement"
dict = getPhonemeDict(dict_path)

fft_span = 0.02
hop_span = 0.01
n_mels = 40

##########################################################################################################################
############################################ MATRICE DE CLUSTERING #######################################################
##########################################################################################################################

#Soit on effectue la transformation
X = mfcc(path, fft_span, hop_span, n_mels)

#Soit on charge la matrice si elle est deja enregistree
# X = np.load('../resultats/resultatsClustering/mfcc.npy')


nb_features,nb_vectors = X.shape
Y = getY(X,path_aligned, hop_span) #Initialisation du tableau contenant les donnees d'alignement

##########################################################################################################################
############################################ CLUSTERING 3 CLASSES ########################################################
##########################################################################################################################

print 'Clustering 3 classes : \n'

#nombre de clusters
nb_cluster = 3

#N.B : difference qu'on souhaite evaluer sur les phonemes : separation consonnes/voyelles = 0, voisee/non-voisee = 1, categories = 2

#KMEANS non initialise 3 classes
clus = MiniBatchKMeans(n_clusters = nb_cluster, init='k-means++',  batch_size=700,
                                  n_init=10, max_no_improvement=10, verbose=0)
f = open(fichier, "a")
f.write("KMEANS non initialise 3 clusters\n")
f.close()
clus.fit(X)
labels = clus.labels_
pourcentage(Y , nb_cluster, labels , dict_path , 0, fichier)
pourcentage(Y , nb_cluster, labels , dict_path , 1, fichier)
pourcentage(Y , nb_cluster, labels , dict_path , 2, fichier)

#KMEANS initialise 3 classes
sous = initialisation_centres(nb_cluster, X)
clus = MiniBatchKMeans(n_clusters = nb_cluster, init=sous,  batch_size=700,
                                  n_init=10, max_no_improvement=10, verbose=0)
f = open(fichier, "a")
f.write("KMEANS initialise 3 clusters\n")
f.close()
clus.fit(X)
labels = clus.labels_
pourcentage(Y , nb_cluster, labels , dict_path , 0, fichier)
pourcentage(Y , nb_cluster, labels , dict_path , 1, fichier)
pourcentage(Y , nb_cluster, labels , dict_path , 2, fichier)

#Agglomerative clustering 3 classes
clus = AgglomerativeClustering(nb_cluster,affinity='cosine',linkage='complete')
f = open(fichier, "a")
f.write("Agglomerative clustering 3 clusters\n")
f.close()
clus.fit(X)
labels = clus.labels_
pourcentage(Y , nb_cluster, labels , dict_path , 0, fichier)
pourcentage(Y , nb_cluster, labels , dict_path , 1, fichier)
pourcentage(Y , nb_cluster, labels , dict_path , 2, fichier)
# y = clus.fit_predict(X)
# pourcentage(Y , nb_cluster, y , dict_path , 0, fichier)
# pourcentage(Y , nb_cluster, y , dict_path , 1, fichier)
# pourcentage(Y , nb_cluster, labels , dict_path , 2, fichier)


##########################################################################################################################
############################################ CLUSTERING 6 CLASSES ########################################################
##########################################################################################################################

print 'Clustering 6 classes : \n'

nb_cluster = 6

#KMEANS non initialise 6 classes
clus = MiniBatchKMeans(n_clusters = nb_cluster, init='k-means++',  batch_size=700,
                                  n_init=10, max_no_improvement=10, verbose=0)
f = open(fichier, "a")
f.write("KMEANS non initialise 6 clusters\n")
f.close()
clus.fit(X)
labels = clus.labels_
pourcentage(Y , nb_cluster, labels , dict_path , 2, fichier)


#KMEANS initialise 6 classes
sous = initialisation_centres(nb_cluster, X)
clus = MiniBatchKMeans(n_clusters = nb_cluster, init=sous,  batch_size=700,
                                  n_init=10, max_no_improvement=10, verbose=0)
f = open(fichier, "a")
f.write("KMEANS initialise 6 clusters\n")
f.close()
clus.fit(X)
labels = clus.labels_
pourcentage(Y , nb_cluster, labels , dict_path , 2, fichier)

#Agglomerative clustering 6 classes
clus = AgglomerativeClustering(nb_cluster,affinity='cosine',linkage='complete')
f = open(fichier, "a")
f.write("Agglomerative clustering 6 clusters\n")
f.close()
clus.fit(X)
labels = clus.labels_
pourcentage(Y , nb_cluster, labels , dict_path , 2, fichier)
#y = clus.fit_predict(X)
#pourcentage(Y , nb_cluster, y , dict_path , 2, fichier)

##########################################################################################################################
############################################# AUTRES ALGORITHMES #########################################################
##########################################################################################################################

print 'Clustering MEAN-SHIFT : \n'

clus = cluster.MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=1)
f = open(fichier, "a")
f.write("MeanShift\n")
f.close()
clus.fit(X)
labels = clus.labels_
nmax = max(labels) +1
pourcentage(Y , nmax, labels , dict_path , 0, fichier)
pourcentage(Y , nmax, labels , dict_path , 1, fichier)
pourcentage(Y , nmax, labels , dict_path , 2, fichier)



# algorithmes qui ne marchent pas avec mfcc : a tester?
# clus = cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None)
# clus = cluster.spectral_clustering(affinity, n_clusters=8, n_components=None, eigen_solver=None, random_state=None, n_init=10, eigen_tol=0.0, assign_labels='kmeans')
# clus = cluster.SpectralClustering(n_clusters=3, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None)
#clus = cluster.ward_tree(X, connectivity=None, n_components=None, n_clusters=None, return_distance=False)uster.SpectralClustering(n_clusters=3, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None)