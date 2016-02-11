from phonemesAnalysis.utiles import *
from phonemesAnalysis.analyse import *
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from phonemesAnalysis.featuresGeneration import waveletsTransformContinue
from sklearn import cluster
import numpy as np


##########################################################################################################################
################################################### DONNEES ##############################################################
##########################################################################################################################

#Chemin du fichier ou on souhaite ecrire les resultats, peut s'ouvrir avec Excel
fichier = "../resultats/resultatsClustering/waveletsClustering.csv"

dt=0.01
dj=0.5
signalPath = '../data/Bref80_L4M01.wav' #A adapter suivant l'emplacement du fichier audio
path_aligned = "../data/Bref80_L4M01.aligned" #A adapter suivant l'emplacement du fichier d'alignement
path_dict = "../data/classement" #A adapter suivant l'emplacement du fichier de classement
dict = getPhonemeDict(path_dict) #realisation de la matrice verite-terrain

##########################################################################################################################
############################################ MATRICE DE CLUSTERING #######################################################
################### Matrice choisie pour le clustering: Matrice "moyenne" en utilisant l'ondelette Paul ##################
##########################################################################################################################

#Soit on effectue la transformation
# X = waveletsTransformContinue(signalPath,'paul',2,dt,dj,affichageSpectrogram=False)

#Soit on charge la matrice si elle est deja enregistree
X = np.load('../resultats/resultatsTransformation/paulmoy.npy')


nb_features,nb_vectors = X.shape
Y = getY(X,path_aligned, dt) #Initialisation du tableau contenant les donnees d'alignement

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
pourcentage(Y , nb_cluster, labels , path_dict , 0, fichier)
pourcentage(Y , nb_cluster, labels , path_dict , 1, fichier)

#KMEANS initialise 3 classes
sous = initialisation_centres(nb_cluster, X)
clus = MiniBatchKMeans(n_clusters = nb_cluster, init=sous,  batch_size=700,
                                  n_init=10, max_no_improvement=10, verbose=0)
f = open(fichier, "a")
f.write("KMEANS initialise 3 clusters\n")
f.close()
clus.fit(X)
labels = clus.labels_
pourcentage(Y , nb_cluster, labels , path_dict , 0, fichier)
pourcentage(Y , nb_cluster, labels , path_dict , 1, fichier)

#Agglomerative clustering 3 classes
clus = AgglomerativeClustering(nb_cluster,affinity='cosine',linkage='complete')
f = open(fichier, "a")
f.write("Agglomerative clustering 3 clusters\n")
f.close()
y = clus.fit_predict(X)
pourcentage(Y , nb_cluster, y , path_dict , 0, fichier)
pourcentage(Y , nb_cluster, y , path_dict , 1, fichier)


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
pourcentage(Y , nb_cluster, labels , path_dict , 2, fichier)


#KMEANS initialise 6 classes
sous = initialisation_centres(nb_cluster, X)
clus = MiniBatchKMeans(n_clusters = nb_cluster, init=sous,  batch_size=700,
                                  n_init=10, max_no_improvement=10, verbose=0)
f = open(fichier, "a")
f.write("KMEANS initialise 6 clusters\n")
f.close()
clus.fit(X)
labels = clus.labels_
pourcentage(Y , nb_cluster, labels , path_dict , 2, fichier)

#Agglomerative clustering 6 classes
clus = AgglomerativeClustering(nb_cluster,affinity='cosine',linkage='complete')
f = open(fichier, "a")
f.write("Agglomerative clustering 6 clusters\n")
f.close()
y = clus.fit_predict(X)
pourcentage(Y , nb_cluster, y , path_dict , 2, fichier)

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
pourcentage(Y , nmax, labels , path_dict , 1, fichier)
pourcentage(Y , nmax, labels , path_dict , 2, fichier)




# DBSCAN, ne marche pas sur les ondelettes
# print 'Clustering DBSCAN : \n'
#
# clus = cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None)
# f = open(fichier, "a")
# f.write("DBSCAN\n")
# f.close()
# y = clus.fit_predict(X)
# nmax = max(y) +1
# pourcentage(Y , nmax, y , path_dict , 1, fichier)
# pourcentage(Y , nmax, y , path_dict , 2, fichier)



# SPECTRAL CLUSTERING, ne fonctionne pas pour les ondelettes
#clus = cluster.spectral_clustering(affinity, n_clusters=8, n_components=None, eigen_solver=None, random_state=None, n_init=10, eigen_tol=0.0, assign_labels='kmeans')
#clus = cluster.SpectralClustering(n_clusters=3, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None)
