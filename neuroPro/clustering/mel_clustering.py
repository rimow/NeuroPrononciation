from phonemesAnalysis.utiles import *
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster
from phonemesAnalysis.analyse import pourcentage, histogrammesPhonemes
from phonemesAnalysis.featuresGeneration import *


##########################################################################################################################
################################################### DONNEES ##############################################################
##########################################################################################################################

#Chemin du fichier ou on souhaite ecrire les resultats, peut s'ouvrir avec Excel
fichier = "../resultats/resultatsClustering/melClustering.csv"

dt=0.01
dj=0.5
signal_path = '../data/Bref80_L4M01.wav'
path_aligned = "../data/Bref80_L4M01.aligned" #A adapter suivant l'emplacement du fichier d'alignement
path_dict = "../data/classement"
dict = getPhonemeDict(path_dict) #realisation de la matrice verite-terrain

##########################################################################################################################
############################################ MATRICE DE CLUSTERING #######################################################
##########################################################################################################################
n_fft=512
hop_length = 221
fmin = 50
fmax = 8000
n_mels = 40

#Soit on effectue la transformation
X = FourierTransform(signal_path, n_fft, hop_length,fmin, fmax, n_mels,affichage=False)

nb_features,nb_vectors = X.shape
Y = getY(X,path_aligned, dt) #Initialisation du tableau contenant les donnees d'alignement

##########################################################################################################################
############################################ CLUSTERING 3 CLASSES ########################################################
##########################################################################################################################

print 'Clustering 3 classes : \n'
print 'Clustering miniBatch K-means non supervise : \n'
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
print 'Clustering miniBatch k-means supervise : \n'
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
print 'Clustering agglomerative clustering : \n'
clus = cluster.AgglomerativeClustering(nb_cluster,affinity='cosine',linkage='complete')
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
print 'Clustering miniBatch k-means non supervise : \n'
clus = MiniBatchKMeans(n_clusters = nb_cluster, init='k-means++',  batch_size=700,
                                  n_init=10, max_no_improvement=10, verbose=0)
f = open(fichier, "a")
f.write("KMEANS non initialise 6 clusters\n")
f.close()
clus.fit(X)
labels = clus.labels_
pourcentage(Y , nb_cluster, labels , path_dict , 2, fichier)


#KMEANS initialise 6 classes
print 'Clustering miniBatch k-means supervise : \n'
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
print 'Aglomerative clustering : \n'
clus = cluster.AgglomerativeClustering(nb_cluster,affinity='cosine',linkage='complete')
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
y = clus.fit_predict(X)
nmax = max(y) +1
pourcentage(Y , nmax, y , path_dict , 1, fichier)
pourcentage(Y , nmax, y , path_dict , 2, fichier)

