from sklearn import cluster
from phonemesAnalysis.analyse import pourcentage
from phonemesAnalysis.featuresGeneration import fbank
from phonemesAnalysis.utiles import getY, getPhonemeDict, initialisation_centres


##########################################################################################################################
################################################### DONNEES ##############################################################
##########################################################################################################################

#Chemin du fichier ou on souhaite ecrire les resultats, peut s'ouvrir avec Excel
fichier = "../resultats/resultatsClustering/fbankClustering.csv"

fft_span = 0.02
hop_span = 0.01
n_mels = 13
fmin = 50
fmax = 8000
path = "../data/Bref80_L4M01.wav"
path_aligned = "../data/Bref80_L4M01.aligned" #A adapter suivant l'emplacement du fichier d'alignement
dict_path = "../data/classement"
dict = getPhonemeDict(dict_path) #realisation de la matrice verite-terrain

##########################################################################################################################
############################################ MATRICE DE CLUSTERING #######################################################
##########################################################################################################################


X = fbank(path,fft_span,hop_span,n_mels,fmin,fmax)
nb_features,nb_vectors = X.shape


#Initialisation du tableau contenant les donnees d'alignement################################################
Y = getY(X,"../data/Bref80_L4M01.aligned", hop_span)


##########################################################################################################################
#################################################### CLUSTERING ##########################################################
##########################################################################################################################
print 'Clustering 3 classes: \n'

#nombre de clusters
nb_cluster = 3

#difference qu'on souhaite evaluer sur les phonemes : separation consonnes/voyelles = 0, voisee/non-voisee = 1, categories = 2
#type_phoneme = 2


# #kmeans initialise ou non
clus = cluster.KMeans(n_clusters=nb_cluster, init='k-means++', n_init=50, max_iter=3000, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
f = open(fichier, "a")
f.write("KMEANS non initialise 3 clusters\n")
f.close()
y = clus.fit_predict(X)
pourcentage(Y , nb_cluster, y , dict_path , 0, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 1, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 2, fichier)

sous = initialisation_centres(nb_cluster, X)
clus = cluster.KMeans(n_clusters=nb_cluster, init=sous, n_init=50, max_iter=3000, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
f = open(fichier, "a")
f.write("KMEANS initialise 3 clusters\n")
f.close()
y = clus.fit_predict(X)
pourcentage(Y , nb_cluster, y , dict_path , 0, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 1, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 2, fichier)

#agglomerative clustering
clus = cluster.AgglomerativeClustering(nb_cluster)
f = open(fichier, "a")
f.write("Agglomerative clustering 3 clusters\n")
f.close()
y = clus.fit_predict(X)
pourcentage(Y , nb_cluster, y , dict_path , 0, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 1, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 2, fichier)


print 'Clustering 6 classes: \n'

nb_cluster = 6

#kmeans initialise ou non
clus = cluster.KMeans(n_clusters=nb_cluster, init='k-means++', n_init=50, max_iter=3000, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
f = open(fichier, "a")
f.write("KMEANS non initialise 6 clusters\n")
f.close()
y = clus.fit_predict(X)
pourcentage(Y , nb_cluster, y , dict_path , 0, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 1, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 2, fichier)

sous = initialisation_centres(nb_cluster, X)
clus = cluster.KMeans(n_clusters=nb_cluster, init=sous, n_init=50, max_iter=3000, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
f = open(fichier, "a")
f.write("KMEANS initialise 6 clusters\n")
f.close()
y = clus.fit_predict(X)
pourcentage(Y , nb_cluster, y , dict_path , 0, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 1, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 2, fichier)

#agglomerative clustering
clus = cluster.AgglomerativeClustering(nb_cluster)
f = open(fichier, "a")
f.write("Agglomerative clustering 6 clusters\n")
f.close()
y = clus.fit_predict(X)
pourcentage(Y , nb_cluster, y , dict_path , 0, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 1, fichier)
pourcentage(Y , nb_cluster, y , dict_path , 2, fichier)

#meanShift
clus = cluster.MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=1)
f = open(fichier, "a")
f.write("MeanShift\n")
f.close()
y = clus.fit_predict(X)
nmax = max(y) +1
pourcentage(Y , nmax, y , dict_path , 0, fichier)
pourcentage(Y , nmax, y , dict_path , 1, fichier)
pourcentage(Y , nmax, y , dict_path , 2, fichier)
