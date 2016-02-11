from sklearn import cluster
from analyse import pourcentage, histogrammesPhonemes
from featuresGeneration import mfcc
from minibatch import initialisation_centres
from utiles import getY, getPhonemeDict

#extraction de features######################################################################################


fft_span = 0.02
hop_span = 0.01
n_mels = 13
fmin = 50
fmax = 8000
path = "/home/marianne/Developpement/Bref80_L4M01.wav"
X = mfcc(path, fft_span, hop_span, n_mels)
nb_features,nb_vectors = X.shape


# #Initialisation du tableau contenant les donnees d'alignement################################################

Y = getY(X,"/home/marianne/Developpement/Bref80_L4M01.aligned", hop_span)


#clustering #################################################################################################"
print 'Clustering: \n'
#realisation de la matrice verite-terrain
dict_path = "classement"
dict = getPhonemeDict(dict_path)
#nombre de clusters
nb_cluster = 3
#chemin du fichier ou on souhaite ecrire les resultats
fichier = "data/test.csv"
#difference qu'on souhaite evaluer sur les phonemes : separation consonnes/voyelles = 0, voisee/non-voisee = 1, categories = 2
#type_phoneme = 2

#kmeans initialise ou non
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


#algorithmes qui ne marchent pas avec mfcc : a tester?
#clus = cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None)
#clus = cluster.spectral_clustering(affinity, n_clusters=8, n_components=None, eigen_solver=None, random_state=None, n_init=10, eigen_tol=0.0, assign_labels='kmeans')
#clus = cluster.SpectralClustering(n_clusters=3, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None)
#clus = cluster.ward_tree(X, connectivity=None, n_components=None, n_clusters=None, return_distance=False)
