import numpy as np
from sklearn import cluster
from analyse import pourcentage, histogrammesPhonemes
from analyseTableau import pourcentageTableau
from featuresGeneration import mfcc
from minibatch import initialisation_centres
from preprocess import create_reference
from utiles import getY, getPhonemeDict

#extraction de features######################################################################################


fft_span = 0.02
hop_span = 0.01
n_mels = 40
fmin = 50
fmax = 8000
path = "/home/marianne/Developpement/Bref80_L4M01.wav"
#X = fbank(path,fft_span,hop_span,n_mels,fmin,fmax)
X = mfcc(path, fft_span, hop_span, n_mels)
X = np.transpose(X)
nb_features,nb_vectors = X.shape




# #Initialisation du tableau contenant les donnees d'alignement################################################

Y = create_reference(X,"/home/marianne/Developpement/Bref80_L4M01.aligned")
Y = np.array(Y)


#clustering #################################################################################################"
print 'Clustering: \n'
#realisation de la matrice verite-terrain
dict_path = "classement"
dict = getPhonemeDict(dict_path)
#nombre de clusters
nb_cluster = 3
#chemin du fichier ou on souhaite ecrire les resultats
fichier = "data/pourcentages.csv"
#difference qu'on souhaite evaluer sur les phonemes : separation consonnes/voyelles = 0, voisee/non-voisee = 1, categories = 2
type_phoneme = 2

#kmeans initialise ou non
sous = initialisation_centres(nb_cluster, X.transpose())
clus = cluster.KMeans(n_clusters=nb_cluster, init=sous, n_init=50, max_iter=3000, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
#agglomerative clustering
#clus = cluster.AgglomerativeClustering(nb_cluster)
#meanShift
#clus = cluster.MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=1)


#ecrire le type d'algorithme qu'on utilise : KMEANS, MeanShift...
f = open(fichier, "a")
f.write("KMEANS\n")
f.close()

y = clus.fit_predict(X.transpose())
X2 = getY(X.transpose(),"/home/marianne/Developpement/Bref80_L4M01.aligned", 0.01)
nmax = max(y)+1

#si l'algorithme determine lui-meme le nombre de clusters mettre nmax au lieu de nb_cluster
pourcentageTableau(X2 , nb_cluster, y , dict_path , type_phoneme, fichier)
#pourcentage(X2 , nb_cluster, y , dict_path , type_phoneme)
histogrammesPhonemes(nb_cluster, y , X2)

#algorithmes qui ne marchent pas avec mfcc : a tester?
#clus = cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None)
#clus = cluster.spectral_clustering(affinity, n_clusters=8, n_components=None, eigen_solver=None, random_state=None, n_init=10, eigen_tol=0.0, assign_labels='kmeans')
#clus = cluster.SpectralClustering(n_clusters=3, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None)
#clus = cluster.ward_tree(X, connectivity=None, n_components=None, n_clusters=None, return_distance=False)