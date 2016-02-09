from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from phonemesAnalysis.analyse import *
from phonemesAnalysis.featuresGeneration import *
from phonemesAnalysis.utiles import *

# Script faisant l'analyse d'un signal audio en utilisant les fbank (version python)
# Transformation en fbank, clustering des features vectors, analyse des resultats (proportion de voises/non voises
# dans les classes, histogrammes, ...
# Possibilite de choisir differente methode de clustering, possibilite d'en ajouter ...

fft_span = 0.02
hop_span = 0.01
n_mels = 40
fmin = 50
fmax = 8000
nb_classes = 3
path = "./data/Bref80_L4/Bref80_L4M01.wav"
path_aligned = "./data/Bref80_L4/Bref80_L4M01.txt"
path_dict = "./data/classement"

dict = getPhonemeDict(path_dict)

X = fbank(path,fft_span,hop_span,n_mels,fmin,fmax)
#np.save('/home/guery/Documents/n7/ProjetLong/data/X.npy',X)

# Si le fichier est sauvegarde
#X = np.load('./data/X.npy')

#alignement de chaque vecteur avec un phoneme
Y = getY(X,path_aligned,hop_span=hop_span)

#Alignement de chaque vecteur avec differentes categories de phonemes
Y_cons_voy = getY_v_non_v(Y,dict,0)
Y_v_non_v = getY_v_non_v(Y,dict,1)
Y_categories = getY_v_non_v(Y,dict,2)

#Kmeans non supervise
clus = KMeans(nb_classes)

#Kmeans supervise, initialisation des centres. Exemple ci-dessous vaut pour 3 classes. Adaptable
#means = getMeanVectors(X,Y_v_non_v) #ici, centres initialise au vecteur moyen des voyelles, consonnes et silences.
#centres = np.array([means[0],means[1],means[2]])
#clus = KMeans(nb_classes,init=centres)

#Clustering hierarchique
#clus = AgglomerativeClustering(nb_classes,affinity='cosine',linkage='complete')
#clus = AgglomerativeClustering(nb_classes,affinity='l1',linkage='average')
#clus = AgglomerativeClustering(nb_classes,affinity='l1',linkage='complete')

Y_cluster = clus.fit_predict(X) #prediction

#Affichage des resultats
histogrammesPhonemes(nb_classes,Y_cluster,Y)
printRatiosVoise(nb_classes,Y_cluster,Y_v_non_v)
printRatiosCategories(nb_classes,Y_cluster,Y_categories)
printRatiosConsonnes(nb_classes , Y_cluster, Y_cons_voy)
