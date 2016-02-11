from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from phonemesAnalysis.analyse import *
from phonemesAnalysis.featuresGeneration import *
from phonemesAnalysis.utiles import *

# Similaire a clustering_pythonFbank, mais prends plus de donnees.
# Au dela de deux signaux, le temps de calcul est tres long et n'aboutit pas sur une machine peu puissante

fft_span = 0.02
hop_span = 0.01
n_mels = 40
fmin = 50
fmax = 8000
nb_classes = 2
paths_wav = ['../data/Bref80_L4/Bref80_L4M01.wav',
              '../data/Bref80_L4/Bref80_L4M02.wav']
              #'../data/Bref80_L4/Bref80_L4M03.wav',
              #'../data/Bref80_L4/Bref80_L4M04.wav',
              #'../data/Bref80_L4/Bref80_L4M05.wav',
              #'../data/Bref80_L4/Bref80_L4M06.wav',
              #'../data/Bref80_L4/Bref80_L4M07.wav',
              #'../data/Bref80_L4/Bref80_L4M08.wav',
              #'../data/Bref80_L4/Bref80_L4M09.wav']

paths_aligned = ['../data/Bref80_L4/Bref80_L4M01.txt',
              '../data/Bref80_L4/Bref80_L4M02.txt']
              #'../data/Bref80_L4/Bref80_L4M03.txt',
              # '../data/Bref80_L4/Bref80_L4M04.txt',
              # '../data/Bref80_L4/Bref80_L4M05.txt',
              # '../data/Bref80_L4/Bref80_L4M06.txt',
              # '../data/Bref80_L4/Bref80_L4M07.txt',
              # '../data/Bref80_L4/Bref80_L4M08.txt',
              # '../data/Bref80_L4/Bref80_L4M09.txt']


path_dict = "../data/classement"

dict = getPhonemeDict(path_dict)


X,Y = fbankPlus(paths_wav,paths_aligned,fft_span,hop_span,n_mels,fmin,fmax)

#Alignement de chaque vecteur avec differentes categories de phonemes
Y_v_non_v = getY_v_non_v(Y,dict,1)
Y_categories = getY_v_non_v(Y,dict,2)
Y_cons_voy = getY_v_non_v(Y,dict,0)

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

#Mean shift
# clus = MeanShift()

Y_cluster = clus.fit_predict(X)
nb_classes = len(list(set(list(Y_cluster))))

#Affichage resultats
histogrammesPhonemes(nb_classes,Y_cluster,Y)
printRatiosVoise(nb_classes,Y_cluster,Y_v_non_v)
printRatiosConsonnes(nb_classes , Y_cluster, Y_cons_voy)
printRatiosCategories(nb_classes,Y_cluster,Y_categories)

#CoeffsHistogrammes(X , 40 , Y_v_non_v, 2 , 20)

