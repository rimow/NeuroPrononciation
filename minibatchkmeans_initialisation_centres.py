from time import time
import matplotlib.pyplot as plt
import operator
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import scipy.io as sio
from sklearn.cluster import KMeans,MiniBatchKMeans
import numpy as np
from phonemesAnalysis.utiles import *
from phonemesAnalysis.analyse import *
from phonemesAnalysis.featuresGeneration import *
import Erreurs


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def include_period(p1,p2):
    if((p1[0]<=p2[0]) and (p1[1]>=p2[1])):
        return 1
    elif((p1[0]<=p2[0]) and (p1[1]<p2[1])):
        return 2
    else:
        return 3



def initialisation_centres(nb_centers, X):
    '''
    initialisation des centres de clusters
    :param nb_centers: le nombre de centres (de clusters) qu'on veut definir = 3 pour voise non voise silence et =6 pour occlusive, fricative, semi-consonne, silence, nasale et voyelle
    :param X: la matrice contenant les parametres nb_fenetres*nb_parametres
    :return: une sous matrice de taille nb_centres*nb_parametres qu'on peut rentrer en argument de kmeans par exemple pour l initialisation
    '''
    #
    #definition d'un tableau booleen pour faire l extraction des lignes de fband pour les considerer comme des centres de clusters
    boo = np.ones(3449, dtype=bool)
    boo = [False]*boo
    if nb_centers==3:
        boo[1]= True #silence
        boo[37]= True #lettre 'k' occlusive
        boo[83]=True  #voyelle 'e'
    elif nb_centers==6:
        boo[1]= True #silence
        boo[37]= True #lettre 'k' occlusive
        boo[83]=True  #voyelle 'e'
        boo[1498] = True # fricative 'f'
        boo[381] = True #nasale 'n'
        boo[157] = True #semi consonne H ui
    else:
        raise Erreurs.initialisationError('Erreur : vous ne pourrez choisir qu un nombre de clusters 3 ou 6, utilisez plutot la fonction initialisation_centres_utilisateur')
    sous_matrice = X[boo,:]
    return sous_matrice

def initialisation_centres_utilisateur(indices_centres,X):
    '''

    :param indices_centres: les indices dans la matrice X des fenetres qu'on va choisir comme centres pour les futurs clusters
    :param X: matrice des parametres nb_fenetres*nb_parametres
    :return: sous matrice de taille taille(indices_centres)*nb_parametres
    '''
    if X.shape[0]<len(indices_centres):
        raise Erreurs.initialisationError('Erreur : le nombre de centres que vous voulez definir est superieur au nombre de fenetres existants')

    boo = np.ones(3449, dtype=bool)
    boo = [False]*boo
    for i in indices_centres:
        boo[i]=True
    sous_matrice = X[boo,:]
    return sous_matrice



def minibatchkmeans_avec_initialisation_centres(X, n_clusters, path_classement, path_aligned, type_resultats):
    '''

    :param X: la matrice contenant les parametres, elle est de taille : n_samples*n_features (3449*40)
    :param n_clusters: 3 pour une clusterisation voise on voise silence et 6 pour les categories fricatives occlusives
    :param path_classement : chemin vers le fichier classement qui contient les classement des phonemes
    :param path_aligned : chemin vers le fichier aligned qui contient la verite terrain
    :param type_resultats : 0 : pour afficher les pourcentages des categories dans chaque classe, 1: pour afficher les histogrammes representant les phonemes de chaque classe
    :return: Affiche les pourcentages
    '''

    fband = X
    n_samples, n_features = fband.shape
    period = np.loadtxt(alignfile,delimiter=' ',usecols=(0,1))
    phoneme = np.loadtxt(alignfile,dtype= str ,delimiter=' ',usecols=[2])
    n_phoneme = len(np.unique(phoneme))
    cut_period = [[round(start,2),round(end,2)] for start,end in zip(drange(0,(n_samples+1)*0.01,0.01),drange(0.02,(n_samples+2)*0.01,0.01))]


    pho = [None]*n_samples


    i=0
    j=0

    while (i<n_samples and j<len(period)-1):
        if(include_period(period[j],cut_period[i])==1):
            pho[i] = phoneme[j]
            i+=1
        elif(include_period(period[j],cut_period[i])==2):
            pho[i] = phoneme[j]#+'+'+phoneme[j+1]
            j+=1
            i+=1
        elif (include_period(period[j],cut_period[i])==3):

            pho[i-1]=phoneme[j]
            j+=1
    while(i<n_samples):
        pho[i]=phoneme[-1]
        i+=1

    n_clusters = n_clusters

    try:
        sous = initialisation_centres(n_clusters, fband)
        #sous =initialisation_centres_utilisateur([1,37,83],fband)


        kmeans = MiniBatchKMeans(n_clusters = n_clusters, init=sous,  batch_size=700,
                              n_init=10, max_no_improvement=10, verbose=0)
        kmeans.fit(fband)


        labels = kmeans.labels_


        dict = getPhonemeDict(path_classement)
        Y =getY(X,path_aligned,0.01)
        #CoeffsHistogrammes(X,20,y_voise_non_voise,10,20)
        if type_resultats==0:
            pourcentage(Y , n_clusters, labels , path_classement , 2)
        if type_resultats==1:
            histogrammesPhonemes(n_clusters , labels , pho)

    except Erreurs.initialisationError as e :
        print(e.value)


signal, sampling_rate = librosa.load('1.wav') #load du fichier audio
fband = FourierTransform('1.wav', int(0.02*sampling_rate), int(0.01*sampling_rate))
alignfile = '11.aligned' # le fichier contenant les annotations expert
minibatchkmeans_avec_initialisation_centres(fband, 6, 'classement1', alignfile,0)

