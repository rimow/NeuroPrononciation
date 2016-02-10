import librosa
import numpy as np
import mlpy.wavelet as wave
import matplotlib.pyplot as plt

############################################################################
################################# C.W.T ####################################
############################################################################
def waveletsTransformContinue(signalPath, wf, wf_param, dt, dj):
    '''
    Calcule la transformee en ondelettes continue du signal
    :param signalPath: Le chemin du signal audio
    :param wf: La fonction de l'ondelette ('morlet', 'paul', 'dog')
    :param wf_param: Parametre de la l'ondelette
    :param dt: Pas
    :param dj: Resolution de l'echelle (plus dj est petit plus la resolution est fine)
    :return: la transformee en ondelettes continue du signal, matrice 40*len(signal)
    '''

    # Load the wav file, y is the data and sr the sampling frequency
    signal, fe = librosa.load(signalPath)

    scales = wave.autoscales(len(signal), dt=dt, dj=dj, wf=wf, p=wf_param)
    spec = wave.cwt(signal, dt=dt, scales=scales, wf=wf, p=wf_param)
    spec= np.abs(spec)
    spec=spec.transpose()
    return spec

############################################################################
##################### AMELIORATION RESULTATS C.W.T #########################
############################################################################

def moyennerMatrice(x):
    '''
    Effectue la moyenne sur les lignes suivant des fenetres de 20ms avec un saut de 10ms
    :param x: Matrice resultante de la transformee en ondelettes continue
    :return: Matrice 3449*40
    '''
    out=[]
    y=np.array(x)
    for i in range(0,762049):
        if i % 221 == 0:
            sousMatrice = np.array(y[i:i+441,:])
            moyenne = sousMatrice.mean(0)
            out.append(moyenne)
    out=np.array(out)
    return out

def concatenerVecteurs(x):
    '''
    Concatene les lignes correspondant aux fenetres de 20ms avec un saut de 10ms
    :param x: Matrice resultante de la transformee en ondelettes continue
    :return: Matrice 3447*17640
    '''
    out=[]
    y=np.array(x)
    for i in range(0,762049):
        if i % 221 == 0:
            sousMatrice = np.array(y[i:i+441,:])
            vectorise = sousMatrice.flatten()
            out.append(vectorise)
    out = out[:-2]
    out=np.array(out)
    return out

##########################
###### Jeu de Test #######
##########################
dt=0.01
dj=0.46

# spec = waveletsTransformContinue('1.wav', 'morlet', 8, dt, dj)
# np.save('spec.npy',spec)
# print(spec)
# print(spec.shape)

# spec=np.load('spec.npy')

# sm=moyennerMatrice(spec)
# np.save('sm.npy',sm)
sm= np.load('sm.npy')
# print(sm.shape)

# cv=concatenerVecteurs(spec)
# np.save('cv.npy',cv)
# cv=np.load('cv.npy')
# print(cv.shape)


############################################################################
################################ FIGURES ###################################
############################################################################

# print(np.min(spec, axis=1), np.max(spec, axis=1), np.mean(spec, axis=1), np.std(spec, axis=1))

# doPlot=True
# if doPlot:
#     fig = plt.figure(1)

    # ax2 = plt.subplot(2,1,1)
    # p2 = ax2.imshow(cv, origin='lower', aspect='auto', interpolation='nearest')

    # ax2 = plt.subplot(2,1,2)
    # p2 = ax2.imshow(20. * np.log(cv), origin='lower', aspect='auto', interpolation='nearest')

    # plt.show()
    # plt.savefig('matriceconcatenee.png')

# plt.imshow(np.abs(spec), origin='lower', aspect='auto', interpolation='nearest')
# plt.show()

############################################################################
############################## CLUSTERING ##################################
############################################################################

from analyse import *
from utiles import *
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from minibatchKmeans import minibatchkmeans

nb_classes = 3
hop_span = 0.01

#X = Matrice choisie pour le clustering
X=sm

audioPath = "1.wav"
path_aligned = "11.aligned"
path_dict = "classement"


dict = getPhonemeDict(path_dict)
Y = getY(X,path_aligned,hop_span=hop_span)
Y_v_non_v = getY_v_non_v(Y,dict,1)
y_cons_voy = getY_v_non_v(Y,dict,0)

#################
#### K-MEANS ####
#################

#Avec initialisation des centres, 3 classes voises/non voises(type_separation=1)
minibatchkmeans(X, 3 ,path_dict,path_aligned,0,1, True)

#Sans initialisation des centres, 3 classes voises/non voises
# minibatchkmeans(X, 3 ,path_dict,path_aligned,0,1, False)

#Avec initialisation des centres, 6 classes
# minibatchkmeans(X, 6 ,path_dict,path_aligned,0,2, True)

#Sans initialisation des centres, 6 classes
# minibatchkmeans(X, 6 ,path_dict,path_aligned,0,2, False)

##################################
#### AGGLOMERATIVE CLUSTERING ####
##################################
# clus = AgglomerativeClustering(nb_classes,affinity='cosine',linkage='complete')
#
#
# Y_cluster = clus.fit_predict(X)
# print set(Y_cluster)
# #histogrammesPhonemes(nb_classes,Y_cluster,Y)
#
# Y_v_non_v = getY_v_non_v(Y,dict,1)
# printRatiosVoise(nb_classes,Y_cluster,Y_v_non_v)
#
# Y_categories = getY_v_non_v(Y,dict,2)
# #printRatiosCategories(nb_classes,Y_cluster,Y_categories)
#
# printRatiosConsonnes(nb_classes , Y_cluster, y_cons_voy)
