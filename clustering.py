import numpy as np
import scipy as sc
import scipy.io.wavfile
import librosa
from librosa import feature
from librosa import filters
from librosa import util
import matplotlib.pyplot as plt
from sklearn import cluster
from fbank import fbank
from fonctions_utiles import getPhonemeDict

#extraction de features
fft_span = 0.02
hop_span = 0.01
n_mels = 40
fmin = 50
fmax = 8000
path = "/home/guery/Documents/n7/ProjetLong/data/Bref80_L4M01.wav"
X = fbank(path,fft_span,hop_span,n_mels,fmin,fmax)
X = np.transpose(X)
nb_features,nb_vectors = X.shape
print X.shape, nb_vectors

#Initialisation du tableau contenant les donnees d'alignement
beginning = []
end = []
phonemes = []
#lecture fichier alignement
alignement = open("/home/guery/Documents/n7/ProjetLong/data/Bref80_L4M01.aligned")
lines  = alignement.readlines()
alignement.close()
for line in lines:
    decomposed_line = line.split(' ')
    beginning.append(float(decomposed_line[0]))
    end.append(float(decomposed_line[1]))
    ph = decomposed_line[2].split('\n')
    phonemes.append(ph[0])

#Creation du vecteur Y contenant le phoneme correspondant pour chacun des vecteurs
phoneme_courant = 0;
Y = []
for i in range(nb_vectors):
    if (end[phoneme_courant]<hop_span*i and phoneme_courant<len(phonemes)-2): #on enleve le dernier qui ne compte pas
        phoneme_courant = phoneme_courant + 1
    #if not(phoneme_courant==len(phonemes)-2):
    Y.append(phonemes[phoneme_courant])   #On peut faire plus precis sans doute
Y = np.array(Y)
# print 'Y',Y
# print 'len Y',len(Y)
# print 'shape X',shape_X


#clustering #################################################################################################"

print 'Clustering sur toutes les lettres : \n'
#kmeans
dict_path = "/home/guery/Documents/n7/ProjetLong/data/classement"
dict = getPhonemeDict(dict_path)
Y_c_ou_v = np.array([i for i,j in enumerate(Y) if dict[j][0]!=2])  # On selectionne seulement quand il y a un veritable phoneme
X_c_ou_v = X[:,Y_c_ou_v]

kmean = cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
y = kmean.fit_predict(X_c_ou_v.transpose())
# print y
# print y.shape
classe_0_clus = np.array([j for (j,i) in enumerate(y) if i==0])
classe_1_clus = np.array([j for (j,i) in enumerate(y) if i==1])

dict_path = "/home/guery/Documents/n7/ProjetLong/data/classement"
dict = getPhonemeDict(dict_path)
y_cons_voy = []
for ph in Y[Y_c_ou_v]:
    y_cons_voy.append(dict[ph][0])
y_cons_voy = np.array(y_cons_voy) #0 : consonnes, 1 : voyelles
p1 = 100.*len([ i for i in y_cons_voy[classe_0_clus] if i == 0 ])/len(y_cons_voy) # % classe 0 et consonne
p2 = 100.*len([ i for i in y_cons_voy[classe_0_clus] if i == 1 ])/len(y_cons_voy) # % classe 0 et voyelle
p3 = 100.*len([ i for i in y_cons_voy[classe_1_clus] if i == 0 ])/len(y_cons_voy) # % classe 1 et consonne
p4 = 100.*len([ i for i in y_cons_voy[classe_1_clus] if i == 1 ])/len(y_cons_voy) # %  classe 1 et voyelle
print 'classe 0 et consonne :',p1,'\n classe 0 et voyelle :',p2,'\n classe 1 et consonne :',p3,'\nclasse 1 et voyelle :',p4,'\n'

y_voise_ou_non = []
for ph in Y[Y_c_ou_v]:
    y_voise_ou_non.append(dict[ph][1])
y_voise_ou_non = np.array(y_voise_ou_non) #0 : consonnes, 1 : voyelles
p21 = 100.*len([ i for i in y_voise_ou_non[classe_0_clus] if i == 0 ])/len(y_voise_ou_non) # % classe 0 et non voise
p22 = 100.*len([ i for i in y_voise_ou_non[classe_0_clus] if i == 1 ])/len(y_voise_ou_non) # % classe 0 et voise
p23 = 100.*len([ i for i in y_voise_ou_non[classe_1_clus] if i == 0 ])/len(y_voise_ou_non) # % classe 1 et non voise
p24 = 100.*len([ i for i in y_voise_ou_non[classe_1_clus] if i == 1 ])/len(y_voise_ou_non) # %  classe 1 et voise
print 'classe 0 et non voise :',p21,'\n classe 0 et voise :',p22,'\n classe 1 et non voise :',p23,'\nclasse 1 et voise :',p24,'\n'


#hierarchique
# hiera = cluster.AgglomerativeClustering(2)
# y = hiera.fit_predict(X.transpose())
# print y


#clustering sur les consonnes uniquement
print 'Clustering sur les consonnes : \n'
dict_path = "/home/guery/Documents/n7/ProjetLong/data/classement"
dict = getPhonemeDict(dict_path)
Y_consonnes = np.array([i for i,j in enumerate(Y) if dict[j][0]==0])
X_consonnes = X[:,Y_consonnes]


sh = X.shape
centres = np.zeros([sh[0],2])
#Trouver un vecteur d'un phoneme voise
pasTrouve = True
i = 0
while pasTrouve:
    if dict[Y[Y_consonnes[i]]][1]==0:
        centres[:,0]= X_consonnes[:,i]
        pasTrouve = False
    else:
        i=i+1

pasTrouve = True
i = 0
#Trouver un vecteur d'un phoneme non voise
while pasTrouve:
    if dict[Y[Y_consonnes[i]]][1]==1:
        centres[:,1]= X_consonnes[:,i]
        pasTrouve = False
    else:
        i=i+1

#kmeans
kmean = cluster.KMeans(n_clusters=2, init=np.transpose(centres), n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
y = kmean.fit_predict(X_consonnes.transpose())

#clustering hierarchique
#hiera = cluster.AgglomerativeClustering(2)
#y = hiera.fit_predict(X_consonnes.transpose())

# print y
# print y.shape
classe_0_clus = np.array([j for (j,i) in enumerate(y) if i==0])
classe_1_clus = np.array([j for (j,i) in enumerate(y) if i==1])

y_voise_ou_non = []
for ph in Y[Y_consonnes]:
    y_voise_ou_non.append(dict[ph][1])
y_voise_ou_non = np.array(y_voise_ou_non) #0 : non voise, 1 : voise
p21 = 100.*len([ i for i in y_voise_ou_non[classe_0_clus] if i == 0 ])/len(y_voise_ou_non) # % classe 0 et non voise
p22 = 100.*len([ i for i in y_voise_ou_non[classe_0_clus] if i == 1 ])/len(y_voise_ou_non) # % classe 0 et voise
p23 = 100.*len([ i for i in y_voise_ou_non[classe_1_clus] if i == 0 ])/len(y_voise_ou_non) # % classe 1 et non voise
p24 = 100.*len([ i for i in y_voise_ou_non[classe_1_clus] if i == 1 ])/len(y_voise_ou_non) # %  classe 1 et voise
print 'classe 0 et non voise :',p21,'\n classe 0 et voise :',p22,'\n classe 1 et non voise :',p23,'\nclasse 1 et voise :',p24,'\n'

