import pickle
import numpy as np
from utiles_classification import *
import mapsAnalysis.utiles
from process_activation_maps import load_maps


# Classification LDA en prenant en compte qu'une seule carte d'activation, tests et ecritures des resultats.
# Fait pour toutes les combinaisons de dictionnaires, toutes les combinaisons de phonemes, pour tous les types de classifications,

#Files
mapconv1J_file='maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
mapconv2J_file='maps/PHONIM_l_conv2_35maps_th0.001000.pkl'
mapmp2J_file='maps/PHONIM_l_mp2_35maps_th0.001000.pkl'
denseJ_file = 'maps/PHONIM_l_dense1_35maps_th0.001000.pkl'
mapconv1F_file='maps/BREF80_l_conv1_35maps_th0.500000.pkl'
mapconv2F_file='maps/BREF80_l_conv2_35maps_th0.500000.pkl'
mapmp2F_file='maps/BREF80_l_mp2_35maps_th0.500000.pkl'
denseF_file = 'maps/BREF80_l_dense1_35maps_th0.500000.pkl'

#Load data
conv1J = load_maps(mapconv1J_file)
conv2J = load_maps(mapconv2J_file)
mpJ = load_maps(mapmp2J_file)
#denseJ = load_maps(denseJ_file)
conv1F = load_maps(mapconv1F_file)
conv2F = load_maps(mapconv2F_file)
mpF = load_maps(mapmp2F_file)
# denseF = load_maps(denseF_file)

def getY(R,type):
    """
    :param R: matrice contenant dans la 1ere colonne l'indice de la langue, dans la 2eme l'indice de la categorie, dans la 3eme
              l'indice du phoneme. De taille (nb_languesc*nb_phonemes*nb_categories*nb_exemplaires)*3
    :param type: c_inc pour seprarer correct et incorrect, fr_jap pour separer francais et japonais, r_v pour separer r et v
    :return: Y de taille nb_languesc*nb_phonemes*nb_categories*nb_exemplaires, contenant le vecteur faisant la separation voulue
    """
    if type=='c_inc':
        Y = R[:,1]
        for i in range(len(Y)):
            if Y[i]==0 or Y[i]==1:
                Y[i]=1
            else:
                Y[i]=0
    elif type == 'fr_jap':
        Y = R[:,0]
    elif type == 'r_v':
        Y = R[:,2]
    return Y

# liste des ensembles de dictionnaires que l'on veut prendre en donnees d'entree
#liste_dics = [[mpF],[mpJ],[mpF,mpJ]]
#dics_labels = ['mp2F','mp2J','mp2F_mp2J']

liste_dics = [[mpF],[mpJ],[mpF,mpJ],[conv1F],[conv1J],[conv1F,conv1J],[conv2F],[conv2J],[conv2F,conv2J]]
dics_labels = ['mp2F','mp2J','mp2F_mp2J','conv1F','conv1J','conv1F_conv1J','conv2F','conv2J','conv2F_conv2J']
#liste_dics = [[conv1F],[conv1J],[conv1F,conv1J]]
#dics_labels = ['conv1F','conv1J','conv1F_conv1J']
#liste_dics = [[conv2F],[conv2J],[conv2F,conv2J]]
#dics_labels = ['conv2F','conv2J','conv2F_conv2J']

# Listes des phonemes que l'on veut traiter
liste_phonemes = [['R'],['v'],['R','v']]
phonemes_labels = ['R','v','Rv']

#Categories que l'on veut traiter, on prend tout
categories = mpJ.keys()

#Type de classification que l'on veut faire
types = ['c_inc','r_v','fr_jap']

n_folds = 5
f_bon_scores = open('./resultats_mapPerMap/LDA_mapPerMap_bon_resultats','w')
for idic,dics in enumerate(liste_dics):
    for iph,phs in enumerate(liste_phonemes):
        for type in types:
            f_res = open('./resultats_mapPerMap/LDA_mapPerMap_'+dics_labels[idic]+'_'+phonemes_labels[iph]+'_'+type, 'w')

            Mat,R = pretraitementMatrice(liste_dictionnaires = dics, liste_categories = categories, liste_phonemes = phs)
            Y = getY(R,type)
            i = 0
            for X in Mat:
                if 100.*len([j for j in np.concatenate(X) if j==0])/len(np.concatenate(X))>97: #Pour enlever les matrices vides ou presque vides
                    f_res.write(str(i)+' Carte nulle \n')
                else:
                  score = LDAmeanScore(X,Y,n_folds)
                  f_res.write(str(i)+' Score moyen:'+str(score)+'\n')
                  if score>75 and score<100:
                      f_bon_scores.write(str(i)+'_'+dics_labels[idic]+'_'+phonemes_labels[iph]+'_'+type+' : '+str(score)+'\n')
                i = i + 1
            f_res.close()

f_bon_scores.close()

