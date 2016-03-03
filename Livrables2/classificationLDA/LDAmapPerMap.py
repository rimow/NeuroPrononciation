import pickle
import numpy as np
from utiles_classification import *
from mapsAnalysis.utiles import *
from process_activation_maps import load_maps


# Classification LDA en prenant en compte qu'une seule carte d'activation, tests et ecritures des resultats.
# Fait pour toutes les combinaisons de dictionnaires, toutes les combinaisons de phonemes, pour tous les types de classifications,

#Files
mapconv1J_file='../maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
mapconv2J_file='../maps/PHONIM_l_conv2_35maps_th0.001000.pkl'
mapmp2J_file='../maps/PHONIM_l_mp2_35maps_th0.001000.pkl'
denseJ_file = '../maps/PHONIM_l_dense1_35maps_th0.001000.pkl'
mapconv1F_file='../maps/BREF80_l_conv1_35maps_th0.500000.pkl'
mapconv2F_file='../maps/BREF80_l_conv2_35maps_th0.500000.pkl'
mapmp2F_file='../maps/BREF80_l_mp2_35maps_th0.500000.pkl'
denseF_file = '../maps/BREF80_l_dense1_35maps_th0.500000.pkl'
mapconv1F_file_newFbank = '../maps2/BREF80_l_conv1_35maps_th0.500000.pkl'
mapconv1J_file_newFbank = '../maps2/PHONIM_l_conv1_35maps_th0.001000.pkl'

#Load data
conv1J = load_maps(mapconv1J_file)
conv2J = load_maps(mapconv2J_file)
mpJ = load_maps(mapmp2J_file)
#denseJ = load_maps(denseJ_file)
conv1F = load_maps(mapconv1F_file)
conv2F = load_maps(mapconv2F_file)
mpF = load_maps(mapmp2F_file)
# denseF = load_maps(denseF_file)
#conv1F_newFbank = load_maps(mapconv1F_file_newFbank)
#conv1J_newFbank = load_maps(mapconv1J_file_newFbank)

# liste des ensembles de dictionnaires que l'on veut prendre en donnees d'entree
#liste_dics = [[mpF],[mpJ],[mpF,mpJ]]
#dics_labels = ['mp2F','mp2J','mp2F_mp2J']
#liste_dics = [[conv1F],[conv1J],[conv1F,conv1J]]
#dics_labels = ['conv1F','conv1J','conv1F_conv1J']
#liste_dics = [[conv2F],[conv2J],[conv2F,conv2J]]
#dics_labels = ['conv2F','conv2J','conv2F_conv2J']
#liste_dics = [[conv1F_newFbank],[conv1J_newFbank],[conv1F_newFbank,conv1J_newFbank]]
#dics_labels = ['conv1F fbank python','conv1J fbank python','conv1FJ fbank python']

liste_dics = [[mpF],[mpJ],[mpF,mpJ],[conv1F],[conv1J],[conv1F,conv1J],[conv2F],[conv2J],[conv2F,conv2J]]
dics_labels = ['mp2F','mp2J','mp2F_mp2J','conv1F','conv1J','conv1F_conv1J','conv2F','conv2J','conv2F_conv2J']


# Listes des phonemes que l'on veut traiter
liste_phonemes = [['R'],['v'],['R','v']]
phonemes_labels = ['R','v','Rv']

#Categories que l'on veut traiter, on prend tout
categories = mpJ.keys()

#Type de classification que l'on veut faire
types = ['c_inc','r_v','fr_jap']

n_folds = 5
dim_reduction = -1

f_bon_scores = open('./resultats_temp/details_resultats_mapPerMap/LDA_mapPerMap_bon_resultats','w') # To modify according to the repository
f_scores_max = open('./resultats_temp/details_resultats_mapPerMap/LDA_mapPerMap_resultats_max','w') # To modify according to the repository

for idic,dics in enumerate(liste_dics):
    for iph,phs in enumerate(liste_phonemes):
        for type in types:
            f_res = open('./resultats_temp/details_resultats_mapPerMap/LDA_mapPerMap_'+dics_labels[idic]+'_'+phonemes_labels[iph]+'_'+type, 'w') # To modify according to the repository

            Mat,R = pretraitementMatrice(liste_dictionnaires = dics, liste_categories = categories, liste_phonemes = phs)
            Y = getY(R,type)
            i = 0
            scores = []
            for X in Mat:
                if 100.*len([j for j in np.concatenate(X) if j==0])/len(np.concatenate(X))>97: #Pour enlever les matrices vides ou presque vides
                    f_res.write(str(i)+' Carte nulle \n')
                else:
                  score = LDAmeanScore(X,Y,n_folds,dim_reduction=dim_reduction)
                  scores.append(score)
                  f_res.write(str(i)+' Score moyen:'+str(score)+'\n')
                  if score>75 and score<100:
                      f_bon_scores.write(str(i)+'_'+dics_labels[idic]+'_'+phonemes_labels[iph]+'_'+type+' : '+str(score)+'\n')

                i = i + 1
            f_res.close()
            max_s = max(scores)
            if max_s==100:
                ind_max = 'all'
            else:
              ind_max = [i for i,s in enumerate(scores) if s==max_s ]
            f_scores_max.write(dics_labels[idic]+'_'+phonemes_labels[iph]+'_'+type+'Carte :'+str(ind_max)+' : '+str(max_s)+'\n')

f_bon_scores.close()
f_scores_max.close()
