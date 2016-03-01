import pickle
import numpy as np
from utiles_classification import *
from process_activation_maps import load_maps

# LDA classification, as input we have either set of flatten maps, or a set of maps, grouping according to phonemes, languages or categories
# (we create different subsets of the initial data)
# Results (Kfold mean scores) are printed into a text file

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
#conv2J = load_maps(mapconv2J_file)
#mpJ = load_maps(mapmp2J_file)
# denseJ = load_maps(denseJ_file)
conv1F = load_maps(mapconv1F_file)
#conv2F = load_maps(mapconv2F_file)
#mpF = load_maps(mapmp2F_file)
# denseF = load_maps(denseF_file)
#conv1F_newFbank = load_maps(mapconv1F_file_newFbank)
#conv1J_newFbank = load_maps(mapconv1J_file_newFbank)

# liste des ensembles de dictionnaires que l'on veut prendre en donnees d'entree
all_liste_dics = [[conv1F],[conv1J],[conv1J,conv1F]]
labels_dictionnaire = ['FR','JAP','FR JAP'] #Pour ecrire dans le fichier

# Listes des phonemes que l'on veut traiter
all_listes_phonemes = [['R'],['v'],['R','v']]
labels_phonemes = [' R ',' v ',' R v '] #Pour ecrire dans le fichier

#Categories que l'on veut traiter, on prend tout en general
all_categories = conv1F.keys()

#Type de classification que l'on veut faire
types = ['c_inc','r_v','fr_jap']

a_ignorer = []
l_cartes = []
n_folds = 5
f_res = open('LDA_resultats_complet_conv1_dim10.txt', 'w') # To modify according to the repository
dim_reduction = 10

for i,dics in enumerate(all_liste_dics):
    for j,ph in enumerate(all_listes_phonemes):
        for type in types:
            f_res.write('Dictionnaires :'+labels_dictionnaire[i]+' Phonemes :'+labels_phonemes[j]+' Type de classification :'+type+'\n')
            score1,score2 = ldaClassification(dics,liste_phonemes=ph,liste_categories=all_categories,num_cartes=l_cartes,a_ignorer=a_ignorer,n_folds=n_folds,type=type,dim_reduction=dim_reduction)
            f_res.write('Score de la classification, une donnee=une carte : '+str(score2)+'\nScore de la classification, une donnee=un ensemble de cartes : '+str(score1)+'\n \n')

f_res.close()
