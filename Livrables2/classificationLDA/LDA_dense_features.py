import pickle
import numpy as np
from utiles_classification import *
from process_activation_maps import load_maps


# LDA on dense parameters

denseJ_file = '../maps/PHONIM_l_dense1_35maps_th0.001000.pkl'
denseF_file = '../maps/BREF80_l_dense1_35maps_th0.500000.pkl'
denseJ = load_maps(denseJ_file)
denseF = load_maps(denseF_file)

# liste des ensembles de dictionnaires que l'on veut prendre en donnees d'entree
all_liste_dics = [[denseF],[denseJ],[denseF,denseJ]]
labels_dictionnaire = ['FR','JAP','FR JAP'] #Pour ecrire dans le fichier

# Listes des phonemes que l'on veut traiter
all_listes_phonemes = [['R'],['v'],['R','v']]
labels_phonemes = [' R ',' v ',' R v '] #Pour ecrire dans le fichier

#Categories que l'on veut traiter, on prend tout en general
all_categories = denseF.keys()

#Type de classification que l'on veut faire
types = ['c_inc','r_v','fr_jap']

a_ignorer = []
l_cartes = []
n_folds = 5
f_res = open('../resultats/resultats_LDA/matlab_fbank/LDA_resultats_dense.txt', 'w')

for i,dics in enumerate(all_liste_dics):
    for j,ph in enumerate(all_listes_phonemes):
        for type in types:
              X,Y_c_inc,Y_r_v, Y_fr_jap = getData_dense(dics,all_categories,ph)
              if type=='c_inc':
                Y = Y_c_inc
              elif type=='r_v':
                Y = Y_r_v
              else:
                Y = Y_fr_jap
              score = LDAmeanScore(X,Y,n_folds)
              if score<100:
               f_res.write('Dictionnaires :'+labels_dictionnaire[i]+' Phonemes :'+labels_phonemes[j]+' Type de classification :'+type+'\n')
               f_res.write('Score :'+str(score)+'\n')
f_res.close()