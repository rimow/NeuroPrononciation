import pickle
import numpy as np
from utiles_classification import *
from utils_maps import *

# LDA classification, as input we have either set of flatten maps, or a set of maps, grouping according to phonemes, languages or categories
# (we create different subsets of the initial data)
# Results (Kfold mean scores) are printed into a text file



mapconv1F_file_newFbank = '../../maps/newFbank/BREF80_l_conv1_35maps_th0.500000.pkl'
mapconv2F_file_newFbank = '../../maps/newFbank/BREF80_l_conv2_35maps_th0.500000.pkl'


#Load data
conv1F_newFbank = load_maps(mapconv1F_file_newFbank)
conv2F_newFbank = load_maps(mapconv2F_file_newFbank)


# liste des ensembles de dictionnaires que l'on veut prendre en donnees d'entree
all_liste_dics = [[conv1F_newFbank],[conv2F_newFbank]]
labels_dictionnaire = ['conv1F fbank python','conv2F fbank python']

# Listes des phonemes que l'on veut traiter
liste_phonemes = [['R','v'],['b','l'],['a','R','b','v','l']]
labels_phonemes = [' R v ',' b l ',' a R v b l '] #Pour ecrire dans le fichier

#Categories que l'on veut traiter, on prend tout en general
all_categories = conv1F_newFbank.keys()
print all_categories
#Type de classification que l'on veut faire
types = ['r_v']

a_ignorer = []
l_cartes = []
n_folds = 5
f_res = open('LDA_resultats_fbank_python_conv1-2_dim10_5phones.txt', 'w')
dim_reduction = 10
indices_corrects = [0]

for i,dics in enumerate(all_liste_dics):
    for j,ph in enumerate(liste_phonemes):
        for type in types:
            f_res.write('Dictionnaires :'+labels_dictionnaire[i]+' Phonemes :'+labels_phonemes[j]+' Type de classification :'+type+'\n')
            score1,score2 = ldaClassification(dics,liste_phonemes=ph,liste_categories=all_categories,num_cartes=l_cartes,a_ignorer=a_ignorer,n_folds=n_folds,type=type,dim_reduction=dim_reduction,indices_corrects=indices_corrects)
            f_res.write('Score de la classification, une donnee=une carte : '+str(score2)+'\nScore de la classification, une donnee=un ensemble de cartes : '+str(score1)+'\n \n')

f_res.close()
