import pickle
import numpy as np
from classificationLDA.utiles_classification import *
from process_activation_maps import *

# LDA classification, as input we have either set of flatten maps, or a set of maps, grouping according to phonemes, languages or categories
# (we create different subsets of the initial data)
# Results (Kfold mean scores) are printed into a text file


mapconv1F_file_newFbank = '../../maps2/BREF80_l_conv1_35maps_th0.500000.pkl'
mapconv1J_file_newFbank = '../../maps2/PHONIM_l_conv1_35maps_th0.001000.pkl'

#Load data
conv1F_newFbank = load_maps(mapconv1F_file_newFbank)
conv1J_newFbank = load_maps(mapconv1J_file_newFbank)


# liste des ensembles de dictionnaires que l'on veut prendre en donnees d'entree
all_liste_dics = [[conv1F_newFbank]]
labels_dictionnaire = ['conv1F fbank python']

# Listes des phonemes que l'on veut traiter
liste_phonemes = [['R','v'],['b','l'],['R','b','v','l']]
labels_phonemes = [' R v ',' b l ','R v b l '] #Pour ecrire dans le fichier

#Categories que l'on veut traiter, on prend tout en general
all_categories = conv1F_newFbank.keys()
liste_categories = [all_categories,['OK']]
categories_labels = [' all categories ',' only correct categories ']

#Type de classification que l'on veut faire
types = ['r_v']

a_ignorer = []
l_cartes = []
n_folds = 5
f_res = open('../resultats_temp/LDA_resultats_fbank_python_conv1_bestDim_4phones.txt', 'w') # To modify according to the repository
dim_reduction = -1
indices_corrects = [0,1] #respectivement aux categories

for i,dics in enumerate(all_liste_dics):
    for j,ph in enumerate(liste_phonemes):
      for icat,cat in enumerate(liste_categories):
        for type in types:
            f_res.write('Dictionnaires :'+labels_dictionnaire[i]+categories_labels[icat]+' Phonemes :'+labels_phonemes[j]+' Type de classification :'+type+'\n')
            score1,score2 = ldaClassification(dics,liste_phonemes=ph,liste_categories=all_categories,num_cartes=l_cartes,a_ignorer=a_ignorer,n_folds=n_folds,type=type,dim_reduction=dim_reduction,indices_corrects=indices_corrects)
            f_res.write('Score de la classification, une donnee=une carte : '+str(score2)+'\nScore de la classification, une donnee=un ensemble de cartes : '+str(score1)+'\n \n')

f_res.close()
