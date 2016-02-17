import pickle
import numpy as np
from utiles_classification import *
from supprimerCartesVides import *

#Files
mapconv1J_file='maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
mapconv1F_file='maps/BREF80_l_conv1_35maps_th0.500000.pkl'

#Load data
conv1J = load_maps(mapconv1J_file)
conv1F = load_maps(mapconv1F_file)

# creation de la matrice de donnees et du vecteur des labels
ph = ['R']
#X,Y = getData_onePerMap([conv1J,conv1F],conv1J.keys(),ph)
X,Y = getData_maps([conv1J,conv1F],conv1J.keys(),ph)

# Test du classifieur LDA et resultat apres validation croisee
n_folds = 5
LDAmeanScore(X,Y,n_folds)

