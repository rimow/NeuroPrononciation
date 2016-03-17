import pickle
import numpy as np
from utiles_classification import *
from process_activation_maps import load_maps

# One exemple of classification for understanding easily how it works

#Files
mapconv1J_file='../maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
mapconv1F_file='../maps/BREF80_l_conv1_35maps_th0.500000.pkl'

#Load data
conv1J = load_maps(mapconv1J_file)
conv1F = load_maps(mapconv1F_file)
ph = ['R']
dics = [conv1J]
cat = conv1J.keys()
l_cartes = [25, 56, 114, 120]
a_ignorer = []
n_folds = 5
type = 'c_inc'

ldaClassification(dics,ph,cat,l_cartes,a_ignorer,n_folds,type)

