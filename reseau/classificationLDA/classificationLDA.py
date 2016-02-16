import pickle
import numpy as np
from utiles_classification import *

# Load data
mapconv1J_file='maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
mapconv2J_file='maps/PHONIM_l_conv2_35maps_th0.001000.pkl'
mapmp2J_file='maps/PHONIM_l_mp2_35maps_th0.001000.pkl'
denseJ_file = 'maps/PHONIM_l_dense1_35maps_th0.001000.pkl'
mapconv1F_file='maps/BREF80_l_conv1_35maps_th0.500000.pkl'
mapconv2F_file='maps/BREF80_l_conv2_35maps_th0.500000.pkl'
mapmp2F_file='maps/BREF80_l_mp2_35maps_th0.500000.pkl'
denseF_file = 'maps/BREF80_l_dense1_35maps_th0.500000.pkl'

conv1J = load_maps(mapconv1J_file)
conv2J = load_maps(mapconv2J_file)
mpJ = load_maps(mapmp2J_file)
denseJ = load_maps(denseJ_file)

conv1F = load_maps(mapconv1F_file)
conv2F = load_maps(mapconv2F_file)
mpF = load_maps(mapmp2F_file)
denseF = load_maps(denseF_file)

f_res = open('LDA_resultats.txt', 'w')
# Modifier selon la classification voulue
n_folds = 5
phonemes = ['R','v']
for ph in phonemes:
    f_res.write(ph+'\n')
    X,Y = getData_dense([denseF,denseJ],denseJ.keys(),ph)
    f_res.write('getData_dense, denseF, denseJ :'+str(LDAmeanScore(X,Y,n_folds))+'\n')
    X,Y = getData_onePerMap([conv1J,conv1F],conv1J.keys(),ph)
    f_res.write('getData_onePerMap, conv1F, conc1J :'+str(LDAmeanScore(X,Y,n_folds))+'\n')
    X,Y = getData_onePerMap([conv2J,conv2F],conv2J.keys(),ph)
    f_res.write('getData_onePerMap, conv2F, conc2J :'+str(LDAmeanScore(X,Y,n_folds))+'\n')
    X,Y = getData_onePerMap([mpF,mpJ],mpF.keys(),ph)
    f_res.write('getData_onePerMap, mpF, mpJ :'+str(LDAmeanScore(X,Y,n_folds))+'\n')
    X,Y = getData_maps([conv1J,conv1F],conv1J.keys(),ph)
    f_res.write('getData_maps, conv1F, conc1J :'+str(LDAmeanScore(X,Y,n_folds))+'\n')
    X,Y = getData_maps([conv2J,conv2F],conv2J.keys(),ph)
    f_res.write('getData_maps, conv2F, conc2J :'+str(LDAmeanScore(X,Y,n_folds))+'\n')
    X,Y = getData_maps([mpF,mpJ],mpJ.keys(),ph)
    f_res.write('getData_maps, mpF, mpJ :'+str(LDAmeanScore(X,Y,n_folds))+'\n')

print Y.shape
print X.shape



f_res.close()