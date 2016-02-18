import pickle
import numpy as np
from utiles_classification import *
from supprimerCartesVides import *

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
liste_vide_c1 = []
liste_vide_c2 = []
liste_vide_mp = []

#Strategies de suppression de cartes vides : on selectionne les

#Strategie3
# nb_vides_c1 = 500
# nb_vides_c2 = 500
# nb_vides_mp = 200
# liste_vide_c1 = strategie_trois_l1([conv1F,conv1J], nb_vides_c1)
# liste_vide_c2 = strategie_trois_l1([conv2F,conv2J], nb_vides_c2)
# liste_vide_mp = strategie_trois_l1([mpF,mpJ], nb_vides_mp)
# np.save('mp_200.npy',liste_vide_mp)
# np.save('c1_200.npy',liste_vide_c1)
# np.save('c2_200.npy',liste_vide_c2)

#Strategie1
# liste_vide_c1 = strategie_une_l1([conv1F,conv1J])
# liste_vide_c2 = strategie_une_l1([conv2F,conv2J])
# liste_vide_mp = strategie_une_l1([mpF,mpJ])
# np.save('mp_s1.npy',liste_vide_mp)
# np.save('c1_s1.npy',liste_vide_c1)
# np.save('c2_s1.npy',liste_vide_c2)

#Strategie2
# liste_vide_c1 = strategie_deux_l1([conv1F,conv1J])
# liste_vide_c2 = strategie_deux_l1([conv2F,conv2J])
# liste_vide_mp = strategie_deux_l1([mpF,mpJ])
# np.save('mp_s2.npy',liste_vide_mp)
# np.save('c1_s2.npy',liste_vide_c1)
# np.save('c2_s2.npy',liste_vide_c2)



#liste_vide_c1 = np.load('c1.npy')
#liste_vide_c2 = np.load('c2.npy')
#liste_vide_mp = np.load('mp.npy')



#liste_vide = []
f_res = open('LDA_resultats_strategie3_500c1_500c2_200mp.txt', 'w')
# Modifier selon la classification voulue
n_folds = 5
phonemes = ['R','v']
for ph in phonemes:
    f_res.write(ph+'\n')
    X,Y_c_inc,Y_r_v = getData_dense([denseF,denseJ],denseJ.keys(),[ph])
    f_res.write('getData_dense, denseF, denseJ :'+str(LDAmeanScore(X,Y_c_inc,n_folds))+'\n')
    X,Y_c_inc,Y_r_v = getData_onePerMap([conv1J,conv1F],conv1J.keys(),[ph],liste_vide_c1)
    f_res.write('getData_onePerMap, conv1F, conc1J :'+str(LDAmeanScore(X,Y_c_inc,n_folds))+'\n')
    X,Y_c_inc,Y_r_v = getData_onePerMap([conv2J,conv2F],conv2J.keys(),[ph],liste_vide_c2)
    f_res.write('getData_onePerMap, conv2F, conc2J :'+str(LDAmeanScore(X,Y_c_inc,n_folds))+'\n')
    X,Y_c_inc,Y_r_v = getData_onePerMap([mpF,mpJ],mpF.keys(),[ph],liste_vide_mp)
    f_res.write('getData_onePerMap, mpF, mpJ :'+str(LDAmeanScore(X,Y_c_inc,n_folds))+'\n')
    X,Y_c_inc,Y_r_v = getData_maps([conv1J,conv1F],conv1J.keys(),[ph],liste_vide_c1)
    f_res.write('getData_maps, conv1F, conc1J :'+str(LDAmeanScore(X,Y_c_inc,n_folds))+'\n')
    X,Y_c_inc,Y_r_v = getData_maps([conv2J,conv2F],conv2J.keys(),[ph],liste_vide_c2)
    f_res.write('getData_maps, conv2F, conc2J :'+str(LDAmeanScore(X,Y_c_inc,n_folds))+'\n')
    X,Y_c_inc,Y_r_v = getData_maps([mpF,mpJ],mpJ.keys(),[ph],liste_vide_mp)
    f_res.write('getData_maps, mpF, mpJ :'+str(LDAmeanScore(X,Y_c_inc,n_folds))+'\n')

f_res.close()