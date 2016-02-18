import pickle
import numpy as np
from utiles_classification import *
from mapsAnalysis.SupprimerCartesVides import *
from dim_reduction import *
from mapsAnalysis.utiles import *
#Files
mapconv1J_file='../maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
mapconv1F_file='../maps/BREF80_l_conv1_35maps_th0.500000.pkl'

#Load data
conv1J = load_maps(mapconv1J_file)
conv1F = load_maps(mapconv1F_file)

# creation de la matrice de donnees et du vecteur des labels
ph = ['R','v']
dics = [conv1J,conv1F]

#X,Y_c_inc,Y_r_v = getData_onePerMap(dics,conv1J.keys(),ph)
X,Y_c_inc,Y_r_v = getData_maps(dics,conv1J.keys(),ph)

# Verification des dimensions
# Xsh = X.shape
# convSh = np.array(conv1F['correct_OK']['v']).shape
# print 'X : ',Xsh
# print 'getData_maps : ',len(dics)*len(ph)*len(conv1F.keys())*convSh[0]*convSh[1]
# print 'getData_onePerMap : ',len(dics)*len(ph)*len(conv1F.keys())*convSh[0]


# Test du classifieur LDA et resultat apres validation croisee
n_folds = 5
#LDAmeanScore(X,Y_c_inc,n_folds)
#LDAmeanScore(X,Y_r_v,n_folds)
X_reduced_PCA = dim_reduction_PCA(X,2)
X_reduced_LDA = dim_reduction_LDA(X,Y_c_inc,2)

plot_data(X_reduced_PCA,Y_r_v,"PCA")
plot_data(X_reduced_LDA,Y_r_v,"LDA")

