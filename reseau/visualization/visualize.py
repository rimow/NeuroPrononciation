from dim_reduction  import *
from mapsClustering.MapsClustering import *
from classificationLDA.utiles_classification import *
#Files
mapconv1J_file='../maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
mapconv1F_file='../maps/BREF80_l_conv1_35maps_th0.500000.pkl'
mapconv2J_file="../maps/PHONIM_l_conv2_35maps_th0.001000.pkl"
mapconv2F_file='../maps/BREF80_l_conv2_35maps_th0.500000.pkl'
#Load data
conv1J = load_maps(mapconv1J_file)
conv1F = load_maps(mapconv1F_file)
conv2J = load_maps(mapconv2J_file)
conv2F = load_maps(mapconv2F_file)
# creation de la matrice de donnees et du vecteur des labels


#X,Y_c_inc,Y_r_v = getData_onePerMap(dics,conv1J.keys(),ph)
#X,Y_c_inc,Y_r_v = getData_maps(dics,conv1J.keys(),ph)

# Verification des dimensions
# Xsh = X.shape
# convSh = np.array(conv1F['correct_OK']['v']).shape
# print 'X : ',Xsh
# print 'getData_maps : ',len(dics)*len(ph)*len(conv1F.keys())*convSh[0]*convSh[1]
# print 'getData_onePerMap : ',len(dics)*len(ph)*len(conv1F.keys())*convSh[0]


###########################conv2##################################
### 0:JA(bleu)  1:FR(rough)
### 0:R (bleu)  1:V(rough)
### 0:correct(bleu) 1:incorrect(rough)

vide_KMNI, pFRJA_R_KMNI, pFRJA_V_KMNI, pFR_RV_KMNI, pCIC_R_KMNI, pCIC_V_KMNI, ind = MapsClustering("conv2", 559, "kmeansNonInit", False)
dict_goodmaps = goodmaps("conv2","kmeansNonInit",vide_KMNI,ind,30)
#print dict_goodmaps
ph = ['R']
dics = [conv2J,conv2F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv2J.keys(),ph,dict_goodmaps[0])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_fr_jap,"conv2-R_FR_JA")

ph = ['v']
dics = [conv2J,conv2F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv2J.keys(),ph,dict_goodmaps[1])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_fr_jap,"conv2-V_FR_JA")

ph = ['R','v']
dics = [conv2F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv2F.keys(),ph,dict_goodmaps[2])
print X
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_r_v,"conv2-V_R_FR")


ph = ['R']
dics = [conv2J]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv2J.keys(),ph,dict_goodmaps[3])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_c_inc,"conv2-R_JA_Correct_Incorrect")

ph = ['v']
dics = [conv2J]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv2J.keys(),ph,dict_goodmaps[4])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_c_inc,"conv2-V_JA_Correct_Incorrect")

#plot best maps

bestmaps = {0:[223],1:[81],2:[154],3:[223],4:[37]}

ph = ['R']
dics = [conv2J,conv2F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,bestmaps[0])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_fr_jap,"conv2-R_FR_JA_bestmaps")

ph = ['v']
dics = [conv2J,conv2F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,bestmaps[1])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_fr_jap,"conv2-V_FR_JA_bestmaps")

ph = ['R','v']
dics = [conv2F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,bestmaps[2])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_r_v,"conv2-V_R_FR_bestmaps")


ph = ['R']
dics = [conv2J]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,bestmaps[3])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_c_inc,"conv2-R_JA_Correct_Incorrect_bestmaps")

ph = ['v']
dics = [conv2J]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,bestmaps[4])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_c_inc,"conv2-V_JA_Correct_Incorrect_besmaps")


#######conv1########################################
### 0:JA(bleu)  1:FR(rough)
### 0:R (bleu)  1:V(rough)
### 0:correct(bleu) 1:incorrect(rough)

vide_KMNI, pFRJA_R_KMNI, pFRJA_V_KMNI, pFR_RV_KMNI, pCIC_R_KMNI, pCIC_V_KMNI, ind = MapsClustering("conv1", 559, "kmeansNonInit", False)
dict_goodmaps = goodmaps("conv1","kmeansNonInit",vide_KMNI,ind,30)
#print dict_goodmaps
ph = ['R']
dics = [conv1J,conv1F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[0])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_fr_jap,"conv1-R_FR_JA")

ph = ['v']
dics = [conv1J,conv1F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[1])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_fr_jap,"conv1-V_FR_JA")

ph = ['R','v']
dics = [conv1F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[2])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_r_v,"conv1-V_R_FR")


ph = ['R']
dics = [conv1J]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[3])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_c_inc,"conv1-R_JA_Correct_Incorrect")

ph = ['v']
dics = [conv1J]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[4])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_c_inc,"conv1-V_JA_Correct_Incorrect")

#plot best maps

bestmaps = {0:[50],1:[79],2:[19],3:[114],4:[5]}

ph = ['R']
dics = [conv1J,conv1F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,bestmaps[0])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_fr_jap,"conv1-R_FR_JA_bestmaps")

ph = ['v']
dics = [conv1J,conv1F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,bestmaps[1])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_fr_jap,"conv1-V_FR_JA_bestmaps")

ph = ['R','v']
dics = [conv1F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,bestmaps[2])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_r_v,"conv1-V_R_FR_bestmaps")


ph = ['R']
dics = [conv1J]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,bestmaps[3])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_c_inc,"conv1-R_JA_Correct_Incorrect_bestmaps")

ph = ['v']
dics = [conv1J]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,bestmaps[4])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_c_inc,"conv1-V_JA_Correct_Incorrect_besmaps")