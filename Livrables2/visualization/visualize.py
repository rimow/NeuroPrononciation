from dim_reduction  import *
from mapsClustering.MapsClustering import *
from classificationLDA.utiles_classification import *
from mapsAnalysis.utiles import *
#Files

mapconv1J_file='../maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
mapconv1F_file='../maps/BREF80_l_conv1_35maps_th0.500000.pkl'
mapconv2J_file="../maps/PHONIM_l_conv2_35maps_th0.001000.pkl"
mapconv2F_file='../maps/BREF80_l_conv2_35maps_th0.500000.pkl'
map2conv1J_file='../maps/maps2/PHONIM_l_conv1_35maps_th0.001000.pkl'
map2conv1F_file='../maps/maps2/BREF80_l_conv1_35maps_th0.500000.pkl'
#Load data
Status = False
if Status:
    conv1J = load_maps(mapconv1J_file)
    conv1F = load_maps(mapconv1F_file)
    conv2J = load_maps(mapconv2J_file)
    conv2F = load_maps(mapconv2F_file)
map2conv1J = load_maps(map2conv1J_file)
map2conv1F = load_maps(map2conv1F_file)

if Status:
	dict_goodmaps = goodmaps("conv2","kmeansNonInit")
	ph = ['R']
	dics = [conv2J,conv2F]
	X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv2J.keys(),ph,dict_goodmaps[0])
	print best_dimension(X,"mle")
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
	
	dict_goodmaps = goodmaps("conv1","kmeansNonInit")
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
	




### Fbank python conv1

#######conv1########################################
### 0:JA(bleu)  1:FR(rough)
### 0:R (bleu)  1:V(rough)
### 0:correct(bleu) 1:incorrect(rough)

#dict_goodmaps = goodmaps("mp2","kmeansNonInit")
dict_goodmaps={0:[7,23,33,59,96,107,111],1:[1,23,60,107],2:[1,107],3:[],4:[]}
#print dict_goodmaps
ph = ['R']
dics = [map2conv1J,map2conv1F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,map2conv1J.keys(),ph,dict_goodmaps[0])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_fr_jap,"map2-conv1-R_FR_JA")
for key in map2conv1J.keys():
	print len(map2conv1J[key]['v'])
ph = ['v']
dics = [map2conv1J,map2conv1F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,map2conv1J.keys(),ph,dict_goodmaps[1])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_fr_jap,"map2-conv1-V_FR_JA")

ph = ['R','v']
dics = [map2conv1F]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,map2conv1J.keys(),ph,dict_goodmaps[2])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_r_v,"map2-conv1-V_R_FR")


ph = ['R']
dics = [map2conv1J]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,map2conv1J.keys(),ph,dict_goodmaps[3])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_c_inc,"map2-conv1-R_JA_Correct_Incorrect")

ph = ['v']
dics = [map2conv1J]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,map2conv1J.keys(),ph,dict_goodmaps[4])
X_reduced_PCA = dim_reduction_PCA(X,2)
plot_data(X_reduced_PCA,Y_c_inc,"map2-conv1-V_JA_Correct_Incorrect")
