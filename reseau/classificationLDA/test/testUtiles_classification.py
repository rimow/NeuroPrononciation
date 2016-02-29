from classificationLDA.utiles_classification import *

####### test getData_goodmaps #######
mapconv1J_file='../../maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
mapconv1F_file='../../maps/BREF80_l_conv1_35maps_th0.500000.pkl'
mapconv2J_file="../../maps/PHONIM_l_conv2_35maps_th0.001000.pkl"
mapconv2F_file='../../maps/BREF80_l_conv2_35maps_th0.500000.pkl'
#Load data
conv1J = load_maps(mapconv1J_file)
conv1F = load_maps(mapconv1F_file)
conv2J = load_maps(mapconv2J_file)
conv2F = load_maps(mapconv2F_file)

#test case 1
ph = ['R']
dics = [conv1J,conv1F]
ind_goodmaps = []
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv2J.keys(),ph,ind_goodmaps)
if not X:
    print("Test Successes")
else:
    print("Test Fails")

#test case 2
ph = []
dics = [conv1J,conv1F]
ind_goodmaps = [1,2,3]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,ind_goodmaps)
if not X:
    print("Test Successes")
else:
    print("Test Fails")

#test case 3
ph = ['R']
dics = []
ind_goodmaps = [1,2,3]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,ind_goodmaps)
if not X:
    print("Test Successes")
else:
    print("Test Fails")

#test case 4
ph = ['R']
dics = [conv1F,conv1J]
ind_goodmaps = [1,2]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,ind_goodmaps)
if X.shape == (280,304):
    print("Test Successes")
else:
    print("Test Fails")

#test case 5
ph = ['R']
dics = [conv1F]
ind_goodmaps = [1]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,ind_goodmaps)
if X.shape == (140,152):
    print("Test Successes")
else:
    print("Test Fails")

#test case 6
ph = ['R','v']
dics = [conv2F]
ind_goodmaps = [1]
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,ind_goodmaps)
if X.shape == (280,18):
    print("Test Successes")
else:
    print("Test Fails")