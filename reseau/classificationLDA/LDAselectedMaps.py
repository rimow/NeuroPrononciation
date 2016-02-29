import pickle
import numpy as np
from utiles_classification import *
from mapsAnalysis.utiles import *
from process_activation_maps import load_maps

# LDA classification, as input we have either set of flatten maps, or a set of maps, grouping according to phonemes, languages or categories
# (we create different subsets of the initial data)
# We select the interesting windows related to clustering results
# Into the result file, score is -1 if there is no interesting maps fot the corresponding classification

#Files
mapconv1J_file='../maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
mapconv2J_file='../maps/PHONIM_l_conv2_35maps_th0.001000.pkl'
mapmp2J_file='../maps/PHONIM_l_mp2_35maps_th0.001000.pkl'
denseJ_file = '../maps/PHONIM_l_dense1_35maps_th0.001000.pkl'
mapconv1F_file='../maps/BREF80_l_conv1_35maps_th0.500000.pkl'
mapconv2F_file='../maps/BREF80_l_conv2_35maps_th0.500000.pkl'
mapmp2F_file='../maps/BREF80_l_mp2_35maps_th0.500000.pkl'
denseF_file = '../maps/BREF80_l_dense1_35maps_th0.500000.pkl'

#Load data
conv1J = load_maps(mapconv1J_file)
conv2J = load_maps(mapconv2J_file)
mp2J = load_maps(mapmp2J_file)
# denseJ = load_maps(denseJ_file)
conv1F = load_maps(mapconv1F_file)
conv2F = load_maps(mapconv2F_file)
mp2F = load_maps(mapmp2F_file)
# denseF = load_maps(denseF_file)

f_res = open('results_LDA_selected_maps','w')

dics = [conv1J,conv1F]
ph = ['R']
dict_goodmaps = goodmaps([],30,'conv1')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[0])
if dict_goodmaps[0]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_fr_jap,5)
f_res.write('conv1 FR JAP, phoneme R, FR/JAP classification : '+str(score)+'\n')

dics = [conv1F]
ph = ['R','v']
dict_goodmaps = goodmaps([],30,'conv1')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[1])
if dict_goodmaps[1]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_r_v,5)
f_res.write('conv1 FR, phonemes R and v, R/v classification, : '+str(score)+'\n')

dics = [conv1F,conv1J]
ph = ['R']
dict_goodmaps = goodmaps([],30,'conv1')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[2])
if dict_goodmaps[2]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_c_inc,5)
f_res.write('conv1 FR JAP, phoneme R, correct/incorrect classification: '+str(score)+'\n')

dics = [conv1F,conv1J]

ph = ['v']
dict_goodmaps = goodmaps([],30,'conv1')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[3])
if dict_goodmaps[3]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_c_inc,5)
f_res.write('conv1 FR JAP, phoneme v, correct/incorrect classification : '+str(score)+'\n')


######################################################################################################
dics = [conv2J,conv2F]
ph = ['R']
dict_goodmaps = goodmaps([],30,'conv2')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[0])
if dict_goodmaps[0]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_fr_jap,5)
f_res.write('conv2 FR JAP, phoneme R, FR/JAP classification : '+str(score)+'\n')

dics = [conv2F]
ph = ['R','v']
dict_goodmaps = goodmaps([],30,'conv2')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[1])
if dict_goodmaps[1]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_r_v,5)
f_res.write('conv2 FR, phonemes R and v, R/v classification, : '+str(score)+'\n')

dics = [conv2F,conv2J]
ph = ['R']
dict_goodmaps = goodmaps([],30,'conv2')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[2])
if dict_goodmaps[2]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_c_inc,5)

f_res.write('conv2 FR JAP, phoneme R, correct/incorrect classification: '+str(score)+'\n')

dics = [conv2F,conv2J]
ph = ['v']
dict_goodmaps = goodmaps([],30,'conv2')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[3])
if dict_goodmaps[3]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_c_inc,5)
f_res.write('conv2 FR JAP, phoneme v, correct/incorrect classification : '+str(score)+'\n')

######################################################################

dics = [mp2J,mp2F]
ph = ['R']
dict_goodmaps = goodmaps([],30,'mp2')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[0])
if dict_goodmaps[0]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_fr_jap,5)
f_res.write('mp2 FR JAP, phoneme R, FR/JAP classification : '+str(score)+'\n')

dics = [mp2F]
ph = ['R','v']
dict_goodmaps = goodmaps([],30,'mp2')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[1])
if dict_goodmaps[1]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_r_v,5)
f_res.write('mp2 FR, phonemes R and v, R/v classification, : '+str(score)+'\n')

dics = [mp2F,mp2J]
ph = ['R']
dict_goodmaps = goodmaps([],30,'mp2')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[2])
if dict_goodmaps[2]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_c_inc,5)
f_res.write('mp2 FR JAP, phoneme R, correct/incorrect classification: '+str(score)+'\n')

dics = [mp2F,mp2J]
ph = ['v']
dict_goodmaps = goodmaps([],30,'mp2')
X,Y_c_inc,Y_r_v,Y_fr_jap = getData_goodmaps(dics,conv1J.keys(),ph,dict_goodmaps[3])
if dict_goodmaps[3]==[]:
  score = -1
else:
  score = LDAmeanScore(X,Y_c_inc,5)
f_res.write('mp2 FR JAP, phoneme v, correct/incorrect classification : '+str(score)+'\n')

f_res.close()





