import pickle
import numpy as np
from utiles_classification import *
from numpy.random import rand
from process_activation_maps import *

# LDA classification, as input we have either set of flatten maps, or a set of maps, grouping according to phonemes, languages or categories
# (we create different subsets of the initial data)
# We select the interesting windows related to clustering results

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
categories = conv1J.keys()

#dim=0 if no reduction of dimension
dim = -1 #if reduction to the minimum dimension containing 80% of variability

score_moyen = 0.
nb_scores = 0.

f_res = open('./resultats_temp/random_selected_maps/results_selected_maps2_random-8','w')
#Conv1
nb_f = np.array(conv1F['correct_OK']['v'][0]).shape[0]


f_res.write('Conv1 \n')
interesting_maps = [5, 8, 11, 12, 16, 19, 21, 23, 24, 31, 33, 36, 41, 42, 50, 51, 55, 57, 58, 59, 61, 65, 74, 84, 86, 88, 91, 93, 94, 95, 98, 99, 102, 108, 113, 118, 119, 122]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv1J,conv1F],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, R : '+str(score)+'\n')

interesting_maps = [5, 8, 11, 12, 16, 19, 21, 23, 24, 31, 33, 36, 41, 42, 50, 51, 55, 57, 58, 59, 61, 65, 67, 74, 84, 86, 88, 91, 93, 94, 95, 98, 99, 102, 108, 118, 119, 122]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv1J,conv1F],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, R : '+str(score)+'\n')

interesting_maps = [8]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv1J,conv1F],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, R : '+str(score)+'\n')

interesting_maps = [8, 9, 19, 22, 24, 31, 51, 56, 57, 79, 84, 94, 105, 114, 123]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv1J,conv1F],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, v : '+str(score)+'\n')

interesting_maps = [19, 22, 31, 51, 57, 79, 94, 105]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv1J,conv1F],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, v : '+str(score)+'\n')

interesting_maps = [24]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv1F],categories,['R','v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_r_v,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR, R v : '+str(score)+'\n')


interesting_maps = [56, 94, 114, 119]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv1J],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_c_inc,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('JAP, R, correct/incorrect : '+str(score)+'\n')

interesting_maps = [56, 79, 114, 118, 119]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv1J],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_c_inc,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('JAP, R, correct/incorrect : '+str(score)+'\n')

interesting_maps = [8, 11, 16, 19, 31, 32, 41, 56, 59, 74, 94, 95, 114, 119]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv1J],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_c_inc,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('JAP, v , correct/incorrect :'+str(score)+'\n')

interesting_maps = [11, 16, 19, 59, 74, 94, 95, 119]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv1J],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_c_inc,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('JAP, v , correct/incorrect :'+str(score)+'\n')

interesting_maps = [46]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv1J],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_c_inc,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('JAP, v , correct/incorrect :'+str(score)+'\n')

#conv2
nb_f = np.array(conv2F['correct_OK']['v'][0]).shape[0]


f_res.write('Conv2 \n')
interesting_maps = [43, 81, 159, 192, 220, 223, 253, 255]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv2J,conv2F],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, R : '+str(score)+'\n')

interesting_maps = [43, 81, 136, 192, 205, 220, 223, 253, 255]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv2J,conv2F],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, R : '+str(score)+'\n')

interesting_maps = [92, 108, 127, 154, 182, 184, 216]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv2J,conv2F],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, R : '+str(score)+'\n')

interesting_maps = [81]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv2J,conv2F],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, v : '+str(score)+'\n')

interesting_maps = [81]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv2J,conv2F],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, v : '+str(score)+'\n')


interesting_maps = [2, 11, 41, 55, 92, 144, 218, 220, 238, 246]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv2J,conv2F],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, v : '+str(score)+'\n')

interesting_maps = [169, 216, 220]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv2F],categories,['R','v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_r_v,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR, R v : '+str(score)+'\n')

interesting_maps = [63]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([conv2J],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_c_inc,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('JAP, v, correct/incorrect : '+str(score)+'\n')

#mp2
nb_f = np.array(mp2F['correct_OK']['v'][0]).shape[0]

f_res.write('Mp2 \n')
interesting_maps = [20, 43, 44, 81, 85, 145, 150, 159, 192, 220, 223, 240, 253, 255]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2F,mp2J],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, R : '+str(score)+'\n')

interesting_maps = [43, 81, 85, 136, 150, 159, 192, 220, 223, 240, 253, 255]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2F,mp2J],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, R : '+str(score)+'\n')

interesting_maps = [136, 150, 183, 184, 205, 216, 217, 254, 255]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2F,mp2J],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, R : '+str(score)+'\n')

interesting_maps = [81]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2F,mp2J],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, R : '+str(score)+'\n')

interesting_maps = [81, 150]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2F,mp2J],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, v : '+str(score)+'\n')

interesting_maps = [27, 41, 43, 54, 85, 112, 160, 252]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2F,mp2J],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, v : '+str(score)+'\n')

interesting_maps = [81]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2F,mp2J],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_fr_jap,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR JAP, v : '+str(score)+'\n')

interesting_maps = [192, 216, 223]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2F],categories,['R','v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_r_v,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR, R v : '+str(score)+'\n')

interesting_maps = [166]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2F],categories,['R','v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_r_v,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('FR, R v : '+str(score)+'\n')

interesting_maps = [136, 150]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2J],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_c_inc,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('JAP, R, correct/incorrect : '+str(score)+'\n')

interesting_maps = [174]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2J],categories,['R'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_c_inc,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('JAP, R, correct/incorrect : '+str(score)+'\n')

interesting_maps = [97, 220]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2J],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_c_inc,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('JAP, v, correct/incorrect : '+str(score)+'\n')

interesting_maps = [220]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2J],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_c_inc,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('JAP, v, correct/incorrect : '+str(score)+'\n')

interesting_maps = [53, 118, 184, 222]
interesting_maps = [int(elt) for elt in list(np.round(nb_f*rand(len(interesting_maps))-1))]
X, Y_c_inc,Y_r_v,Y_fr_jap = getData_onePerMap([mp2J],categories,['v'],a_ignorer=[],liste_cartes=interesting_maps)
score = LDAmeanScore(X,Y_c_inc,5,dim_reduction=dim);score_moyen = score_moyen+score;nb_scores=nb_scores+1
f_res.write('JAP, v, correct/incorrect : '+str(score)+'\n')


score_moyen = score_moyen/nb_scores
f_res.write('\n Score moyen : '+str(score_moyen)+'\n')
f_res.close()

