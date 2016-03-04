import pickle
import numpy as np
from utiles_classification import *
from mapsAnalysis.utiles import *
from process_activation_maps import load_maps

# LDA classification using mfcc raw data

mfcc_phone = '../maps/Train_mfcc_phone_array_NORMED.pkl'
mfcc = load_maps(mfcc_phone)

Y_c_inc = np.array(mfcc['y'])
X = np.array(mfcc['X'])
X_phones = np.array(mfcc['X_phone'])
X = X.reshape((X.shape[0],X.shape[2]*X.shape[3])) #To put in lda

# All sets of data by phone
phone_sets = [[1,1,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,1],[1,0,1,0],[0,1,0,1],[1,1,1,0],[0,1,1,1],[1,0,1,1],[1,1,0,1]]
n_folds = 5
dim = -1 #call to best dimension
f_res = open('./resultats_temp/LDA_resultats_mfcc_phones.txt','w')

# Correct/incorrect classification
f_res.write('MFCC LDA classification \n2 classes : correct/incorrect \n')
for phones in phone_sets:
    X_t,Y_c_inc_t = selectPhones(X,X_phones,Y_c_inc,phones)
    score = LDAmeanScore(X_t,Y_c_inc_t,n_folds,dim_reduction=dim)
    f_res.write('Selected phones :'+str(phones)+' Score : '+str(score)+'\n')

# Phone category classification
f_res.write('MFCC LDA classification \nnb_phonemes classes (one class per phone) \n')
for phones in phone_sets:
    X_t,Y_ph = getPhonesLabels(X,X_phones,Y_c_inc,phones,correctOnly=False)
    score = LDAmeanScore(X_t,Y_ph,n_folds,dim_reduction=dim)
    f_res.write('Selected phones :'+str(phones)+' Score : '+str(score)+'\n')

# Phone category classification
f_res.write('MFCC LDA classification (Only on corrects phonemes) \nnb_phonemes classes (one class per phone) \n')
for phones in phone_sets:
    X_t,Y_ph = getPhonesLabels(X,X_phones,Y_c_inc,phones,correctOnly=True)
    score = LDAmeanScore(X_t,Y_ph,n_folds,dim_reduction=dim)
    f_res.write('Selected phones :'+str(phones)+' Score : '+str(score)+'\n')

f_res.close()


