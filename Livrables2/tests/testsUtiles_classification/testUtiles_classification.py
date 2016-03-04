from classificationLDA.utiles_classification import *
import numpy as np
from process_activation_maps import load_maps

# utiles_classification Tests

dic = {}
dic_ok = {}
dic_pasok = {}
dic_ok['a'] = np.array([[[0,0],[0,0]],[[0,0],[0,0]]])
dic_ok['b'] = np.array([[[1,1],[1,1]],[[1,1],[1,1]]])
dic_pasok['a'] = np.array([[[2,2],[2,2]],[[2,2],[2,2]]])
dic_pasok['b'] = np.array([[[3,3],[3,3]],[[3,3],[3,3]]])
dic['OK'] = dic_ok
dic['pasOK'] = dic_pasok


# Tests getData_maps
X, Y_c_inc, Y_r_v, Y_fr_jap = getData_maps(liste_dictionnaires = [dic], liste_categories = ['OK','pasOK'], liste_phonemes = ['a','b'],a_ignorer=[],liste_cartes=[],indices_corrects=[0])
resX = np.array([[0,0], [0, 0], [0,0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [3, 3], [3, 3], [3, 3], [3, 3]])
resYc = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
resYr = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,])
resYf = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
if ((X == resX).all() and (Y_c_inc == resYc).all() and (Y_r_v == resYr).all() and (Y_fr_jap == resYf).all()) :
    print 'getData_maps Test 1 OK'
else:
    print 'getData_maps Test 1 pas OK'

X, Y_c_inc, Y_r_v, Y_fr_jap = getData_maps(liste_dictionnaires = [dic], liste_categories = ['OK','pasOK'], liste_phonemes = ['a','b'],a_ignorer=[0],liste_cartes=[],indices_corrects=[0])
resX = np.array([ [0,0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]])
resYc = np.array([1, 1, 1, 1, 0, 0, 0, 0])
resYr = np.array([ 0, 0, 1, 1, 0, 0, 1, 1,])
resYf = np.array([0, 0, 0, 0, 0, 0, 0, 0])
if ( (X == resX).all() and (Y_c_inc == resYc).all() and (Y_r_v == resYr).all() and (Y_fr_jap == resYf).all() ) :
    print 'getData_maps Test 2 OK'
else:
    print 'getData_maps Test 2 pas OK'

X, Y_c_inc, Y_r_v, Y_fr_jap = getData_maps(liste_dictionnaires = [dic], liste_categories = ['OK','pasOK'], liste_phonemes = ['a','b'],a_ignorer=[],liste_cartes=[1],indices_corrects=[0])
resX = np.array([ [0,0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]])
resYc = np.array([1, 1, 1, 1, 0, 0, 0, 0])
resYr = np.array([ 0, 0, 1, 1, 0, 0, 1, 1,])
resYf = np.array([0, 0, 0, 0, 0, 0, 0, 0])
if ( (X == resX).all() and (Y_c_inc == resYc).all() and (Y_r_v == resYr).all() and (Y_fr_jap == resYf).all() ) :
    print 'getData_maps Test 3 OK'
else:
    print 'getData_maps Test 3 pas OK'

#Tests getData_onePerMap

X, Y_c_inc, Y_r_v, Y_fr_jap = getData_onePerMap(liste_dictionnaires = [dic], liste_categories = ['OK','pasOK'], liste_phonemes = ['a','b'],a_ignorer=[],liste_cartes=[],indices_corrects=[0])
resX = np.array([[0,0,0,0], [0,0,0, 0], [1, 1,1, 1], [1, 1,1, 1], [2, 2,2, 2], [2, 2,2, 2], [3, 3,3, 3], [3, 3,3, 3]])
resYc = np.array([1, 1, 1, 1, 0, 0, 0, 0])
resYr = np.array([0,  0,  1,  1,  0,  0,  1,  1])
resYf = np.array([0, 0, 0, 0, 0, 0, 0, 0])
if ((X == resX).all() and (Y_c_inc == resYc).all() and (Y_r_v == resYr).all() and (Y_fr_jap == resYf).all()) :
    print 'getData_onePerMap Test 1 OK'
else:
    print 'getData_onePerMap Test 1 pas OK'

X, Y_c_inc, Y_r_v, Y_fr_jap = getData_onePerMap(liste_dictionnaires = [dic], liste_categories = ['OK','pasOK'], liste_phonemes = ['a','b'],a_ignorer=[0],liste_cartes=[],indices_corrects=[0])
resX = np.array([[0,0], [0,0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]])
resYc = np.array([1, 1, 1, 1, 0, 0, 0, 0])
resYr = np.array([0,  0,  1,  1,  0,  0,  1,  1])
resYf = np.array([0, 0, 0, 0, 0, 0, 0, 0])
if ((X == resX).all() and (Y_c_inc == resYc).all() and (Y_r_v == resYr).all() and (Y_fr_jap == resYf).all()) :
    print 'getData_onePerMap Test 2 OK'
else:
    print 'getData_onePerMap Test 2 pas OK'

X, Y_c_inc, Y_r_v, Y_fr_jap = getData_onePerMap(liste_dictionnaires = [dic], liste_categories = ['OK','pasOK'], liste_phonemes = ['a','b'],a_ignorer=[],liste_cartes=[1],indices_corrects=[0])
resX = np.array([[0,0], [0,0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]])
resYc = np.array([1, 1, 1, 1, 0, 0, 0, 0])
resYr = np.array([0,  0,  1,  1,  0,  0,  1,  1])
resYf = np.array([0, 0, 0, 0, 0, 0, 0, 0])
if ((X == resX).all() and (Y_c_inc == resYc).all() and (Y_r_v == resYr).all() and (Y_fr_jap == resYf).all()) :
    print 'getData_onePerMap Test 3 OK'
else:
    print 'getData_onePerMap Test 3 pas OK'

# Test LDA classification
dic = {}
dic_ok = {}
dic_pasok = {}
dic_ok['a'] = np.array([[[0,1],[2,0]],[[0,3],[4,0]],[[0,8],[4,8]],[[0,8],[8,0]]])
dic_ok['b'] = np.array([[[1,1],[1,6]],[[5,1],[1,7],[[0,1],[0,0]],[[0,5],[5,0]]]])
dic_pasok['a'] = np.array([[[2,1],[0,2]],[[2,7],[7,2],[[0,4],[4,4]],[[0,4],[4,4]]]])
dic_pasok['b'] = np.array([[[3,1],[0,3]],[[3,6],[6,3],[[0,9],[4,8]],[[0,8],[4,4]]]])
dic['OK'] = dic_ok
dic['pasOK'] = dic_pasok

score1,score2 = ldaClassification([dic],liste_phonemes=['a'],liste_categories=['OK','pasOK'],num_cartes=[],a_ignorer=[],n_folds=4,type='r_v',dim_reduction=0,indices_corrects=[0])
if score1==100 and score2==100:
    print 'ldaClassification Test 1 OK'
else:
    print 'ldaClassification Test 1 pas OK'

score1,score2 = ldaClassification([dic],liste_phonemes=['a','b'],liste_categories=['OK'],num_cartes=[],a_ignorer=[],n_folds=4,type='c_inc',dim_reduction=0,indices_corrects=[0])
if score1==100 and score2==100:
    print 'ldaClassification Test 2 OK'
else:
    print 'ldaClassification Test 2 pas OK'

score1,score2 = ldaClassification([dic],liste_phonemes=['a','b'],liste_categories=['OK'],num_cartes=[],a_ignorer=[],n_folds=4,type='fr_jap',dim_reduction=0,indices_corrects=[0])
if score1==100 and score2==100:
    print 'ldaClassification Test 3 OK'
else:
    print 'ldaClassification Test 3 pas OK'

# Tests getData_dense
dic = {}
dic_ok = {}
dic_pasok = {}
dic_ok['a'] = np.array([[0,0],[0,0]])
dic_ok['b'] = np.array([[1,1],[1,1]])
dic_pasok['a'] = np.array([[2,2],[2,2]])
dic_pasok['b'] = np.array([[3,3],[3,3]])
dic['OK'] = dic_ok
dic['pasOK'] = dic_pasok
X, Y_c_inc, Y_r_v, Y_fr_jap = getData_dense(liste_dictionnaires = [dic], liste_categories = ['OK','pasOK'], liste_phonemes = ['a','b'],indices_corrects=[0])
resX = np.array([[0,0], [0,0],  [1, 1], [1, 1], [2, 2],[2, 2], [3, 3], [3, 3]])
resYc = np.array([1, 1, 1, 1, 0, 0, 0, 0])
resYr = np.array([0,  0,   1,  1,0,  0,  1,  1])
resYf = np.array([0, 0, 0, 0, 0, 0, 0, 0])
if ((X == resX).all() and (Y_c_inc == resYc).all() and (Y_r_v == resYr).all() and (Y_fr_jap == resYf).all()) :
    print 'getData_dense Test 1 OK'
else:
    print 'getData_dense Test 1 pas OK'

X, Y_c_inc, Y_r_v, Y_fr_jap = getData_dense(liste_dictionnaires = [dic], liste_categories = ['OK','pasOK'], liste_phonemes = ['a','b'],indices_corrects=[0])
resX = np.array([[0,0], [0,0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]])
resYc = np.array([1, 1, 1, 1, 0, 0, 0, 0])
resYr = np.array([0,  0,  1,  1,  0,  0,  1,  1])
resYf = np.array([0, 0, 0, 0, 0, 0, 0, 0])
if ((X == resX).all() and (Y_c_inc == resYc).all() and (Y_r_v == resYr).all() and (Y_fr_jap == resYf).all()) :
    print 'getData_dense Test 2 OK'
else:
    print 'getData_dense Test 2 pas OK'

X, Y_c_inc, Y_r_v, Y_fr_jap = getData_dense(liste_dictionnaires = [dic], liste_categories = ['OK'], liste_phonemes = ['a','b'],indices_corrects=[0])
resX = np.array([[0,0], [0,0], [1, 1], [1, 1]])
resYc = np.array([1, 1, 1, 1])
resYr = np.array([0,  0,  1,  1])
resYf = np.array([0, 0, 0, 0])
if ((X == resX).all() and (Y_c_inc == resYc).all() and (Y_r_v == resYr).all() and (Y_fr_jap == resYf).all()) :
    print 'getData_dense Test 3 OK'
else:
    print 'getData_dense Test 3 pas OK'

# Tests getY
R = np.array([[2,1,3],[0,0,0],[5,1,6]])
Y = getY(R,'c_inc',[0,1])
if (Y==np.array([1,1,1])).all():
    print 'getY Test 1 OK'
else:
    print 'getY Test 1 pas OK'

R = np.array([[2,1,3],[0,0,0],[5,1,6]])
Y = getY(R,'r_v',[0,1])
if (Y==np.array([3,0,6])).all():
    print 'getY Test 2 OK'
else:
    print 'getY Test 2 pas OK'

R = np.array([[2,1,3],[0,0,0],[5,1,6]])
Y = getY(R,'fr_jap',[0,1])
if (Y==np.array([2,0,5])).all():
    print 'getY Test 3 OK'
else:
    print 'getY Test 3 pas OK'

# Test LDA mean score
X = np.transpose(np.array([range(100),range(100)]))
Y = np.array([0]*100)
n_folds = 5
score = LDAmeanScore(X,Y,n_folds,dim_reduction=0)
if score == 100.:
    print 'LDAmeanScore Test 1 OK'
else:
    print 'LDAmeanScore Test 1 pas OK'

n_folds = len(Y)+1
score = LDAmeanScore(X,Y,n_folds,dim_reduction=0)
if score == -1:
    print 'LDAmeanScore Test 2 OK'
else:
    print 'LDAmeanScore Test 2 pas OK'

# Tests getPhone
X = np.array([[1,2,3],[4,5,6]])
X_phone = np.array([[0,1,0,0],[0,0,1,0]])
Y_phones = [0,1]
X1,Y1 = selectPhones(X,X_phone,Y_phones,[0,1,1,0])
if (X1==X).all() and (Y1==np.array(Y_phones)).all():
    print 'selectPhones Test 1 OK'
else:
    print 'selectPhones Test 1 pas OK'

X1,Y1 = selectPhones(X,X_phone,Y_phones,[0,0,1,0])
if (X1==np.array([4,5,6])).all() and (Y1==np.array([1])).all():
    print 'selectPhones Test 2 OK'
else:
    print 'selectPhones Test 2 pas OK'

# Tests getPhone Labels
X1,Y1 = getPhonesLabels(X,X_phone,phones=[0,1,1,0])
if (X1==X).all() and (Y1==np.array([1,2])).all():
    print 'getPhonesLabels Test 1 OK'
else:
    print 'getPhonesLabels Test 1 pas OK'

X = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
X_phone = np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1],[0,1,0,0]])
X1,Y1 = getPhonesLabels(X,X_phone,phones=[1,1,1,0])
if (X1==np.array([[1,2,3],[4,5,6],[7,8,9],[13,14,15]])).all() and (Y1==np.array([1,2,0,1])).all():
    print 'getPhonesLabels Test 2 OK'
else:
    print 'getPhonesLabels Test 2 pas OK'


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
