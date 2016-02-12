from phonemesAnalysis.utiles import *
from phonemesAnalysis.analyse import *
from phonemesAnalysis.featuresGeneration import *
import numpy as np

#Tests de certaines fonctions d'analyse.py et d'utiles.py

#######################################################
path_dico = './test_dictionnaire'
dict = getPhonemeDict(path_dico)
if dict['a'][0] ==1  and dict['a'][1] == 2 and dict['b'][0] ==3  and dict['b'][1] == 4 and dict['b'][2] == 5:
    print 'TEST getPhonemeDict OK'

path_aligned = './test_aligned'
X = np.zeros((3,2))
Y = getY(X,path_aligned,1)
if Y[0]=='a' and Y[1]=='b':
    print 'TEST getY OK'

######################################################
dict = {'a':[1,0,1],'b':[0,0,1],'c':[0,0,0]}
type_separation = 2
Y = ['a','b','c','a','b','c']
Y_v_non_v = getY_v_non_v(Y , dict , type_separation)
if list(Y_v_non_v)==[1,1,0,1,1,0]:
    print 'TEST getY_v_non_v OK'

#######################################################
dict = {'a':[1,0,1],'b':[0,0,1]}
X = np.array([[1,1,1],[2,1,1],[3,1,1]])
Y = ['a','b','a']
X_c, Y_c = getConsonnes(X,Y,dict)
if X_c[0][0]==2 and Y_c==['b']:
    print 'TEST getConsonnes OK'

#######################################################
X = np.array([[1,1],[0,1],[1,1]])
classes = [0,0,1]
mean = getMeanVectors(X,classes)
if mean[0][0]==0.5 and mean[0][1]==1 and mean[1][0]==1 and mean[1][1]==1:
    print 'Test getMeanVectors OK'

#######################################################


