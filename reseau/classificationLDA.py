import numpy as np
from sklearn.lda import LDA
from sklearn.cross_validation import KFold
import pickle

def load_maps(fname):
    '''loads a map dictionary'''
    maps = pickle.load(open(fname, 'rb'))
    return maps


def pretraitementMatrice (liste_dictionnaires = [], liste_categories = [], liste_phonemes = []):

 tableau = np.array(liste_dictionnaires[0][liste_categories[0]][liste_phonemes[0]])
 nb_exemple,nb_carte,lign,col=tableau.shape
 Mat = []
 Reference = []
 for num in range(nb_carte):
     Matinter = []
     for inddict,dict in enumerate(liste_dictionnaires):
        for indcat,cat in enumerate(liste_categories):
            for indpho,pho in enumerate(liste_phonemes):
                for ex in range(nb_exemple):
                    Matinter.append((dict[cat][pho][ex][num]).flatten())
                    if num == 0:
                        Reference.append([inddict,indcat ,indpho])
     Mat.append(Matinter)
 Reference = np.array(Reference)
 Mat = np.array(Mat)
 return Mat, Reference #128*560*152 = 128*(4*4*35)*152 = nb_cartes *(nb_dict*nb_categories*nb_pho*nb_exemplaire)*nb_coeff_dans_carte

map_file='maps/BREF80_l_conv2_35maps_th0.500000.pkl'
FR = load_maps(map_file)

map_file='maps/PHONIM_l_conv2_35maps_th0.001000.pkl'
JA = load_maps(map_file)

M,R = pretraitementMatrice([FR, JA],FR.keys(),['R'])
print M.shape, R.shape
print M[0][0][0]

# donnees pour entrainer le classifieur
X = np.concatenate(M)

# labels pour entrainer le classifieur
Y = []
for i in range(len(R[:,1])):
  if (R[:,1][i]==0 or R[:,1][i]==1): #1 pour les corrects, 0 pour les incorrects
   Y.append([1]*M.shape[0])
  else:
   Y.append([0]*M.shape[0])
Y = np.concatenate(np.array(Y))

print Y.shape
print X.shape

kf = KFold(n=len(Y), n_folds=5, shuffle=False,
                               random_state=None)
scores = []
for train_index, test_index in kf:
    print len(train_index),len(test_index)
    X_train, X_test = X[train_index,:],X[test_index,:]
    Y_train, Y_test = Y[train_index],Y[test_index]
    cl = LDA()
    cl.fit(X_train,Y_train)
    scores.append(cl.score(X_test,Y_test))

print scores




