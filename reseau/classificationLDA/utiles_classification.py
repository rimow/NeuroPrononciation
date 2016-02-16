import pickle
import numpy as np
from sklearn.lda import LDA
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from  sklearn.neighbors import KNeighborsClassifier

def load_maps(fname):
    '''loads a map dictionary'''
    maps = pickle.load(open(fname, 'rb'))
    return maps

def getData_dense(liste_dictionnaires = [], liste_categories = [], liste_phonemes = []):
 """
    :param liste_dictionnaires: liste de dictionnaires de vecteurs denses
    :param liste_categories: liste de categories
    :param liste_phonemes:  liste de phonemes
    :return: une matrice de parametre X ((nb_dic*nb_cat*nb_ph*nb_cartes)*taille_vecteur), et un vecteur de parametre Y (nb_dic*nb_cat*nb_ph*nb_cartes)
 """
 X = []
 Y = []
 for idic,dic in enumerate(liste_dictionnaires):
     for icat,cat in enumerate(liste_categories):
         if icat==0 or icat==1:
             classe=1
         else:
             classe=0
         for iph,ph in enumerate(liste_phonemes):
             for exemplaire in range(np.array(dic[cat][ph]).shape[0]):
               X.append(dic[cat][ph][exemplaire])
               Y.append(classe)
 return np.array(X), np.array(Y)

def getData_maps(liste_dictionnaires = [], liste_categories = [], liste_phonemes = [], a_ignorer=[]):
 """
    :param liste_dictionnaires: liste de dictionnaires de cartes
    :param liste_categories: liste de categories
    :param liste_phonemes: liste de phonemes
    :return: une matrice de parametre X ((nb_dic*nb_cat*nb_ph*nb_cartes)*nb_elts_carte), et un vecteur de parametre Y (nb_dic*nb_cat*nb_ph*nb_cartes)
 """
 X = []
 Y = []

 for idic,dic in enumerate(liste_dictionnaires):
     for icat,cat in enumerate(liste_categories):
         if icat==0 or icat==1:
             classe=1
         else:
             classe=0
         for iph,ph in enumerate(liste_phonemes):
             for exemplaire in range(np.array(dic[cat][ph]).shape[0]):
                 for map in range(np.array(dic[cat][ph]).shape[1]):
                    if not(map in a_ignorer):
                      X.append(np.array(dic[cat][ph][exemplaire][map]).flatten())
                      Y.append(classe)
 return np.array(X), np.array(Y)

def getData_onePerMap(liste_dictionnaires = [], liste_categories = [], liste_phonemes = [],a_ignorer=[]):
 """

    :param liste_dictionnaires: liste de dictionnaires de cartes
    :param liste_categories: liste de categories
    :param liste_phonemes: liste de phonemes
    :return: une matrice de parametre X ((nb_dic*nb_cat*nb_ph)*(nb_cartes*nb_elts_carte), et un vecteur de parametre Y (nb_dic*nb_cat*nb_ph)
 """
 X = []
 Y = []

 for idic,dic in enumerate(liste_dictionnaires):
     for icat,cat in enumerate(liste_categories):
         if icat==0 or icat==1:
             classe=1
         else:
             classe=0
         for iph,ph in enumerate(liste_phonemes):
             for exemplaire in range(np.array(dic[cat][ph]).shape[0]):
                 inter = []
                 for map in range(np.array(dic[cat][ph]).shape[1]):
                    if not(map in a_ignorer):
                      inter.append(np.array(dic[cat][ph][exemplaire][map]).flatten())
                 if not(inter==[]):
                   X.append(np.concatenate(np.array(inter)))
                   Y.append(classe)
 return np.array(X), np.array(Y)

def LDAmeanScore(X,Y,n_folds):
 """
    :param X: matrice d'entree du classifieur, n_samples*n_parameters
    :param Y: matrice des labels, n_samples
    :param n_folds: nombre de tests pour le KFold
    :return: le score moyen de la validation croisee, affiche ce score
 """
 if (X.shape[0]>n_folds):
    # Cross validation pour estimer la performance d'un classifieur LDA
    kf = KFold(n=len(Y), n_folds=n_folds, shuffle=False,
                               random_state=None)
    scores = []
    for train_index, test_index in kf:
      X_train, X_test = X[train_index,:],X[test_index,:]
      Y_train, Y_test = Y[train_index],Y[test_index]
      cl = LDA()
      #cl = SVC()
      #cl = GaussianNB()
      #cl = KNeighborsClassifier(3)
      cl.fit(X_train,Y_train)
      scores.append(cl.score(X_test,Y_test))


    print 'Score moyen : ',np.mean(np.array(scores))
    return 100.*np.mean(np.array(scores))
 else:
      return -1