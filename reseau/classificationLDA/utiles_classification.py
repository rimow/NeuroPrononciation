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
    :return: une matrice de parametre X ((nb_dic*nb_cat*nb_ph*nb_cartes)*taille_vecteur), et des vecteurs de parametre Y (nb_dic*nb_cat*nb_ph*nb_cartes)
    differenciant les corrects des incorrects, les phonemes, les langues
 """
 X = []
 Y_c_inc = []
 Y_r_v = []
 Y_fr_jap = []
 for idic,dic in enumerate(liste_dictionnaires):
     for icat,cat in enumerate(liste_categories):
         if icat==0 or icat==1:
             classe_c_inc=1
         else:
             classe_c_inc=0
         for iph,ph in enumerate(liste_phonemes):
             # if ph=='R':
             #     classe_r_v = 1
             # else:
             #     classe_r_v = 0
             for exemplaire in range(np.array(dic[cat][ph]).shape[0]):
               X.append(dic[cat][ph][exemplaire])
               Y_c_inc.append(classe_c_inc)
               Y_r_v.append(iph)
               Y_fr_jap.append(idic)
 return np.array(X), np.array(Y_c_inc),np.array(Y_r_v), np.array(Y_fr_jap)

def getData_maps(liste_dictionnaires = [], liste_categories = [], liste_phonemes = [], a_ignorer=[],liste_cartes=[]):
 """
    :param liste_dictionnaires: liste de dictionnaires de cartes
    :param liste_categories: liste de categories
    :param liste_phonemes: liste de phonemes
    :param a_ignorer: cartes que l'on veut ignorer
    :param liste_cartes : cartes que l'on considere
    :return: une matrice de parametre X ((nb_dic*nb_cat*nb_ph*nb_cartes)*taille_vecteur), et des vecteurs de parametre Y (nb_dic*nb_cat*nb_ph*nb_cartes)
    differenciant les corrects des incorrects, les phonemes, les langues """
 X = []
 Y_c_inc = []
 Y_r_v = []
 Y_fr_jap = []
 if liste_cartes==[]:
     liste_cartes=range(np.array(liste_dictionnaires[0][liste_categories[0]][liste_phonemes[0]]).shape[1])
 for idic,dic in enumerate(liste_dictionnaires):
     for icat,cat in enumerate(liste_categories):
         if icat==0 or icat==1:
             classe_c_inc=1
         else:
             classe_c_inc=0
         for iph,ph in enumerate(liste_phonemes):
             # if ph=='R':
             #     classe_r_v = 1
             # else:
             #     classe_r_v = 0
             for exemplaire in range(np.array(dic[cat][ph]).shape[0]):
                 for map in liste_cartes:
                    if not(map in a_ignorer):
                      X.append(np.array(dic[cat][ph][exemplaire][map]).flatten())
                      Y_c_inc.append(classe_c_inc)
                      Y_r_v.append(iph)
                      Y_fr_jap.append(idic)
 return np.array(X), np.array(Y_c_inc), np.array(Y_r_v), np.array(Y_fr_jap)

def getData_onePerMap(liste_dictionnaires = [], liste_categories = [], liste_phonemes = [],a_ignorer=[],liste_cartes=[]):
 """

    :param liste_dictionnaires: liste de dictionnaires de cartes
    :param liste_categories: liste de categories
    :param liste_phonemes: liste de phonemes
    :param a_ignorer: cartes que l'on veut ignorer
    :param liste_cartes: cartes que l'on considere
    :return: une matrice de parametre X ((nb_dic*nb_cat*nb_ph*nb_cartes)*taille_vecteur), et des vecteurs de parametre Y (nb_dic*nb_cat*nb_ph*nb_cartes)
    differenciant les corrects des incorrects, les phonemes, les langues """
 X = []
 Y_c_inc = []
 Y_r_v = []
 Y_fr_jap = []
 if liste_cartes==[]:
     liste_cartes=range(np.array(liste_dictionnaires[0][liste_categories[0]][liste_phonemes[0]]).shape[1])
 for idic,dic in enumerate(liste_dictionnaires):
     for icat,cat in enumerate(liste_categories):
         if icat==0 or icat==1:
             classe_c_inc=1
         else:
             classe_c_inc=0
         for iph,ph in enumerate(liste_phonemes):
             # if ph=='R':
             #     classe_r_v = 1
             # else:
             #     classe_r_v = 0
             for exemplaire in range(np.array(dic[cat][ph]).shape[0]):
                 inter = []
                 for map in liste_cartes:
                    if not(map in a_ignorer):
                      inter.append(np.array(dic[cat][ph][exemplaire][map]).flatten())
                 if not(inter==[]):
                   X.append(np.concatenate(np.array(inter)))
                   Y_c_inc.append(classe_c_inc)
                   Y_r_v.append(iph)
                   Y_fr_jap.append(idic)

 return np.array(X), np.array(Y_c_inc), np.array(Y_r_v), np.array(Y_fr_jap)

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


def ldaClassification(liste_dictionnaires,liste_phonemes,liste_categories,num_cartes=[],a_ignorer=[],n_folds=5,type='c_inc'):
 """
 :param liste_dictionnaires: liste de dictionnaires
 :param liste_phonemes: liste de phonemes
 :param liste_categories: liste des categories
 :param num_cartes: numeros des cartes interessantes (si vide, on prend tout)
 :param a_ignorer: numeros des cartes que l'on ignore
 :param n_folds: n_folds pour le kfold
 :param type: type de classification, c_inc pour correct/incorrect, fr_jap pour francais/japonais, r_v pour les phonemes r et v
 :return: deux scores moyens de classification, obtenus avec deux types de donnees ( getData_onePerMap, et getData_maps)
 """

 print 'getData_OnePerMap : '
 X,Y_c_inc,Y_r_v, Y_fr_jap = getData_onePerMap(liste_dictionnaires=liste_dictionnaires,liste_categories=liste_categories,liste_phonemes=liste_phonemes,liste_cartes=num_cartes,a_ignorer=a_ignorer)
 if type=='r_v':
     Y=Y_r_v
 elif type=='c_inc':
     Y=Y_c_inc
 elif type=='fr_jap':
     Y=Y_fr_jap
 else:
     Y=Y_c_inc
 score1 = LDAmeanScore(X,Y,n_folds)
 # Verification des dimensions
 Xsh = X.shape
 convSh = np.array(liste_dictionnaires[0]['correct_OK']['v']).shape
 print 'Verification des dimensions : '
 print 'X : ',Xsh
 print 'getData_onePerMap : ',len(liste_dictionnaires)*len(liste_phonemes)*len(liste_categories)*convSh[0]

 print '\ngetData_maps :'
 X,Y_c_inc,Y_r_v, Y_fr_jap = getData_maps(liste_dictionnaires=liste_dictionnaires,liste_categories=liste_categories,liste_phonemes=liste_phonemes,liste_cartes=num_cartes,a_ignorer=a_ignorer)
 if type=='r_v':
     Y=Y_r_v
 elif type=='c_inc':
     Y=Y_c_inc
 elif type=='fr_jap':
     Y=Y_fr_jap
 else:
     Y=Y_c_inc
 score2 = LDAmeanScore(X,Y,n_folds)
 # Verification des dimensions
 Xsh = X.shape
 convSh = np.array(liste_dictionnaires[0]['correct_OK']['v']).shape
 print 'Verification des dimensions : '
 print 'X : ',Xsh
 if len(num_cartes)==0:
     nb_cartes = convSh[1]
 else:
     nb_cartes = len(num_cartes)
 print 'getData_maps : ',len(liste_dictionnaires)*len(liste_phonemes)*len(liste_categories)*convSh[0]*nb_cartes
 print '\n'

 return score1,score2