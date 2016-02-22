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


def getData_goodmaps(liste_dictionnaires = [], liste_categories = [], liste_phonemes = [],liste_cartes=[]):
    """construct all good maps to a big vector

    :param liste_dictionnaires: liste de dictionnaires de cartes
    :param liste_categories: liste de categories
    :param liste_phonemes: liste de phonemes
    :param liste_cartes : cartes que l'on considere

    :return: une matrice de parametre Mat ((nb_dic*nb_cat*nb_ph*nb_cartes)*taille_vecteur), et des vecteurs de parametre Y (nb_dic*nb_cat*nb_ph*nb_cartes)
    differenciant les corrects des incorrects, les phonSemes, les langues """
    tableau = np.array(liste_dictionnaires[0][liste_categories[0]][liste_phonemes[0]])
    nb_exemple,nb_carte,lign,col=tableau.shape

    Mat = []
    Reference = []


    for inddict,dict in enumerate(liste_dictionnaires):
        for indcat,cat in enumerate(liste_categories):
            for indpho,pho in enumerate(liste_phonemes):
                for ex in range(nb_exemple):
                    goodmaps = []
                    for map in liste_cartes:
                        goodmaps.append(np.array(dict[cat][pho][ex][map]).flatten())
                    Mat.append(np.array(goodmaps).flatten())
                    Reference.append([inddict,indcat ,indpho])
    Reference = np.array(Reference)
    Y_c_inc = Reference[:,1]
    Y_r_v = Reference[:,2]
    Y_fr_jap = Reference[:,0]
    return np.array(Mat), np.array(Y_c_inc), np.array(Y_r_v), np.array(Y_fr_jap)
