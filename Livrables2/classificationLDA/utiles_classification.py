import pickle
import numpy as np
from sklearn.lda import LDA
from sklearn.cross_validation import KFold
from visualization.dim_reduction import dim_reduction_PCA

def getData_dense(liste_dictionnaires = [], liste_categories = [], liste_phonemes = [],indices_corrects=[0,1]):
 """
    :param liste_dictionnaires: liste de dictionnaires de vecteurs denses. Doit etre non vide, chaque dictionnaire est du format
            n_categories*n_phonemes*n_exemplaires*taille_vecteur
    :param liste_categories: liste de categories. Doit etre non vide, les cles doivent correspondre au dictionnaire
    :param liste_phonemes:  liste de phonemes. Doit etre non vide, les cles doivent correspondre au dictionnaire
    :param indices_corrects ; numeros des categories qui sont corrects (dans liste_categories).
    :return: une matrice de parametre X ((nb_dic*nb_cat*nb_ph*nb_cartes)*taille_vecteur), et des vecteurs de parametre Y (nb_dic*nb_cat*nb_ph*nb_cartes)
    differenciant les corrects des incorrects, les phonemes, les langues
 """
 X = []
 Y_c_inc = []
 Y_r_v = []
 Y_fr_jap = []
 for idic,dic in enumerate(liste_dictionnaires):
     for icat,cat in enumerate(liste_categories):
         if icat in indices_corrects:
             classe_c_inc=1
         else:
             classe_c_inc=0
         for iph,ph in enumerate(liste_phonemes):
             for exemplaire in range(np.array(dic[cat][ph]).shape[0]):
               X.append(dic[cat][ph][exemplaire])
               Y_c_inc.append(classe_c_inc)
               Y_r_v.append(iph)
               Y_fr_jap.append(idic)
 return np.array(X), np.array(Y_c_inc),np.array(Y_r_v), np.array(Y_fr_jap)

def getData_maps(liste_dictionnaires = [], liste_categories = [], liste_phonemes = [],a_ignorer=[],liste_cartes=[],indices_corrects=[0,1]):
 """
    :param liste_dictionnaires: liste de dictionnaires de cartes. Doit etre non vide, chaque dictionnaire est du format
            n_categories*n_phonemes*n_exemplaires*nb_fenetres*taille_vecteur
    :param liste_categories: liste de categories. Doit etre non vide, les cles doivent correspondre au dictionnaire
    :param liste_phonemes: liste de phonemes. Doit etre non vide, les cles doivent correspondre au dictionnaire
    :param a_ignorer: cartes que l'on veut ignorer
    :param liste_cartes : cartes que l'on considere.Si non vides, les indices doivent etre coherent avec le dictionnaire.
    :param indices_corrects ; numeros des categories qui sont corrects (dans liste_categories)
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
         if icat in indices_corrects:
             classe_c_inc=1
         else:
             classe_c_inc=0
         for iph,ph in enumerate(liste_phonemes):
             for exemplaire in range(np.array(dic[cat][ph]).shape[0]):
                 for map in liste_cartes:
                    if not(map in a_ignorer):
                      X.append(np.array(dic[cat][ph][exemplaire][map]).flatten())
                      Y_c_inc.append(classe_c_inc)
                      Y_r_v.append(iph)
                      Y_fr_jap.append(idic)
 return np.array(X), np.array(Y_c_inc), np.array(Y_r_v), np.array(Y_fr_jap)

def getData_onePerMap(liste_dictionnaires = [], liste_categories = [], liste_phonemes = [],a_ignorer=[],liste_cartes=[],indices_corrects=[0,1]):
 """

    :param liste_dictionnaires: liste de dictionnaires de cartes. Doit etre non vide, chaque dictionnaire est du format
            n_categories*n_phonemes*n_exemplaires*nb_fenetres*taille_vecteur
    :param liste_categories: liste de categories.  Doit etre non vide, les cles doivent correspondre au dictionnaire
    :param liste_phonemes: liste de phonemes.  Doit etre non vide, les cles doivent correspondre au dictionnaire
    :param a_ignorer: cartes que l'on veut ignorer
    :param liste_cartes: cartes que l'on considere. Si non vides, les indices doivent etre coherent avec le dictionnaire.
    :param indices_corrects ; numeros des categories qui sont corrects (dans liste_categories)
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
         if icat in indices_corrects:
             classe_c_inc=1
         else:
             classe_c_inc=0
         for iph,ph in enumerate(liste_phonemes):
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

def LDAmeanScore(X,Y,n_folds,dim_reduction=0):
 """
    :param X: matrice d'entree du classifieur, n_samples*n_parameters, n_paramters>=2, n_samples>0. DONNES COHERENTES POUR CLASSIFICATION LDA
    :param Y: matrice des labels, n_samples
    :param n_folds: nombre de tests pour le KFold, >1
    :param dim_reduction: si inferieur ou egale a 0, pas de reduction, sinon, si le nombre de parametre est superieur a dim_reduction, on fait une reduction PCA
    :return: le score moyen de la validation croisee, affiche ce score. Si n_folds>n_samples, renvoie -1
 """
 if dim_reduction>0 and X.shape[1]>dim_reduction:
     X = dim_reduction_PCA(X,dim_reduction)

 if (X.shape[0]>n_folds):
    # Cross validation pour estimer la performance d'un classifieur LDA
    kf = KFold(n=len(Y), n_folds=n_folds, shuffle=False,
                               random_state=None)
    scores = []
    for train_index, test_index in kf:
      X_train, X_test = X[train_index,:],X[test_index,:]
      Y_train, Y_test = Y[train_index],Y[test_index]
      cl = LDA()
      cl.fit(X_train,Y_train)
      scores.append(cl.score(X_test,Y_test))


    print 'Score moyen : ',np.mean(np.array(scores))
    return 100.*np.mean(np.array(scores))
 else:
      return -1


def ldaClassification(liste_dictionnaires,liste_phonemes,liste_categories,num_cartes=[],a_ignorer=[],n_folds=5,type='c_inc',dim_reduction=0,indices_corrects=[0,1]):
 """
 :param liste_dictionnaires: liste de dictionnaires.Doit etre non vide, chaque dictionnaire est du format
            n_categories*n_phonemes*n_exemplaires*nb_fenetres*taille_vecteur. DONNES COHERENTES POUR CLASSIFICATION LDA
 :param liste_phonemes: liste de phonemes. Doit etre non vide, les cles doivent correspondre au dictionnaire
 :param liste_categories: liste des categories. Doit etre non vide, les cles doivent correspondre au dictionnaire
 :param num_cartes: numeros des cartes interessantes (si vide, on prend tout). Si non vides, les indices doivent etre coherent avec le dictionnaire.
 :param a_ignorer: numeros des cartes que l'on ignore
 :param n_folds: n_folds pour le kfold
 :param type: type de classification, c_inc pour correct/incorrect, fr_jap pour francais/japonais, r_v pour les phonemes r et v
 :return: deux scores moyens de classification, obtenus avec deux types de donnees ( getData_onePerMap, et getData_maps)
 """

 print 'getData_OnePerMap : '
 X,Y_c_inc,Y_r_v, Y_fr_jap = getData_onePerMap(liste_dictionnaires=liste_dictionnaires,liste_categories=liste_categories,liste_phonemes=liste_phonemes,liste_cartes=num_cartes,a_ignorer=a_ignorer,indices_corrects=indices_corrects)
 if type=='r_v':
     Y=Y_r_v
 elif type=='c_inc':
     Y=Y_c_inc
 elif type=='fr_jap':
     Y=Y_fr_jap
 else:
     Y=Y_c_inc
 score1 = LDAmeanScore(X,Y,n_folds,dim_reduction)
 # Verification des dimensions
 # Xsh = X.shape
 # convSh = np.array(liste_dictionnaires[0]['correct_OK']['v']).shape
 # print 'Verification des dimensions : '
 # print 'X : ',Xsh
 # print 'getData_onePerMap : ',len(liste_dictionnaires)*len(liste_phonemes)*len(liste_categories)*convSh[0]

 print 'getData_maps :'
 X,Y_c_inc,Y_r_v, Y_fr_jap = getData_maps(liste_dictionnaires=liste_dictionnaires,liste_categories=liste_categories,liste_phonemes=liste_phonemes,liste_cartes=num_cartes,a_ignorer=a_ignorer,indices_corrects=indices_corrects)
 if type=='r_v':
     Y=Y_r_v
 elif type=='c_inc':
     Y=Y_c_inc
 elif type=='fr_jap':
     Y=Y_fr_jap
 else:
     Y=Y_c_inc
 score2 = LDAmeanScore(X,Y,n_folds,dim_reduction)
 # Verification des dimensions
 # Xsh = X.shape
 # convSh = np.array(liste_dictionnaires[0]['correct_OK']['v']).shape
 # print 'Verification des dimensions : '
 # print 'X : ',Xsh
 # if len(num_cartes)==0:
 #     nb_cartes = convSh[1]
 # else:
 #     nb_cartes = len(num_cartes)
 # print 'getData_maps : ',len(liste_dictionnaires)*len(liste_phonemes)*len(liste_categories)*convSh[0]*nb_cartes

 return score1,score2

def getY(R,type,inds_correct=[0,1]):
    """
    :param R: matrice contenant dans la 1ere colonne l'indice de la langue, dans la 2eme l'indice de la categorie, dans la 3eme
              l'indice du phoneme. De taille (nb_languesc*nb_phonemes*nb_categories*nb_exemplaires)*3
    :param type: c_inc pour seprarer correct et incorrect, fr_jap pour separer les dictionnaires (langues), r_v pour separer les phonemes
    :param inds_correct: indices des categories qui sont corrects
    :return: Y de taille nb_languesc*nb_phonemes*nb_categories*nb_exemplaires, contenant le vecteur faisant la separation voulue
    """
    if type=='c_inc':
        Y = R[:,1]
        for i in range(len(Y)):
             if Y[i] in inds_correct:
                 Y[i]=1
             else:
                 Y[i]=0
    elif type == 'fr_jap':
        Y = R[:,0]
    elif type == 'r_v':
        Y = R[:,2]
    return Y


def getData_goodmaps(liste_dictionnaires = [], liste_categories = [], liste_phonemes = [],liste_cartes=[]):
    """construct all good maps to a big vector

    :param liste_dictionnaires: liste de dictionnaires de cartes
    :param liste_categories: liste de categories
    :param liste_phonemes: liste de phonemes
    :param liste_cartes : cartes que l'on considere

    :return: une matrice de parametre Mat ((nb_dic*nb_cat*nb_ph*nb_cartes)*taille_vecteur), et des vecteurs de parametre Y (nb_dic*nb_cat*nb_ph*nb_cartes)
    differenciant les corrects des incorrects, les phonSemes, les langues """
    if liste_dictionnaires!=[] and liste_categories!=[] and liste_phonemes!=[]:
        tableau = np.array(liste_dictionnaires[0][liste_categories[0]][liste_phonemes[0]])
        nb_exemple,nb_carte,lign,col=tableau.shape
    else:
        return [],[],[],[]

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
    Y_c_inc = change_reference(Reference[:,1])
    Y_r_v = Reference[:,2]
    Y_fr_jap = Reference[:,0]
    return np.array(Mat), np.array(Y_c_inc), np.array(Y_r_v), np.array(Y_fr_jap)

def change_reference(Y_c_inc):
    return [0 if x<2 else 1 for x in Y_c_inc]

