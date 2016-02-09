import numpy as np
from matplotlib import pyplot as plt
import math
import operator

from fonctions_utiles import getPhonemeDict


def pourcentageTableau(Y , n_clusters , labels , dict_path , type_separation, fichier):
    '''
    :param n_clusters: nombre de clusters: doit matcher avec type separation : 3 classes pour voise et consonnes, plus pour fricatives
    :param labels: tableau resultat du clustering
    :param dict: chemin du classement des phonemes selon leur caracteristiques (consonne = 0, voise = 1, fricative... = 2)
    :param type_separation: type de phonemes discrimines : voises. consonnes? fricatives/occlusives?
    :return:
    '''

    dict = getPhonemeDict(dict_path)
    Y_v_non_v = getY_v_non_v(Y , dict , type_separation)

    if type_separation == 0:
        printRatiosConsonnesTableau(n_clusters , labels , Y_v_non_v, fichier)
    elif type_separation == 1:
        printRatiosVoiseTableau(n_clusters , labels , Y_v_non_v,fichier)
    else:
        printRatiosCategoriesTableau(n_clusters , labels , Y_v_non_v, fichier)


def printRatiosConsonnesTableau(nb_classes , Y_cluster , y_cons_voy, fichier):
    '''

    :param nb_classes: nb_classes: le nombre de classes resultant du clustering
    :param Y_cluster: tableau contenat les classes attribuees a chaque fenetre de X
    :param y_categorie: vecteur contenant les classes : 0 pour non voise, 1 pour voise, 2 pour silence
    :param fichier: nom du fichier dans lequel on enregistre les resultats
    :return:
    '''
    # Au cas ou :
    Y_cluster = np.array(Y_cluster)

    # Cree les listes des indices correpondant a chacune des classes
    classes = []
    for cl in range(nb_classes):
        classes.append(np.array([j for (j , i) in enumerate(Y_cluster) if i == cl]))

    nb_silences = len([i for i in y_cons_voy if i == 2])
    nb_voyelles = len([i for i in y_cons_voy if i == 1])
    nb_n_consonnes = len([i for i in y_cons_voy if i == 0])
    print 'Voyelles et consonnes (en pourcentage de chacune des classes):'
    # Enregistrement des resultats sous forme de tableau
    valeurs = []
    for cl in range(nb_classes):
        p1 = 100. * len([i for i in y_cons_voy[classes[cl]] if i == 0]) / nb_n_consonnes  # % classe cl et consonne
        p2 = 100. * len([i for i in y_cons_voy[classes[cl]] if i == 1]) / nb_voyelles  # % classe cl et voyelle
        p3 = 100. * len([i for i in y_cons_voy[classes[cl]] if i == 2]) / nb_silences  # % classe cl et silence
        print '   Classe ' , cl , ':'
        print '     Consonnes :' , p1 , '\n     Voyelles :' , p2 , '\n     Silences :' , p3 , '\n'
        ligne = [int(cl), round(p1), round(p2),round(p3)]
        valeurs.append(ligne)


    f = open(fichier, "a")
    f.write("classes,consonnes,voyelles,silences\n")
    tableau = np.asarray(valeurs, dtype=np.float64)
    np.savetxt(f, tableau, delimiter =',')
    f.close()




def printRatiosVoiseTableau(nb_classes , Y_cluster , y_voise_non_voise, fichier):
    '''

    :param nb_classes: nb_classes: le nombre de classes resultant du clustering
    :param Y_cluster: tableau contenat les classes attribuees a chaque fenetre de X
    :param y_categorie: vecteur contenant les classes : 0 pour non voise, 1 pour voise, 2 pour silence
    :param fichier: nom du fichier dans lequel on enregistre les resultats
    :return:
    '''

    # Au cas ou :
    Y_cluster = np.array(Y_cluster)

    # Cree les listes des indices correpondant a chacune des classes
    classes = []
    for cl in range(nb_classes):
        classes.append(np.array([j for (j , i) in enumerate(Y_cluster) if i == cl]))

    nb_silences = len([i for i in y_voise_non_voise if i == 2])
    nb_voises = len([i for i in y_voise_non_voise if i == 1])
    nb_n_voises = len([i for i in y_voise_non_voise if i == 0])
    print 'Pourcentage des voises, non voises et silences :'
    valeurs = []
    for cl in range(nb_classes):
        p1 = 100. * len([i for i in y_voise_non_voise[classes[cl]] if i == 0]) / nb_n_voises  # % classe cl et non voise
        p2 = 100. * len([i for i in y_voise_non_voise[classes[cl]] if i == 1]) / nb_voises  # % classe cl et voise
        p3 = 100. * len([i for i in y_voise_non_voise[classes[cl]] if i == 2]) / nb_silences  # % classe cl et silence
        print '   Classe ' , cl , ':'
        print '     Non voises :' , p1 , '\n    Voises :' , p2 , '\n   Silences :' , p3 , '\n'
        valeurs.append([int(cl), round(p1), round(p2),round(p3)])

    f = open(fichier, "a")
    f.write("classes,non-voisees,voisees,silences\n")
    tableau = np.asarray(valeurs)
    np.savetxt(f, tableau, delimiter =',')
    f.close()


def printRatiosCategoriesTableau(nb_classes , Y_cluster , y_categorie, fichier):
    '''

    :param nb_classes: nb_classes: le nombre de classes resultant du clustering
    :param Y_cluster: tableau contenat les classes attribuees a chaque fenetre de X
    :param y_categorie: vecteur contenant les classes : 0 pour non voise, 1 pour voise, 2 pour silence
    :param fichier: nom du fichier dans lequel on enregistre les resultats
    :return:
    '''
    # Au cas ou :
    Y_cluster = np.array(Y_cluster)

    # Cree les listes des indices correpondant a chacune des classes
    classes = []
    for cl in range(nb_classes):
        classes.append(np.array([j for (j , i) in enumerate(Y_cluster) if i == cl]))

    print 'La classe i contient x pour cent du total de la categorie nasales par exemple:'
    nb_occlusives = len([i for i in y_categorie if i == 0])
    nb_fricative = len([i for i in y_categorie if i == 1])
    nb_nasale = len([i for i in y_categorie if i == 2])
    nb_voyelle = len([i for i in y_categorie if i == 3])
    nb_semi_consone = len([i for i in y_categorie if i == 4])
    nb_silence = len([i for i in y_categorie if i == 5])
    valeurs = []
    for cl in range(nb_classes):
        p1 = 100. * len([i for i in y_categorie[classes[cl]] if i == 0]) / nb_occlusives  # % classe cl et non voise
        p2 = 100. * len([i for i in y_categorie[classes[cl]] if i == 1]) / nb_fricative  # % classe cl et voise
        p3 = 100. * len([i for i in y_categorie[classes[cl]] if i == 2]) / nb_nasale  # % classe cl et silence
        p4 = 100. * len([i for i in y_categorie[classes[cl]] if i == 3]) / nb_voyelle  # % classe cl et silence
        p5 = 100. * len([i for i in y_categorie[classes[cl]] if i == 4]) / nb_semi_consone  # % classe cl et silence
        p6 = 100. * len([i for i in y_categorie[classes[cl]] if i == 5]) / nb_silence  # % classe cl et silence
        print '   Classe ' , cl , ':'
        print '     Occlusives :' , p1 , '\n    Fricatives :' , p2 , '\n   Nasale :' , p3 , '\n  Voyelle :' , p4 , '\n  Semi_consonne :' , p5 , '\n  Silences :' , p6 , '\n'
        valeurs.append([int(cl), round(p1), round(p2),round(p3), round(p4), round(p5),round(p6)])

    f = open(fichier, "a")
    f.write("classes,occlusives,fricatives,nasales,voyelles,semi-consonnes,silences\n")
    tableau = np.asarray(valeurs)
    np.savetxt(f, tableau, delimiter =',')
    f.close()


def getY_v_non_v(Y , dict , type_separation):
    """
    :param Y: tableau contenant les phonemes correspondant a chaque ligne de X
    :param dict: le dictionnaire contenant les informations sur les phonemes
    :return: un vecteur contenant, respectivement a chaque phoneme de Y, 0 si non voise, 1 si voise, 2 si silence
  """
    y_voise_non_voise = []  # contient pour chaque phoneme de Y sa classe en tant que voise ou non voise ou rien
    for ph in Y:
        y_voise_non_voise.append(dict[ph][type_separation])#indexation des separations de 1 a 3  mais des colonnes de 0 a 2
    y_voise_non_voise = np.array(y_voise_non_voise)
    return y_voise_non_voise


def CoeffsHistogrammes(X , bins , y_voise_non_voise , ind_min , ind_max):
    """
    :param X: matrice contenant les feature vectors (n_vectors x n_param)
    :param bins: nombre de bandes dans les histogrammes
    :param y_voise_non_voise: vecteur contenant les classes : 0 pour non voise, 1 pour voise, 2 pour silence
    :param ind_min: indice correspondant au premier parametre dont on veut l'histogramme des coefficients
    :param ind_max: indice correspondant au dernier parametre dont on veut l'histogramme des coefficients
    :return: affiche, pour chaque parametre entre min et max, l'histogramme des coefficients pour les phonemes voises et non voises
             et les silences
      """
    X_shape = X.shape
    if ind_max >= ind_min and ind_max > 0 and ind_min >= 0 and ind_max < X_shape[1]:
        indices_non_voise = [i for (i , j) in enumerate(y_voise_non_voise) if j == 0]
        indices_voise = [i for (i , j) in enumerate(y_voise_non_voise) if j == 1]
        indices_silence = [i for (i , j) in enumerate(y_voise_non_voise) if j == 2]
        for i in range(ind_min , ind_max + 1):
            ax = plt.figure()
            bins1 = plt.hist(X[indices_non_voise , i] , bins=bins , alpha=0.5 , label='non voises')
            plt.legend()
            bins2 = plt.hist(X[indices_voise , i] , bins=bins , alpha=0.5 , label='voises')
            plt.legend()
            bins3 = plt.hist(X[indices_silence , i] , bins=bins , alpha=0.5 , label='silences')
            plt.legend()
            plt.title('Parametre (indice) : ' + str(i))
            plt.show()
    else:
        print 'Erreur : ind_max doit etre superieur ou egal a ind_min, et ind_max et ind_min doivent etre des indices corrects.'

def histogrammesPhonemes(n_clusters , labels , pho):
    """
    :param n_clusters: le nombre de clusters
    :param labels: le tableau representant l'attribution des classes des feature vectors (tableau obtenu par un algo de clustering)
    :param pho: tableau contenant les phonemes correspondant a chaque feature vector
    :return: affiche, pour chaque classe, l'histogramme des phonemes (pourcentage par rapport au nombre total de ce phoneme)
  """
    nbs_pho = {}
    for ph in set(pho):
        nbs_pho[ph] = 0

    for ph in pho:
        nbs_pho[ph] = nbs_pho[ph] + 1

    cluster_pho = [None] * n_clusters
    for label , ph in zip(labels , pho):
        if (cluster_pho[label] == None):
            cluster_pho[label] = {ph: 1}
        elif ph in cluster_pho[label].keys():
            cluster_pho[label][ph] += 1
        else:
            cluster_pho[label][ph] = 1

    for label in range(n_clusters):
      for ph in cluster_pho[label].keys():
        cluster_pho[label][ph] = 100.*cluster_pho[label][ph]/nbs_pho[ph]

    cluster_pho[:] = [sorted(x.items() , key=operator.itemgetter(1) , reverse=True) for x in cluster_pho]

    # use bar chart to visualize each class
    nrows = int(round(math.sqrt(n_clusters)))
    ncols = int(math.ceil(n_clusters / round(math.sqrt(n_clusters))))
    figures , axs = plt.subplots(nrows=nrows , ncols=ncols)

    for ax , data in zip(axs.ravel() , cluster_pho):
        data = zip(*data)
        ax.bar(range(len(data[0])) , data[1] , width=0.2)
        ax.set_xticks(np.arange(len(data[0])) + 0.1)
        ax.set_xticklabels(data[0] , rotation=0)
    plt.show()