import numpy as np
from matplotlib import pyplot as plt
import math
import operator

from utiles import getPhonemeDict


def pourcentage(Y , n_clusters , labels , dict_path , type_separation, fichier = None):
    """
    :param Y: veceteur contenant les phonemes correpondant a chaque vecteur (taille n_vectors)
    :param n_clusters: nombre de clusters: doit matcher avec type separation : 3 classes pour voise et consonnes, plus pour fricatives
    :param labels: tableau resultat du clustering
    :param dict_path: chemin du classement des phonemes selon leur caracteristiques (consonne = 0, voise = 1, fricative... = 2)
    :param type_separation: type de phonemes discrimines : voises. consonnes? fricatives/occlusives? (voir data/classement et data/detail_classement)
    :param fichier: si on souhaite enregistrer les resultats dans un fichier csv, on precise le chemin sinon on ne met rien
    :return:
    """

    dict = getPhonemeDict(dict_path)
    Y_v_non_v = getY_v_non_v(Y , dict , type_separation)

    if type_separation == 0:
        printRatiosConsonnes(n_clusters , labels , Y_v_non_v, fichier)
    elif type_separation == 1:
        printRatiosVoise(n_clusters , labels , Y_v_non_v, fichier)
    else:
        printRatiosCategories(n_clusters , labels , Y_v_non_v, fichier)


def printRatiosConsonnes(nb_classes , Y_cluster , y_cons_voy, fichier = None):
    """
    :param nb_classes: le nombre de classes resultant du clustering
    :param Y_cluster: tableau contenat les classes attribuees a chaque fenetre de X (taille n_vectors)
    :param y_voise_non_voise: vecteur contenant les classes : 0 pour non voise, 1 pour voise, 2 pour silence (taile n_vectors)
    :param type_separation: type de phonemes discrimines : voises. consonnes? fricatives/occlusives? (voir data/classement et data/detail_classement)
    :param fichier: si on souhaite enregistrer les resultats dans un fichier csv, on precise le chemin sinon on ne met rien
    :return: Pour chacune des classes, ecris le pourcentage de voyelles, consonnes et silences, et le pourcentage de
            voises, non voises et silences (pourcentage du total des phoneme)
  """
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
    valeurs = []
    for cl in range(nb_classes):
        p1 = 100. * len([i for i in y_cons_voy[classes[cl]] if i == 0]) / nb_n_consonnes  # % classe cl et consonne
        p2 = 100. * len([i for i in y_cons_voy[classes[cl]] if i == 1]) / nb_voyelles  # % classe cl et voyelle
        p3 = 100. * len([i for i in y_cons_voy[classes[cl]] if i == 2]) / nb_silences  # % classe cl et silence
        print '   Classe ' , cl , ':'
        print '     Consonnes :' , p1 , '\n     Voyelles :' , p2 , '\n     Silences :' , p3 , '\n'
        ligne = [int(cl), round(p1), round(p2),round(p3)]
        valeurs.append(ligne)

    if fichier != None:
        f = open(fichier, "a")
        f.write("classes,consonnes,voyelles,silences\n")
        tableau = np.asarray(valeurs, dtype=np.float64)
        np.savetxt(f, tableau, delimiter =',')
        f.close()

def printRatiosVoise(nb_classes , Y_cluster , y_voise_non_voise, fichier = None):
    """
    :param nb_classes: le nombre de classes resultant du clustering
    :param Y_cluster: tableau (taille n_vectors) contenant les classes attribuees a chaque fenetre de X (taille n_vectors x n_parametres)
    :param y_voise_non_voise: tableau (taille n_vectors) contenant les classes : 0 pour non voise, 1 pour voise, 2 pour silence
    :param type_separation: type de phonemes discrimines : voises. consonnes? fricatives/occlusives? (voir data/classement et data/detail_classement)
    :param fichier: si on souhaite enregistrer les resultats dans un fichier csv, on precise le chemin sinon on ne met rien
    :return: Pour chacune des classes, ecris le pourcentage de voyelles, consonnes et silences, et le pourcentage de
            voises, non voises et silences (pourcentage du total des phoneme)
  """
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
        ligne = [int(cl), round(p1), round(p2),round(p3)]
        valeurs.append(ligne)

    if fichier != None:
        f = open(fichier, "a")
        f.write("classes,non-voisees,voisees,silences\n")
        tableau = np.asarray(valeurs, dtype=np.float64)
        np.savetxt(f, tableau, delimiter =',')
        f.close()


def printRatiosCategories(nb_classes , Y_cluster , y_categorie, fichier = None):
    """
    :param nb_classes: le nombre de classes resultant du clustering
    :param Y_cluster: tableau (taille n_vectors) contenant les classes attribuees a chaque fenetre de X (taille n_vectors x n_parametres)
    :param y_categorie: tableau (taille n_vectors) contenant les classes : 0 occlusive, 1 fricative ...
    :param type_separation: type de phonemes discrimines : voises. consonnes? fricatives/occlusives? (voir data/classement et data/detail_classement)
    :param fichier: si on souhaite enregistrer les resultats dans un fichier csv, on precise le chemin sinon on ne met rien
    :return: Pour chacune des classes, ecris le pourcentage de fricatives, occlusives ... dans chaque classe
  """
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

    if fichier !=None:
        f = open(fichier, "a")
        f.write("classes,occlusives,fricatives,nasales,voyelles,semi-consonnes,silences\n")
        tableau = np.asarray(valeurs)
        np.savetxt(f, tableau, delimiter =',')
        f.close()


def getY_v_non_v(Y , dict , type_separation):
    """
    :param Y: tableau (taille n_vectors) contenant les phonemes correspondant a chaque ligne de X 
    :param dict: le dictionnaire contenant les informations sur les phonemes (voir data/classement)
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
    :param y_voise_non_voise: vecteur (taille n_vectors) contenant les classes : 0 pour non voise, 1 pour voise, 2 pour silence
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
    :param labels: le tableau de taille n_vectors representant l'attribution des classes des feature vectors (tableau obtenu par un algo de clustering)
    :param pho: (<=>Y) tableau de taille n_vectors contenant les phonemes correspondant a chaque feature vector
    :return: affiche, pour chaque classe, l'histogramme des phonemes (pourcentage par rapport au nombre total de ce phoneme)
  """
    nbs_pho = {}
    for ph in set(pho):
        nbs_pho[ph] = 0

    for ph in pho:
        nbs_pho[ph] +=  1

    cluster_pho = [{i:0 for i in set(pho)}] * n_clusters

    for label,ph in zip(labels , pho):
        cluster_pho[label][ph] += 1

    for label in range(n_clusters):
        for ph in set(pho):
            cluster_pho[label][ph] = (100.0*cluster_pho[label][ph])/nbs_pho[ph]

    #cluster_pho[:] = [sorted(x.items() , key=operator.itemgetter(1) , reverse=True) for x in cluster_pho]
    # use bar chart to visualize each class
    nrows = int(round(math.sqrt(n_clusters)))
    ncols = int(math.ceil(n_clusters / round(math.sqrt(n_clusters))))
    figures , axs = plt.subplots(nrows=nrows , ncols=ncols)
    for ax , data in zip(axs.ravel() , cluster_pho):
        #data = zip(*data)
        ax.bar(range(len(data.keys())) , data.values() , width=0.2)
        ax.set_xticks(np.arange(len(data.keys())) + 0.1)
        ax.set_xticklabels(data.keys() , rotation=0)
    plt.show()

def getMeanVectors(X,classes):
    """
    :param X: matrice contenant les feature vectors (n_vectors x n_param)
    :param classes: tableau de taille n_vectors contenant les classes correspondant a chaque ligne de X (phonemes, voises/non voises/silences ...)
    :return: un dictionnaire de la forme {classe1: vecteur_moyen1, classe2 : vecteur_moyen2, ...}, par ex classe1 vaut 'sil' ou 1 ou 2
     On peut ensuite utiliser ce dictionnaire pour connaitre le vecteur moyen d'une certaine classe. mean_classe1 = return_dict[classe1]
    """
    y_set = list(set(classes))
    print y_set
    means = {}
    for i in range(len(y_set)):
        indices = [jj for (jj,j) in enumerate(classes) if j==y_set[i]]
        means[y_set[i]]= np.mean(X[indices,:],axis=0)
    return means
