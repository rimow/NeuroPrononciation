import csv

import math
import numpy as np



def pretraitementMatrice (liste_dictionnaires = [], liste_categories = [], liste_phonemes = []):
    """
    cree le tenser contenant pour chaque carte d'activation la valeur de cette carte pour tous les exemple
    et la matrice reference rappelant pour chaque exemple de chaque dictionnaire le dictionnaire, la categorie et le phoneme correspondant
    :param liste_dictionnaires: les dictionnaires qu'on utilise pour le clustering
    :param liste_categories: les categories qu'on considere ('correct_OK', 'correct_pasOk', 'incorrect_OK', 'incorrect_pasOk')
    :param liste_phonemes: le phoneme ('r' ou 'v') sur lequel on va effectuer le clustering
    :return: le tenser Mat de taille : nb_cartes x (nb_dict x nb_cat x nb_phoxnb_ex) x taille_carte
             la matrice Reference de taille (nb_dict x nb_cat x nb_phoxnb_ex) x 3
    """

    tableau = np.array(liste_dictionnaires[0][liste_categories[0]][liste_phonemes[0]])
    taille=tableau.shape

    Mat = []
    Reference = []
    for num in range(taille[1]):
        Matinter = []
        for inddict,dict in enumerate(liste_dictionnaires):
           for indcat,cat in enumerate(liste_categories):
               for indpho,pho in enumerate(liste_phonemes):
                   for ex in range(taille[0]):
                       Matinter.append((dict[cat][pho][ex][num]).flatten())
                       if num == 0:
                           Reference.append([inddict,indcat ,indpho])
        Mat.append(Matinter)

    Reference = np.array(Reference)

    Mat = np.array(Mat)

    return Mat, Reference

def ratios ( Y_Cluster , Reference, nb_classes=2, fichier = None):
    """
    calcule le pourcentage de phonemes d'une categorie classes dans chaque cluster
    :param Y_Cluster: la matrice de labels obtenue en sortie de clustering
    :param Reference: la matrice contenant pour chaque element le dictionnaire, le type de phoneme...
    :param nb_classes: le nombre de classes en sortie de cluster
    :param fichier: chemin contenant le fichier dans lequel on enregistre le ratio si l'option est precisee
    :return: le pourcentage de phonemes de chaque categorie classe dans chaque cluster
    """

    Y_Cluster = np.array(Y_Cluster)

    # Cree les listes des indices correpondant a chacune des classes
    classes = []
    for cl in range(nb_classes):
        classes.append(np.array([j for (j , i) in enumerate(Y_Cluster) if i == cl]))
    total = []
    total_class = []
    #on calcule le total d'element dans chacune des categories de la matrice de reference
    for i in range(len(set(Reference))):
        nb = len([j for j in Reference if j == i])
        total.append(nb)

    #pour chaque classe on calcule le nombre d'element de chaque categorie range dans cette classe
    for m in range(nb_classes):
        nb_cluster = []
        for i in range(len(set(Reference))):
            #debugage: si la classe est vide on met le nombre a 0
            if len(classes[m]) == 0:
                nb = 0
            else:
                nb = len([j for j in Reference[classes[m]] if j == i])
            nb_cluster.append(nb)
        total_class.append(nb_cluster)
    #on calcule le pourcentage d'elements de chaque categorie dans chaque classe
    ratio = 100.0 * np.array(total_class)/np.array(total)

    #si on a precise un fichier dans lequel enregistrer;
    if fichier != None:
        f = open(fichier, "a")
        np.savetxt(f, np.atleast_2d(ratio[0]), delimiter =',')
        f.close()

    return ratio

def bienClusterise (fichierClustering = None, MatriceClustering = [],seuil = 30, listeVide = []):
    """

    :param fichierClustering: si on decide de recuperer les resultats du clustering a partir d'un fichier
    :param MatriceClustering: si on prefere donner une matrice. Le fichier csv est prioritaire
    :param seuil: distance entre les deux types clusterises
    :param listeVide: liste des numeros des cartes d'activation jugees vides et retirees avant le clustering
    :return: liste des numeros des cartes d'activation jugees suffisamment discriminantes
    """

    #si le fichier n'est pas precise, on recupere la matrice passee en parametres
    if fichierClustering == None:
        tableau = np.array(MatriceClustering)
    #sinon on recupere le fichier
    else:
        f = open(fichierClustering, "rb")
        tableau = csv.reader(f)

    #pour toutes les cartes d'activation clusterisees, on determine lesquelles sont interessantes-discriminent bien les donnees
    bon = []
    for indligne,ligne in enumerate(tableau):
        #si on a charge une matrice on commence a ala ligne 0
        if fichierClustering == None:
            ligne[0] = float(ligne[0])
            ligne[1] = float(ligne[1])
            if ((ligne[0] >= 50 and ligne[1] <= 50) or (ligne[1] >= 50 and ligne[0] <= 50)) and (abs(ligne[0]-ligne[1])>seuil):
                bon.append(indligne-1)
        #sinon on ommet le titre et on commence a la ligne 1
        else:
            if indligne >0:
                ligne[0] = float(ligne[0])
                ligne[1] = float(ligne[1])
                if ((ligne[0] >= 50 and ligne[1] <= 50) or (ligne[1] >= 50 and ligne[0] <= 50)) and (abs(ligne[0]-ligne[1])>seuil):
                    bon.append(indligne-1)

    #recalage des numeros des cartes en prenant en compte les cartes vides non clusterisees
    decalage = 0
    if (len(listeVide)>0):
        for indligne in range(len(bon)):
            #si l'indice courant depasse la valeur de la carte d'activation de la liste vide consideree on incremente le decalage
            if (decalage<len(listeVide)) and (bon[indligne+decalage]>=listeVide[decalage]):
                decalage += 1
            bon[indligne] = bon[indligne]+decalage


    return bon

