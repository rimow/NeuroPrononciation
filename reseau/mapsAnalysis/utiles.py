import csv

import matplotlib.pyplot as plt
import numpy as np
from process_activation_maps import load_maps

def pretraitementMatrice (liste_dictionnaires = [], liste_categories = [], liste_phonemes = []):
    """
    cree le tenser contenant pour chaque carte d'activation la valeur de cette carte pour tous les exemple
    et la matrice reference rappelant pour chaque exemple de chaque dictionnaire le dictionnaire, la categorie et le phoneme correspondant
    :param liste_dictionnaires: les dictionnaires qu'on utilise pour le clustering
    :param liste_categories: les categories qu'on considere ('correct_OK', 'correct_pasOk', 'incorrect_OK', 'incorrect_pasOk')
    :param liste_phonemes: le phoneme ('r' ou 'v') sur lequel on va effectuer le clustering
    :return: le tenser Mat de taille : nb_maps x (nb_dict x nb_cat x nb_pho x nb_ex) x map_shape
             la matrice Reference de taille (nb_dict x nb_cat x nb_pho x nb_ex) x 3
    """

    if liste_dictionnaires == [] and liste_categories==[] and liste_phonemes==[]:
        Mat=[]
        Reference = []
        return Mat,Reference

    if liste_dictionnaires == [] or liste_categories==[] or liste_phonemes==[]:
        print("Inputs are either all equal to [] or all specified")
        return [],[]

    tableau = np.array(liste_dictionnaires[0][liste_categories[0]][liste_phonemes[0]])
    taille=tableau.shape

    Mat = []
    Reference = []
    for num in range(taille[1]):
        Matinter = []
        for inddict,dict in enumerate(liste_dictionnaires):
           for indcat,cat in enumerate(liste_categories):
               for indpho,pho in enumerate(liste_phonemes):
                   for ex in range(np.array(dict[cat][pho]).shape[0]):
                       Matinter.append((dict[cat][pho][ex][num]).flatten())
                       if num == 0:
                           Reference.append([inddict,indcat ,indpho])
        Mat.append(Matinter)

    Reference = np.array(Reference)

    Mat = np.array(Mat)

    return Mat, Reference



def initialisation_centres (type_clustering, matrice_pretraitement, reference ):
    '''
    initialisation des centres en vue de faire un kmeans initialise
    :param type_clustering: obligatoirement FRJAP_R ou FRJAP_v ou R_v ou CIC_R ou CIC_v
    :param matrice_pretraitement: matrice Mat resultat de pretraitement matrice
    :param reference: matrice Reference resultat de pretraitement matrice
    :return: une matrice de deux lignes qui representent les centres avec lesquels on peut initialiser kmeans selon le type_clustering voulu
    '''


    preMatShape = matrice_pretraitement.shape
    #matrice_initiaux_boolean =  np.ones(preMatShape[0], dtype=bool)
    #print(matrice_initiaux_boolean)
    found1 = False
    found2 = False
    i1=-1
    i2=-1
    boo = np.ones(preMatShape[1], dtype=bool)
    boo = [False]*boo
    if type_clustering=='FRJAP_R' or type_clustering=='FRJAP_v':
        while ((not found1) and (not found2)) or (i1==i2):
            if (not found1):
                i1=i1+1
                if reference[i1,0]==0 and reference[i1,2]==0 :
                    found1 = True
            if ((not found2)):
                i2=i2+1
                if reference[i2,0]==1 and reference[i2,2]==0 :
                    found2 = True

    if type_clustering=='R_v':
        while ((not found1) and (not found2)) or (i1==i2):
            if (not found1):
                i1=i1+1
                if reference[i1,0]==0 and reference[i1,2]==0 :
                    found1 = True
            if ((not found2) ):
                i2=i2+1
                if reference[i2,0]==0 and reference[i2,2]==1 :
                    found2 = True

    if type_clustering=='CIC_R' or type_clustering=='CIC_v':
        while ((not found1) and (not found2)) or (i1==i2):
            if (not found1):
                i1=i1+1
                if reference[i1,0]==0 and reference[i1,1]==0  :
                    found1 = True
            if ((not found2)):
                i2=i2+1
                if reference[i2,0]==0 and reference[i2,1]==1 :
                    found2 = True
    if(i1>reference.shape[0] or i2>reference.shape[0]):
        print("Il n y a pas de bon centre, il y a un probleme! verifiez que la matrice issue du pretraitement est bonne")
    boo[i1]= True
    boo[i2]= True
    resultat_int = matrice_pretraitement[0,:,:]
    resultat = resultat_int[boo,:]
    return resultat



def ratios ( Y_Cluster , Reference, nb_classes=2, fichier = None):
    """
    calcule le pourcentage de phonemes d'une categorie classes dans chaque cluster
    :param Y_Cluster: la matrice de labels obtenue en sortie de clustering
    :param Reference: la matrice contenant pour chaque element le dictionnaire, le type de phoneme...
    :param nb_classes: le nombre de classes en sortie de cluster
    :param fichier: chemin contenant le fichier dans lequel on enregistre le ratio si l'option est precisee
    :return: le pourcentage de phonemes de chaque categorie classe dans chaque cluster
    """
    if Y_Cluster ==[] or Reference ==[]:
        print("Input data is empty list")
        return []
    if len(Y_Cluster) != len(Reference):
        print("Y_Cluster and Reference don't have the same size")
        return []
    Y_Cluster = np.array(Y_Cluster)
    Reference = np.array(Reference)
    nb_classes = len(set(Y_Cluster))
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

def bienClusterise (fichierClustering = None, MatriceClustering = [],seuil = 30,  indices=[]):
    """
    :param fichierClustering: si on decide de recuperer les resultats du clustering a partir d'un fichier
    :param MatriceClustering: si on prefere donner une matrice. Le fichier csv est prioritaire
    :param seuil: distance entre les deux types clusterises
    :return: liste des numeros des cartes d'activation jugees suffisamment discriminantes
    """

    #si le fichier n'est pas precise, on recupere la matrice passee en parametres
    if fichierClustering == None:
        tableau = np.array(MatriceClustering)
    #sinon on recupere le fichier
    else:
        f = open(fichierClustering, "rb")
        tableau = csv.reader(f)
        tableau = list(tableau)
        tableau = np.array(tableau)
    #pour toutes les cartes d'activation clusterisees, on determine lesquelles sont interessantes-discriminent bien les donnees
    bon = []
    for indligne,ligne in enumerate(tableau):
        #si on a charge une matrice on commence a ala ligne 0
        if fichierClustering == None:

            ligne0 = float(ligne[0])
            ligne1= float(ligne[1])
            if ((ligne0 >= 50 and ligne1 <= 50) or (ligne1 >= 50 and ligne0 <= 50)) and (abs(ligne0-ligne1)>=seuil):
                bon.append(indices[indligne])
        #sinon on ommet le titre et on commence a la ligne 1
        else:
            if indligne >0:
                ligne0 = float(ligne[0])
                ligne1 = float(ligne[1])
                if ((ligne0 >= 50 and ligne1 <= 50) or (ligne1 >= 50 and ligne0 <= 50)) and (abs(ligne0-ligne1)>=seuil):
                    bon.append(indices[indligne-1])

    return bon



def goodmaps(couche = "conv1",method="kmeansNonInit"):
    """Return the good maps of different cluster tasks.
       0: good maps for clustring R in FR and R in FRJA
       1: good maps for clustring V in FR and V inFRJA
       2:good maps for clustring R in FR and V in FR
       3: good maps for clustring correct R and  incorrect R in FRJA
       4: good maps for clustring correct V and incorrect V in FRJA
       :param couche: "conv1" or "conv2"
       :param method: method of cluster
       :returns goodmaps: the good maps of different cluster tasks
    """

    #vide_KMNI, pFRJA_R_KMNI, pFRJA_V_KMNI, pFR_RV_KMNI, pCIC_R_KMNI, pCIC_V_KMNI, ind = MapsClustering("conv1", 559, "kmeansNonInit", False)
    goodmaps = {}
    path = "../resultats/"+couche+"/"+method+"/"
    data = np.load(path+"bienClusterise"+method+".npy")
    for i in range(len(data)):
        goodmaps[i] = data[i]
    return goodmaps


def imagesCartesInteressantes(indice_carte_interessante, indice_carte_non_int, couche='conv1', clustering='1'):
    '''
    :param indice_carte_interessante: Indice de la carte interessante (cf fichier bonClustering)
    :param indice_carte_non_int: Indice de la deuxieme carte a laquelle on veut la comparer
    :param couche: ='conv1','conv2','dense1','mp2'
    :param clustering: 1, 1bis, 2, 3, 3bis suivant le type de clustering effectue
    :return: Image de la carte (interessante et non interessante) stockee dans resultats/couche
    '''

    #Load the maps corresponding to the selected layer
    map_file_FR = '../maps/BREF80_l_' + couche + '_35maps_th0.500000.pkl'
    map_file_JA = '../maps/PHONIM_l_' + couche + '_35maps_th0.001000.pkl'
    FR= load_maps(map_file_FR)
    JA = load_maps(map_file_JA)


    #Load and plot giving the chosen clustering
    if clustering == '1':
        phone = 'R'
        cat = 'correct_pasOK'
        ex = 8

        map1 = FR[cat][phone][ex][indice_carte_interessante]
        othermap1 = FR[cat][phone][ex][indice_carte_non_int]

        map2 = JA[cat][phone][ex][indice_carte_interessante]
        othermap2 = JA[cat][phone][ex][indice_carte_non_int]

        plt.figure()
        f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        f.suptitle("Clustering 1 : FRJA-R")
        ax[0,0].imshow(map1, aspect='auto')
        ax[0,0].set_title("FR-int")
        ax[0,1].imshow(othermap1, aspect='auto')
        ax[0,1].set_title("FR-nonInt")
        ax[1,0].imshow(map2, aspect='auto')
        ax[1,0].set_title("JA-int")
        ax[1,1].imshow(othermap2, aspect='auto')
        ax[1,1].set_title("JA-nonInt")

    elif clustering == '1bis':
        phone = 'v'
        cat = 'correct_pasOK'
        ex = 8
        map1 = FR[cat][phone][ex][indice_carte_interessante]
        othermap1 = FR[cat][phone][ex][indice_carte_non_int]

        map2 = JA[cat][phone][ex][indice_carte_interessante]
        othermap2 = JA[cat][phone][ex][indice_carte_non_int]

        plt.figure()
        f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        f.suptitle("Clustering 2 : FRJA-V")
        ax[0,0].imshow(map1, aspect='auto')
        ax[0,0].set_title("FR-int")
        ax[0,1].imshow(othermap1, aspect='auto')
        ax[0,1].set_title("FR-nonInt")
        ax[1,0].imshow(map2, aspect='auto')
        ax[1,0].set_title("JA-int")
        ax[1,1].imshow(othermap2, aspect='auto')
        ax[1,1].set_title("JA-nonInt")

    elif clustering == 2:
        phone1 = 'R'
        phone2 = 'v'
        cat = 'correct_pasOK'
        ex = 8

        map1 = FR[cat][phone1][ex][indice_carte_interessante]
        othermap1 = FR[cat][phone1][ex][indice_carte_non_int]

        map2 = FR[cat][phone2][ex][indice_carte_interessante]
        othermap2 = FR[cat][phone2][ex][indice_carte_non_int]

        plt.figure()
        f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        f.suptitle("Clustering 3 : FR-RV")
        ax[0,0].imshow(map1, aspect='auto')
        ax[0,0].set_title("R-int")
        ax[0,1].imshow(othermap1, aspect='auto')
        ax[0,1].set_title("R-nonInt")
        ax[1,0].imshow(map2, aspect='auto')
        ax[1,0].set_title("V-int")
        ax[1,1].imshow(othermap2, aspect='auto')
        ax[1,1].set_title("V-nonInt")

    elif clustering == '3':
        phone = 'R'
        cat1 = 'correct_OK'
        cat2 = 'incorrect_OK'
        ex = 8

        map1 = JA[cat1][phone][ex][indice_carte_interessante]
        othermap1 = JA[cat1][phone][ex][indice_carte_non_int]

        map2 = JA[cat2][phone][ex][indice_carte_interessante]
        othermap2 = JA[cat2][phone][ex][indice_carte_non_int]

        plt.figure()
        f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        f.suptitle("Clustering 4 : CIC-R (JA)")
        ax[0,0].imshow(map1, aspect='auto')
        ax[0,0].set_title("Correct-int")
        ax[0,1].imshow(othermap1, aspect='auto')
        ax[0,1].set_title("Correct-nonInt")
        ax[1,0].imshow(map2, aspect='auto')
        ax[1,0].set_title("Incorrect-int")
        ax[1,1].imshow(othermap2, aspect='auto')
        ax[1,1].set_title("Incorrect-nonInt")


    elif clustering == '3bis':
        phone = 'v'
        cat1 = 'correct_OK'
        cat2 = 'incorrect_OK'
        ex = 8

        map1 = JA[cat1][phone][ex][indice_carte_interessante]
        othermap1 = JA[cat1][phone][ex][indice_carte_non_int]

        map2 = JA[cat2][phone][ex][indice_carte_interessante]
        othermap2 = JA[cat2][phone][ex][indice_carte_non_int]

        plt.figure()
        f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        f.suptitle("Clustering 5 : CIC-V (JA)")
        ax[0,0].imshow(map1, aspect='auto')
        ax[0,0].set_title("Correct-int")
        ax[0,1].imshow(othermap1, aspect='auto')
        ax[0,1].set_title("Correct-nonInt")
        ax[1,0].imshow(map2, aspect='auto')
        ax[1,0].set_title("Incorrect-int")
        ax[1,1].imshow(othermap2, aspect='auto')
        ax[1,1].set_title("Incorrect-nonInt")


    for i in range(2):
        for j in range(2):
            ax[i,j].xaxis.set_visible(False)
            ax[i,j].yaxis.set_visible(False)
    plt.axis('off')
    plt.show()
    plt.savefig('../resultats/'+str(couche)+'/carte'+str(indice_carte_interessante)+str(couche)+'clus'+clustering+'.png')
