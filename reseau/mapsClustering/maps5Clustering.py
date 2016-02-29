from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN

from mapsAnalysis.SupprimerCartesVides import strategie_trois_l1
from process_activation_maps import load_maps
import numpy as np
from mapsAnalysis.utiles import *



def Maps5Clustering(couche = 'conv1', seuilCartesVides = 559, fichier = True):
    """
    tous les clusterings pour la couche demandee
    :param couche: la couche de convolution ou de reseau de neurones sur laquelle on effectue les cluterings
    :param seuilCartesVides: seuil a partir duquel on considere que la carte est vide pour suffisamment d'exemples et ne doit pas etre prise en compte
    :param algorithme: algorithme de clustering qu'on souhaite utiliser : kmeansNonInit kmeansInit AgglomerativeClustering MeanShift DBSCAN hierarchique
    :param fichier: si on souhaite enregistrer les resultats sous des fichiers
    :return: les fichiers des differents pourcentages, ainsi que la matrice des indices de cartes qui nous donnent deux classes
    """


    ################################################################################
    #Chargement des cartes d'activation et definintion des fichiers d'enregistrement
    ################################################################################

    #chargement des dictionnaires
    map_file_FR = "../maps2/BREF80_l_conv1_35maps_th0.500000.pkl"
    map_file_JA = "../maps2/PHONIM_l_conv1_35maps_th0.001000.pkl"
    FR= load_maps(map_file_FR)
    JA = load_maps(map_file_JA)

    #recuperation des dimensions pour un dictionnaire
    tableau = np.array(FR['correct_OK']['R'])
    taille=tableau.shape
    listeVide = []


    #creation des fichiers d'enregistrement
    if fichier == True:
        fichier1 = "../resultats2/pourcentagesFRJA_R.csv"
        fichier1bis = "../resultats2/pourcentagesFRJA_V.csv"
        fichier2 = "../resultats2/pourcentagesRV.csv"
        fichier3 = "../resultats2/pourcentagesCIC_R.csv"
        fichier3bis = "../resultats2/pourcentagesCIC_V.csv"
        fichier4 = "../resultats2/pourcentagesFRJA_l.csv"
        fichier5 = "../resultats2/pourcentagesFRJA_b.csv"
        fichier6 = "../resultats2/pourcentagesFR_Rlvb_cl0.csv"
        fichier7 = "../resultats2/pourcentagesFR_Rlvb_cl1.csv"
        fichier8 = "../resultats2/pourcentagesFR_Rlvb_cl2.csv"
        fichier9 = "../resultats2/pourcentagesFR_Rlvb_cl3.csv"


    else:
        fichier1 = None
        fichier1bis = None
        fichier2 = None
        fichier3 = None
        fichier3bis = None
        fichier4 = None
        fichier5 = None
        fichier6 = None
        fichier7 =None
        fichier8=None
        fichier9=None

    #creation des matrices ratio
    pourcentagesFRJA_R = []
    pourcentagesFRJA_V = []
    pourcentagesRV = []
    pourcentagesCIC_R = []
    pourcentagesCIC_V = []
    pourcentagesFRJA_l = []
    pourcentagesFRJA_b = []
    pourcentagesFR_Rlvb0 = []
    pourcentagesFR_Rlvb1 = []
    pourcentagesFR_Rlvb2 = []
    pourcentagesFR_Rlvb3 = []

    #Colonne indices que l'on va inserer a chaque fois dans matIndicesCartes
    indices=[]
    #Matrice indices de cartes qui donnent 2 classes, 5 colonnes pour chaque clustering
    matIndicesCartes = []


    ################################################################################
    #Clustering 1
    ################################################################################
    clus = KMeans(n_clusters=2, init='k-means++')
    #creation du tenser
    Mat, Reference = pretraitementMatrice([FR, JA],FR.keys(),['R'])
    #calcul des ratios de classement
    for i in range (taille[1]):
            if not(i in listeVide):
                resCluster = clus.fit(Mat[i])
                Y_Cluster = resCluster.labels_
                if (max(Y_Cluster) +1) ==2:
                    indices.append(i)
                    FRJA = ratios(Y_Cluster, Reference[:,0])
                    pourcentagesFRJA_R.append(FRJA[0])
    matIndicesCartes.append(indices)
    if fichier == True:
        f = open(fichier1, "wb")
        writer = csv.writer(f)
        writer.writerow(["FR-R", "JA-R"])
        np.savetxt(f,np.atleast_2d(pourcentagesFRJA_R), delimiter =',')


    ################################################################################
    #Clustering 1bis
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering
    #creation du tenser
    Mat, Reference = pretraitementMatrice([FR, JA],FR.keys(),['v'])
    indices=[]
    #calcul des ratios de classement
    for i in range (taille[1]):
        if not(i in listeVide):
            resCluster = clus.fit(Mat[i])
            Y_Cluster = resCluster.labels_
            if (max(Y_Cluster) +1) ==2:
                indices.append(i)
                FRJA = ratios(Y_Cluster, Reference[:,0])
                pourcentagesFRJA_V.append(FRJA[0])
    matIndicesCartes.append(indices)
    if fichier == True:
        f = open(fichier1bis, "wb")
        writer = csv.writer(f)
        writer.writerow(["FR-V", "JA-V"])
        np.savetxt(f,np.atleast_2d(pourcentagesFRJA_V), delimiter =',')



    ################################################################################
    #Clustering 2
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering

    #creation du tenser
    Mat, Reference = pretraitementMatrice([FR],FR.keys(),['R', 'v'])
    indices=[]
    #calcul des ratios de classement
    for i in range (taille[1]):
        if not(i in listeVide):
            resCluster = clus.fit(Mat[i])
            Y_Cluster = resCluster.labels_
            if (max(Y_Cluster) +1) ==2:
                indices.append(i)
                FRJA = ratios(Y_Cluster, Reference[:,2])
                pourcentagesRV.append(FRJA[0])
    matIndicesCartes.append(indices)
    if fichier == True:
        f = open(fichier2, "wb")
        writer = csv.writer(f)
        writer.writerow(["FR-R", "FR-V"])
        np.savetxt(f, np.atleast_2d(pourcentagesRV), delimiter =',')


    ################################################################################
    #Clustering 3
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering

    #creation du tenser
    Mat, Reference = pretraitementMatrice([JA],JA.keys(),['R'])
    indices=[]
    #rassemblement des correct_OK et correct_PasOK et des incorrect_OK et incorrect_PasOK en correct et incorrect
    for i in range(len(Reference[:,1])):
        if Reference[i,1] == 1:
            Reference[i,1] = 0
        if Reference[i,1] == 2:
            Reference[i,1] = 1
        if Reference[i,1] == 3:
            Reference[i,1] = 1
    #calcul des ratios de classement
    for i in range (taille[1]):
        if not(i in listeVide):
            resCluster = clus.fit(Mat[i])
            Y_Cluster = resCluster.labels_
            if (max(Y_Cluster) +1) ==2:
                indices.append(i)
                FRJA = ratios(Y_Cluster, Reference[:,1])
                pourcentagesCIC_R.append(FRJA[0])
    matIndicesCartes.append(indices)
    if fichier == True:
        f = open(fichier3, "wb")
        writer = csv.writer(f)
        writer.writerow(["JA-CORR-R", "JA-INC-R"])
        np.savetxt(f, np.atleast_2d(pourcentagesCIC_R), delimiter =',')


    ################################################################################
    #Clustering 3bis
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering

    #creation du tenser
    Mat, Reference = pretraitementMatrice([JA],JA.keys(),['v'])
    indices=[]
    for i in range(len(Reference[:,1])):
        if Reference[i,1] == 1:
            Reference[i,1] = 0
        if Reference[i,1] == 2:
            Reference[i,1] = 1
        if Reference[i,1] == 3:
            Reference[i,1] = 1
    #calcul des ratios de classement
    for i in range (taille[1]):
        if not(i in listeVide):
            resCluster = clus.fit(Mat[i])
            Y_Cluster = resCluster.labels_
            if (max(Y_Cluster) +1) ==2:
                indices.append(i)
                FRJA = ratios(Y_Cluster, Reference[:,1])
                pourcentagesCIC_V.append(FRJA[0])
    matIndicesCartes.append(indices)
    if fichier == True:
        f = open(fichier3bis, "wb")
        writer = csv.writer(f)
        writer.writerow(["JA-CORR-V", "JA-INC-V"])
        np.savetxt(f, np.atleast_2d(pourcentagesCIC_V), delimiter =',')

    # ################################################################################
    # #Clustering 4
    # ################################################################################
    #
    # # ouverture du fichier d'ecriture et precision sur la nature du clustering
    # #creation du tenser
    # Mat, Reference = pretraitementMatrice([FR, JA],FR.keys(),['l'])
    # indices=[]
    # #calcul des ratios de classement
    # for i in range (taille[1]):
    #     if not(i in listeVide):
    #         resCluster = clus.fit(Mat[i])
    #         Y_Cluster = resCluster.labels_
    #         if (max(Y_Cluster) +1) ==2:
    #             indices.append(i)
    #             FRJA = ratios(Y_Cluster, Reference[:,0])
    #             pourcentagesFRJA_l.append(FRJA[0])
    # matIndicesCartes.append(indices)
    # if fichier == True:
    #     f = open(fichier4, "wb")
    #     writer = csv.writer(f)
    #     writer.writerow(["FR-l", "JA-l"])
    #     np.savetxt(f,np.atleast_2d(pourcentagesFRJA_l), delimiter =',')
    #
    #     ################################################################################
    # #Clustering 5
    # ################################################################################
    #
    # # ouverture du fichier d'ecriture et precision sur la nature du clustering
    # #creation du tenser
    # Mat, Reference = pretraitementMatrice([FR, JA],FR.keys(),['b'])
    # indices=[]
    # #calcul des ratios de classement
    # for i in range (taille[1]):
    #     if not(i in listeVide):
    #         resCluster = clus.fit(Mat[i])
    #         Y_Cluster = resCluster.labels_
    #         if (max(Y_Cluster) +1) ==2:
    #             indices.append(i)
    #             FRJA = ratios(Y_Cluster, Reference[:,0])
    #             pourcentagesFRJA_b.append(FRJA[0])
    # matIndicesCartes.append(indices)
    # if fichier == True:
    #     f = open(fichier5, "wb")
    #     writer = csv.writer(f)
    #     writer.writerow(["FR-b", "JA-b"])
    #     np.savetxt(f,np.atleast_2d(pourcentagesFRJA_b), delimiter =',')
    #
    # #
    # ################################################################################
    # #Clustering 6
    # ################################################################################
    clus = KMeans(n_clusters=4, init='k-means++')
    #creation du tenser
    Mat, Reference = pretraitementMatrice([FR],FR.keys(),['R','l','v','b'])
    #calcul des ratios de classement
    for i in range (taille[1]):
            if not(i in listeVide):
                resCluster = clus.fit(Mat[i])
                Y_Cluster = resCluster.labels_
                if (max(Y_Cluster) +1) ==4:
                    indices.append(i)
                    FRJA = ratios(Y_Cluster, Reference[:,2],nb_classes=4)
                    pourcentagesFR_Rlvb0.append(FRJA[0])
                    pourcentagesFR_Rlvb1.append(FRJA[1])
                    pourcentagesFR_Rlvb2.append(FRJA[2])
                    pourcentagesFR_Rlvb3.append(FRJA[3])

    matIndicesCartes.append(indices)
    if fichier == True:
        f = open(fichier6, "wb")
        writer = csv.writer(f)
        writer.writerow(["FR-R", "FR-l", "FR-v", "FR-b"])
        np.savetxt(f,np.atleast_2d(pourcentagesFR_Rlvb0), delimiter =',')

        f = open(fichier7, "wb")
        writer = csv.writer(f)
        writer.writerow(["FR-R", "FR-l", "FR-v", "FR-b"])
        np.savetxt(f,np.atleast_2d(pourcentagesFR_Rlvb1), delimiter =',')

        f = open(fichier8, "wb")
        writer = csv.writer(f)
        writer.writerow(["FR-R", "FR-l", "FR-v", "FR-b"])
        np.savetxt(f,np.atleast_2d(pourcentagesFR_Rlvb2), delimiter =',')

        f = open(fichier9, "wb")
        writer = csv.writer(f)
        writer.writerow(["FR-R", "FR-l", "FR-v", "FR-b"])
        np.savetxt(f,np.atleast_2d(pourcentagesFR_Rlvb3), delimiter =',')

    matIndicesCartes=np.array(matIndicesCartes)
    return pourcentagesFRJA_R, pourcentagesFRJA_V, pourcentagesRV, pourcentagesCIC_R, pourcentagesCIC_V, pourcentagesFR_Rlvb0,pourcentagesFR_Rlvb1,pourcentagesFR_Rlvb2, pourcentagesFR_Rlvb3, matIndicesCartes
