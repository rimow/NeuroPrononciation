from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN

from mapsAnalysis.SupprimerCartesVides import strategie_trois_l1
from process_activation_maps import load_maps
import numpy as np
from mapsAnalysis.utiles import *



def MapsClustering(couche = 'conv1', seuilCartesVides = 559, algorithme = 'kmeansNonInit', fichier = True):
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
    map_file_FR = "../maps/BREF80_l_" + couche + "_35maps_th0.500000.pkl"
    map_file_JA = "../maps/PHONIM_l_" + couche + "_35maps_th0.001000.pkl"
    FR= load_maps(map_file_FR)
    JA = load_maps(map_file_JA)

    #recuperation des dimensions pour un dictionnaire
    tableau = np.array(FR['correct_OK']['R'])
    taille=tableau.shape
    listeVide = []
    if couche != "dense1":
        listeVide = strategie_trois_l1([FR, JA], seuilCartesVides)


    #creation des fichiers d'enregistrement
    if fichier == True:
        fichier1 = "../resultats/" + couche + "/" + algorithme +"/pourcentagesFRJA_R.csv"
        fichier1bis = "../resultats/" + couche + "/" + algorithme +"/pourcentagesFRJA_V.csv"
        fichier2 = "../resultats/" + couche + "/" + algorithme +"/pourcentagesRV.csv"
        fichier3 = "../resultats/" + couche + "/" + algorithme +"/pourcentagesCIC_R.csv"
        fichier3bis = "../resultats/" + couche + "/" + algorithme +"/pourcentagesCIC_V.csv"
    else:
        fichier1 = None
        fichier1bis = None
        fichier2 = None
        fichier3 = None
        fichier3bis = None

    #creation des matrices ratio
    pourcentagesFRJA_R = []
    pourcentagesFRJA_V = []
    pourcentagesRV = []
    pourcentagesCIC_R = []
    pourcentagesCIC_V = []

    #Colonne indices que l'on va inserer a chaque fois dans matIndicesCartes
    indices=[]
    #Matrice indices de cartes qui donnent 2 classes, 5 colonnes pour chaque clustering
    matIndicesCartes = []

    ################################################################################
    #Appel du clustering
    ################################################################################

    if algorithme == "kmeansNonInit":
        clus = KMeans(n_clusters=2, init='k-means++')
    elif algorithme == "kmeansInit":
        None
    elif algorithme == "MeanShift":
        clus = MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=1)
    elif algorithme == "DBSCAN":
        clus = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None)
    else:
        print "l'algorithme demande n'existe pas essayez : kmeansNonInit, kmeansInit, MeanShift, DBSCAN"


    ################################################################################
    #Clustering 1
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering
    #creation du tenser
    Mat, Reference = pretraitementMatrice([FR, JA],FR.keys(),['R'])
    #decommenter ces lignes pour ne prendre en compte que les element qui ont ete bien classes (correctOK et incorrectOK)
    #indicesOK = [index for index,row in enumerate(Reference) if ((row[1]==1) or (row[1]==2))]
    #Mat = Mat[:,indicesOK,:]
    #Reference = Reference[indicesOK,:]
    #calcul des ratios de classement
    for i in range (taille[1]):
            if not(i in listeVide):
                if algorithme=="kmeansInit":
                    type_clustering = 'FRJAP_R'
                    centres = initialisation_centres (type_clustering, Mat, Reference)
                    clus = KMeans(n_clusters=2, init=centres)
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
            if algorithme=="kmeansInit":
                type_clustering = 'FRJAP_v'
                centres = initialisation_centres (type_clustering, Mat, Reference)
                clus = KMeans(n_clusters=2, init=centres)
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
            if algorithme=="kmeansInit":
                type_clustering = 'R_v'
                centres = initialisation_centres (type_clustering, Mat, Reference)
                clus = KMeans(n_clusters=2, init=centres)
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
            if algorithme=="kmeansInit":
                type_clustering = 'CIC_R'
                centres = initialisation_centres (type_clustering, Mat, Reference)
                clus = KMeans(n_clusters=2, init=centres)
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
            if algorithme=="kmeansInit":
                type_clustering = 'CIC_v'
                centres = initialisation_centres (type_clustering, Mat, Reference)
                clus = KMeans(n_clusters=2, init=centres)
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

    matIndicesCartes=np.array(matIndicesCartes)
    return pourcentagesFRJA_R, pourcentagesFRJA_V, pourcentagesRV, pourcentagesCIC_R, pourcentagesCIC_V, matIndicesCartes
