from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN

from mapsAnalysis.SupprimerCartesVides import strategie_trois_l1
from process_activation_maps import load_maps
import numpy as np
from mapsAnalysis.utiles import pretraitementMatrice, ratios



def MapsClustering(couche = 'conv1', seuilCartesVides = 559, algorithme = 'kmeansNonInit', fichier = True):
    """

    tous les clusterings pour la couche demandee
    :param couche: la couche de convolution ou de reseau e neurones sur laquelle on effectue les cluterings
    :param seuilCartesVides: seuil a partir duquel on considere que la carte est vide pour suffisamment d'exemples et ne doit pas etre prise en compte
    :param algorithme: algorithme de clustering qu'on souhaite utiliser : kmeansNonInit kmeansInit AgglomerativeClustering MeanShift DBSCAN hierarchique
    :param fichier: si on souhaite enregistrer les resultats sous des fichiers
    :return:
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

    #matrice des cartes qui marchent pour DBSCAN et MeanShift
    indiceMeanShift = []


    ################################################################################
    #Appel du clustering
    ################################################################################

    if algorithme == "kmeansNonInit":
        clus = KMeans(n_clusters=2, init='k-means++')
    # elif algorithme == "kmeansInit":
    #     clus = kmeansInit()
    elif algorithme == "agglomerativeClustering":
        # clus = AgglomerativeClustering(n_clusters=2, affinity='cosine',linkage='complete')
        clus = AgglomerativeClustering(n_clusters=2)
    elif algorithme == "MeanShift":
        clus = MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=1)
    elif algorithme == "DBSCAN":
        clus = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None)
    else:
        print "l'algorithme demande n'existe pas essayez : kmeansNonInit, kmeansInit, agglomerativeClustering, MeanShift, DBSCAN ou hierarchique"


    ################################################################################
    #Clustering 1
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering
    #creation du tenser
    Mat, Reference = pretraitementMatrice([FR, JA],FR.keys(),['R'])
    #calcul des ratios de classement
    for i in range (taille[1]):
            if not(i in listeVide):
                if algorithme=="kmeansInit":
                    type_clustering = 'FRJAP_R'
                    centres = initialisation_centres (type_clustering, Mat, Reference, [FR,JA] ,FR.keys(),['R'])
                    clus = KMeans(n_clusters=2, init=centres)
                resCluster = clus.fit(Mat[i])
                Y_Cluster = resCluster.labels_
                if ((max(Y_Cluster) +1) ==2):
                    if (algorithme == "MeanShift" or algorithme == "DBSCAN"):
                        indiceMeanShift.append(i)
                    FRJA = ratios(Y_Cluster, Reference[:,0])
                    pourcentagesFRJA_R.append(FRJA[0])
    np.savetxt(fichier1,np.atleast_2d(pourcentagesFRJA_R), delimiter =',')

    ################################################################################
    #Clustering 1bis
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering
    #creation du tenser
    Mat, Reference = pretraitementMatrice([FR, JA],FR.keys(),['v'])
    #calcul des ratios de classement
    for i in range (taille[1]):
        if not(i in listeVide):
            if algorithme=="kmeansInit":
                type_clustering = 'FRJAP_v'
                centres = initialisation_centres (type_clustering, Mat, Reference, [FR,JA] ,FR.keys(),['v'])
                clus = KMeans(n_clusters=2, init=centres)
            resCluster = clus.fit(Mat[i])
            Y_Cluster = resCluster.labels_
            if (max(Y_Cluster) +1) ==2:
                if (algorithme == "MeanShift" or algorithme == "DBSCAN"):
                        indiceMeanShift.append(i)
                FRJA = ratios(Y_Cluster, Reference[:,0])
                pourcentagesFRJA_V.append(FRJA[0])
    np.savetxt(fichier1bis,np.atleast_2d(pourcentagesFRJA_R), delimiter =',')

    ################################################################################
    #Clustering 2
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering

    #creation du tenser
    Mat, Reference = pretraitementMatrice([FR],FR.keys(),['R', 'v'])
    #calcul des ratios de classement
    for i in range (taille[1]):
        if not(i in listeVide):
            if algorithme=="kmeansInit":
                type_clustering = 'R_v'
                centres = initialisation_centres (type_clustering, Mat, Reference, [FR] ,FR.keys(),['R','v'])
                clus = KMeans(n_clusters=2, init=centres)
            resCluster = clus.fit(Mat[i])
            Y_Cluster = resCluster.labels_
            if (max(Y_Cluster) +1) ==2:
                if (algorithme == "MeanShift" or algorithme == "DBSCAN"):
                        indiceMeanShift.append(i)
                FRJA = ratios(Y_Cluster, Reference[:,2])
                pourcentagesRV.append(FRJA[0])
    np.savetxt(fichier2, np.atleast_2d(pourcentagesRV), delimiter =',')

    ################################################################################
    #Clustering 3
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering

    #creation du tenser
    Mat, Reference = pretraitementMatrice([JA],JA.keys(),['R'])
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
                if (algorithme == "MeanShift" or algorithme == "DBSCAN"):
                        indiceMeanShift.append(i)
                FRJA = ratios(Y_Cluster, Reference[:,1])
                pourcentagesCIC_R.append(FRJA[0])
    np.savetxt(fichier3, np.atleast_2d(pourcentagesCIC_R), delimiter =',')


    ################################################################################
    #Clustering 3bis
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering

    #creation du tenser
    Mat, Reference = pretraitementMatrice([JA],JA.keys(),['v'])
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
                if (algorithme == "MeanShift" or algorithme == "DBSCAN"):
                        indiceMeanShift.append(i)
                FRJA = ratios(Y_Cluster, Reference[:,1])
                pourcentagesCIC_V.append(FRJA[0])
    np.savetxt(fichier3bis, np.atleast_2d(pourcentagesCIC_V), delimiter =',')

    return listeVide, pourcentagesFRJA_R, pourcentagesFRJA_V, pourcentagesRV, pourcentagesCIC_R, pourcentagesCIC_V, indiceMeanShift
