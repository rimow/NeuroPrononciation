from sklearn.cluster import KMeans

from mapsAnalysis.SupprimerCartesVides import strategie_trois_l1
from process_activation_maps import load_maps
import numpy as np
from mapsAnalysis.utiles import pretraitementMatrice, ratios



def MapsClustering(couche = 'conv1'):
    """
    effectue les clusterings et pour chacun enregistre les resultats (ratios) dans un fichier different
    :param couche: nom de la couche sur laquelle on effectue les 5 clusterings
    :return: /
    """


    ################################################################################
    #Chargement des cartes d'activation et definintion des fichiers d'enregistrement
    ################################################################################

    #chargement des dictionnaires
    map_file_FR = 'maps/BREF80_l_' + couche + '_35maps_th0.500000.pkl'
    map_file_JA = 'maps/PHONIM_l_' + couche + '_35maps_th0.001000.pkl'
    FR= load_maps(map_file_FR)
    JA = load_maps(map_file_JA)

    #recuperation des dimensions pour un dictionnaire
    tableau = np.array(FR['correct_OK']['R'])
    taille=tableau.shape
    listeVide = []
    if couche != "dense1":
        listeVide = strategie_trois_l1([FR, JA], 559)


    #creation des fichiers d'enregistrement
    fichier1 = "resultats/" + couche + "_pourcentagesFRJA_R.csv"
    fichier1bis = "resultats/" + couche + "_pourcentagesFRJA_V.csv"
    fichier2 = "resultats/" + couche + "_pourcentagesRV.csv"
    fichier3 = "resultats/" + couche + "_pourcentagesCIC_R.csv"
    fichier3bis = "resultats/" + couche + "_pourcentagesCIC_V.csv"


    ################################################################################
    #Clustering 1
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering
    f = open(fichier1, "a")
    f.write("FR,JA\n")
    f.close()
    #creation du tenser et clustering
    Mat, Reference = pretraitementMatrice([FR, JA],FR.keys(),['R'])
    clus = KMeans(n_clusters=2, init='k-means++')
    #calcul des ratios de classement
    for i in range (taille[1]):
            if not(i in listeVide):
                resCluster = clus.fit(Mat[i])
                Y_Cluster = resCluster.labels_
                FRJA = ratios(Y_Cluster, Reference[:,0], fichier =  fichier1)


    ################################################################################
    #Clustering 1bis
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering
    f = open(fichier1bis, "a")
    f.write("FR,JA\n")
    f.close()
    #creation du tenser et clustering
    Mat, Reference = pretraitementMatrice([FR, JA],FR.keys(),['v'])
    clus = KMeans(n_clusters=2, init='k-means++')
    #calcul des ratios de classement
    for i in range (taille[1]):
        if not(i in listeVide):
            resCluster = clus.fit(Mat[i])
            Y_Cluster = resCluster.labels_
            FRJA = ratios(Y_Cluster, Reference[:,0], fichier = fichier1bis)


    ################################################################################
    #Clustering 2
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering
    f = open(fichier2, "a")
    f.write("R,V\n")
    f.close()
    #creation du tenser et clustering
    Mat, Reference = pretraitementMatrice([FR],FR.keys(),['R', 'v'])
    clus = KMeans(n_clusters=2, init='k-means++')
    #calcul des ratios de classement
    for i in range (taille[1]):
        if not(i in listeVide):
            resCluster = clus.fit(Mat[i])
            Y_Cluster = resCluster.labels_
            FRJA = ratios(Y_Cluster, Reference[:,2], fichier = fichier2)


    ################################################################################
    #Clustering 3
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering
    f = open(fichier3, "a")
    f.write("correct,incorrect\n")
    f.close()
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
    #clustering
    clus = KMeans(n_clusters=2, init='k-means++')
    #calcul des ratios de classement
    for i in range (taille[1]):
        if not(i in listeVide):
            resCluster = clus.fit(Mat[i])
            Y_Cluster = resCluster.labels_
            FRJA = ratios(Y_Cluster, Reference[:,1],  fichier = fichier3)


    ################################################################################
    #Clustering 3bis
    ################################################################################

    # ouverture du fichier d'ecriture et precision sur la nature du clustering
    f = open(fichier3bis, "a")
    f.write("correct,incorrect\n")
    f.close()
    #creation du tenser
    Mat, Reference = pretraitementMatrice([JA],JA.keys(),['v'])
    for i in range(len(Reference[:,1])):
        if Reference[i,1] == 1:
            Reference[i,1] = 0
        if Reference[i,1] == 2:
            Reference[i,1] = 1
        if Reference[i,1] == 3:
            Reference[i,1] = 1
    #clustering
    clus = KMeans(n_clusters=2, init='k-means++')
    #calcul des ratios de classement
    for i in range (taille[1]):
        if not(i in listeVide):
            resCluster = clus.fit(Mat[i])
            Y_Cluster = resCluster.labels_
            FRJA = ratios(Y_Cluster, Reference[:,1],  fichier = fichier3bis)

    return listeVide