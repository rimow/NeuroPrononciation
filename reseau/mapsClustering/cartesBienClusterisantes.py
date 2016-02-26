import numpy as np
from mapsAnalysis.utiles import bienClusterise
from mapsClustering.MapsClustering import MapsClustering

def cartesBienClusterisantes(couche = "conv1", seuilSuppression = 559, seuilBonClustering = 30, fichier = False):

    pFRJA_R_KMNI, pFRJA_V_KMNI, pFR_RV_KMNI, pCIC_R_KMNI, pCIC_V_KMNI, ind = MapsClustering(couche, seuilSuppression, "kmeansNonInit", fichier)
    pFRJA_R_KMI, pFRJA_V_KMI, pFR_RV_KMI, pCIC_R_KMI, pCIC_V_KMI, indKmeansInit = MapsClustering(couche, seuilSuppression, "kmeansInit", fichier)
    pFRJA_R_DBSCAN, pFRJA_V_DBSCAN, pFR_RV_DBSCAN, pCIC_R_DBSCAN, pCIC_V_DBSCAN, indDBSCAN = MapsClustering(couche, seuilSuppression, "DBSCAN", fichier)
    # pFRJA_R_MeanShift, pFRJA_V_MeanShift, pFR_RV_MeanShift, pCIC_R_MeanShift, pCIC_V_MeanShift, indMeanShift = MapsClustering(couche, seuilSuppression, "MeanShift", fichier)

    ################################################################################
    #Initialisation des matrices
    ################################################################################

    bienClusterisesKmeansNonInit = []
    bienClusterisesKmeansInit = []
    bienClusterisesDBSCAN = []
    # bienClusterisesMeanShift = []


    ################################################################################
    #Clustering 1
    ################################################################################
    clus = bienClusterise(MatriceClustering=pFRJA_R_KMNI, seuil=seuilBonClustering, indices=ind[0])
    bienClusterisesKmeansNonInit.append(clus)

    clus = bienClusterise(MatriceClustering=pFRJA_R_KMI, seuil=seuilBonClustering, indices=indKmeansInit[0])
    bienClusterisesKmeansInit.append(clus)

    clus = bienClusterise(MatriceClustering=pFRJA_R_DBSCAN, seuil=seuilBonClustering, indices=indDBSCAN[0])
    bienClusterisesDBSCAN.append(clus)

    # clus = bienClusterise(MatriceClustering=pFRJA_R_MeanShift, seuil=seuilBonClustering, indices=indMeanShift[0])
    # bienClusterisesMeanShift.append(clus)

    ################################################################################
    #Clustering 1bis
    ################################################################################

    clus = bienClusterise(MatriceClustering=pFRJA_V_KMNI, seuil=seuilBonClustering,  indices=ind[1])
    bienClusterisesKmeansNonInit.append(clus)

    clus = bienClusterise(MatriceClustering=pFRJA_V_KMI, seuil=seuilBonClustering, indices=indKmeansInit[1])
    bienClusterisesKmeansInit.append(clus)

    clus = bienClusterise(MatriceClustering=pFRJA_V_DBSCAN, seuil=seuilBonClustering, indices=indDBSCAN[1])
    bienClusterisesDBSCAN.append(clus)

    # clus = bienClusterise(MatriceClustering=pFRJA_R_MeanShift, seuil=seuilBonClustering, indices=indMeanShift[1])
    # bienClusterisesMeanShift.append(clus)

    ################################################################################
    #Clustering 2
    ################################################################################

    clus = bienClusterise(MatriceClustering=pFR_RV_KMNI, seuil=seuilBonClustering,  indices=ind[2])
    bienClusterisesKmeansNonInit.append(clus)

    clus = bienClusterise(MatriceClustering=pFR_RV_KMI, seuil=seuilBonClustering,indices=indKmeansInit[2])
    bienClusterisesKmeansInit.append(clus)

    clus = bienClusterise(MatriceClustering=pFR_RV_DBSCAN, seuil=seuilBonClustering, indices=indDBSCAN[2])
    bienClusterisesDBSCAN.append(clus)

    # clus = bienClusterise(MatriceClustering=pFRJA_R_MeanShift, seuil=seuilBonClustering, indices=indMeanShift[2])
    # bienClusterisesMeanShift.append(clus)

    ################################################################################
    #Clustering 3
    ################################################################################

    clus = bienClusterise(MatriceClustering=pCIC_R_KMNI, seuil=seuilBonClustering,  indices=ind[3])
    bienClusterisesKmeansNonInit.append(clus)

    clus = bienClusterise(MatriceClustering=pCIC_R_KMI, seuil=seuilBonClustering, indices=indKmeansInit[3])
    bienClusterisesKmeansInit.append(clus)

    clus = bienClusterise(MatriceClustering=pCIC_R_DBSCAN, seuil=seuilBonClustering,  indices=indDBSCAN[3])
    bienClusterisesDBSCAN.append(clus)

    # clus = bienClusterise(MatriceClustering=pFRJA_R_MeanShift, seuil=seuilBonClustering, indices=indMeanShift[3])
    # bienClusterisesMeanShift.append(clus)

    ################################################################################
    #Clustering 3bis
    ################################################################################

    clus = bienClusterise(MatriceClustering=pCIC_V_KMNI, seuil=seuilBonClustering,  indices=ind[4])
    bienClusterisesKmeansNonInit.append(clus)

    clus = bienClusterise(MatriceClustering=pCIC_V_KMI, seuil=seuilBonClustering,indices=indKmeansInit[4])
    bienClusterisesKmeansInit.append(clus)

    clus = bienClusterise(MatriceClustering=pCIC_V_DBSCAN, seuil=seuilBonClustering,  indices=indDBSCAN[4])
    bienClusterisesDBSCAN.append(clus)

    # clus = bienClusterise(MatriceClustering=pFRJA_R_MeanShift, seuil=seuilBonClustering, indices=indMeanShift[4])
    # bienClusterisesMeanShift.append(clus)

    ################################################################################
    #Enregistrement des differentes matrices
    ################################################################################
    np.save('../resultats/' + couche + '/kmeansNonInit/bienClusteriseKmeansNonInit.npy', bienClusterisesKmeansNonInit)
    np.save('../resultats/' + couche + '/kmeansInit/bienClusteriseKmeansInit.npy', bienClusterisesKmeansInit)
    np.save('../resultats/' + couche + '/DBSCAN/bienClusteriseDBSCAN.npy', bienClusterisesDBSCAN)
    # np.save('../resultats/' + couche + '/MeanShift/bienClusteriseMeanShift.npy', bienClusterisesMeanShift)