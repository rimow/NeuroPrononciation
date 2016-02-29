import csv
import numpy as np
from mapsAnalysis.utiles import bienClusterise
from mapsClustering.MapsClustering import MapsClustering


from itertools import izip_longest

from mapsClustering.maps5Clustering import Maps5Clustering


def GenerationClustering(couche = "conv1", seuilSuppression = 559, seuilBonClustering = 30, fichier = True):

    #############################################################################################
    # Appel de MapsClustering et enregistrement des matrices des indices pour chaque algorithme de chaque couche;
    # Matrice des indices : 5 colonnes representant les 5 clusterings, et dans chaque colonne les indices de cartes
    # qui donnent bien deux classes.
    #############################################################################################

    pFRJA_R_KMNI, pFRJA_V_KMNI, pFR_RV_KMNI, pCIC_R_KMNI, pCIC_V_KMNI, pourcentagesFR_Rlvb0, pourcentagesFR_Rlvb1,pourcentagesFR_Rlvb2, pourcentagesFR_Rlvb3, ind = Maps5Clustering(couche, seuilSuppression, fichier)
    matKmeansNonInit = str("../resultats2/matKmeansNonInit_indices_bonnes_cartes.csv")
    f1 = open(matKmeansNonInit, "wb")
    writer = csv.writer(f1)
    writer.writerow(["KmeansNonInit(1)", "KmeansNonInit(1bis)", "KmeansNonInit(2)", "KmeansNonInit(3)", "KmeansNonInit(3bis)"])
    for values in izip_longest(*ind):
        writer.writerow(values)



    #############################################################################################
    # Fichiers de cartes bon clustering
    #############################################################################################

    filename = str("../resultats2/cartes_bon_clustering_kmeans5")
    f = open(filename, "wb")

    #############################################################################
    #appel directement avec les matrices
    ##############################################################################

    f.write("FRJA R\n")


    clus = bienClusterise(MatriceClustering=pFRJA_R_KMNI, seuil=seuilBonClustering, indices=ind[0])
    f.write("kmeansNonInit:" + str(clus)+"\n")


    f.write("FRJA V\n")

    clus = bienClusterise(MatriceClustering=pFRJA_V_KMNI, seuil=seuilBonClustering,  indices=ind[1])
    f.write("kmeansNonInit:" + str(clus)+"\n")


    f.write("FR  RV\n")

    clus = bienClusterise(MatriceClustering=pFR_RV_KMNI, seuil=seuilBonClustering,  indices=ind[2])
    f.write("kmeansNonInit:" + str(clus)+"\n")



    f.write("JA correct/incorrect R\n")
    #
    clus = bienClusterise(MatriceClustering=pCIC_R_KMNI, seuil=seuilBonClustering,  indices=ind[3])
    f.write("kmeansNonInit:" + str(clus)+"\n")



    f.write("JA correct/incorrect V\n")
    #
    clus = bienClusterise(MatriceClustering=pCIC_V_KMNI, seuil=seuilBonClustering,  indices=ind[4])
    f.write("kmeansNonInit:" + str(clus)+"\n")

    f.write("FR_Rlvb\n")

    f.write("classe0")
    clus = bienClusterise(MatriceClustering=pourcentagesFR_Rlvb0, seuil=seuilBonClustering,  indices=ind[5])
    f.write("kmeansNonInit:" + str(clus)+"\n")
    f.write("classe1")
    clus = bienClusterise(MatriceClustering=pourcentagesFR_Rlvb1, seuil=seuilBonClustering,  indices=ind[5])
    f.write("kmeansNonInit:" + str(clus)+"\n")
    f.write("classe2")
    clus = bienClusterise(MatriceClustering=pourcentagesFR_Rlvb2, seuil=seuilBonClustering,  indices=ind[5])
    f.write("kmeansNonInit:" + str(clus)+"\n")
    f.write("classe3")
    clus = bienClusterise(MatriceClustering=pourcentagesFR_Rlvb3, seuil=seuilBonClustering,  indices=ind[5])
    f.write("kmeansNonInit:" + str(clus)+"\n")

    f.close()




GenerationClustering(couche = "conv1", seuilSuppression = 559, seuilBonClustering = 30, fichier = True)
