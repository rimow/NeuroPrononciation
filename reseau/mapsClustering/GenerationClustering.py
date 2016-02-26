import csv
import numpy as np
from mapsAnalysis.utiles import bienClusterise
from mapsClustering.MapsClustering import MapsClustering


from itertools import izip_longest


def GenerationClustering(couche = "conv1", seuilSuppression = 559, seuilBonClustering = 30, fichier = True):

    #############################################################################################
    # Appel de MapsClustering et enregistrement des matrices des indices pour chaque algorithme de chaque couche;
    # Matrice des indices : 5 colonnes representant les 5 clusterings, et dans chaque colonne les indices de cartes
    # qui donnent bien deux classes.
    #############################################################################################

    vide_KMNI, pFRJA_R_KMNI, pFRJA_V_KMNI, pFR_RV_KMNI, pCIC_R_KMNI, pCIC_V_KMNI, ind = MapsClustering(couche, seuilSuppression, "kmeansNonInit", fichier)
    matKmeansNonInit = str("../resultats/" + couche + "/kmeansNonInit/matKmeansNonInit_indices_bonnes_cartes.csv")
    f1 = open(matKmeansNonInit, "wb")
    writer = csv.writer(f1)
    writer.writerow(["KmeansNonInit(1)", "KmeansNonInit(1bis)", "KmeansNonInit(2)", "KmeansNonInit(2bis)", "KmeansNonInit(3)"])
    for values in izip_longest(*ind):
        writer.writerow(values)

    vide_KMI, pFRJA_R_KMI, pFRJA_V_KMI, pFR_RV_KMI, pCIC_R_KMI, pCIC_V_KMI, indKmeansInit = MapsClustering(couche, seuilSuppression, "kmeansInit", fichier)
    matKmeansInit = str("../resultats/" + couche + "/kmeansInit/matKmeansInit_indices_bonnes_cartes.csv")
    f2 = open(matKmeansInit, "wb")
    writer = csv.writer(f2)
    writer.writerow(["KmeansInit(1)", "KmeansInit(1bis)", "KmeansInit(2)", "KmeansInit(2bis)", "KmeansInit(3)"])
    for values in izip_longest(*indKmeansInit):
        writer.writerow(values)


    vide_DBSCAN, pFRJA_R_DBSCAN, pFRJA_V_DBSCAN, pFR_RV_DBSCAN, pCIC_R_DBSCAN, pCIC_V_DBSCAN, indDBSCAN = MapsClustering(couche, seuilSuppression, "DBSCAN", fichier)
    matDBSCAN = str("../resultats/" + couche + "/DBSCAN/matDBSCAN_indices_bonnes_cartes.csv")
    f3 = open(matDBSCAN, "wb")
    writer = csv.writer(f3)
    writer.writerow(["DBSCAN(1)", "DBSCAN(1bis)", "DBSCAN(2)", "DBSCAN(2bis)", "DBSCAN(3)"])
    for values in izip_longest(*indDBSCAN):
        writer.writerow(values)

    # vide_MeanShift, pFRJA_R_MeanShift, pFRJA_V_MeanShift, pFR_RV_MeanShift, pCIC_R_MeanShift, pCIC_V_MeanShift, indMeanShift = MapsClustering(couche, seuilSuppression, "MeanShift", fichier)
    # matMeanShift = str("../resultats/" + couche + "/MeanShift/matMeanShift_indices_bonnes_cartes.csv")
    # f4 = open(matMeanShift, "wb")
    # writer = csv.writer(f4)
    # writer.writerow(["MeanShift(1)", "MeanShift(1bis)", "MeanShift(2)", "MeanShift(2bis)", "MeanShift(3)"])
    # for values in izip_longest(*indMeanShift):
    #     writer.writerow(values)



    #############################################################################################
    # Fichiers de cartes bon clustering
    #############################################################################################

    filename = str("../resultats/" + couche + "/cartes_bon_clustering_MeanShift")
    f = open(filename, "wb")

    #############################################################################
    #appel directement avec les matrices
    ##############################################################################

    f.write("FRJA R\n")


    clus = bienClusterise(MatriceClustering=pFRJA_R_KMNI, seuil=seuilBonClustering, indices=ind[0])
    f.write("kmeansNonInit:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFRJA_R_KMI, seuil=seuilBonClustering, indices=indKmeansInit[0])
    f.write("kmeansInit:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFRJA_R_DBSCAN, seuil=seuilBonClustering, indices=indDBSCAN[0])
    f.write("DBSCAN:"+ str(clus)+"\n")

    # clus = bienClusterise(MatriceClustering=pFRJA_R_MeanShift, seuil=seuilBonClustering,indices= indMeanShift[0])
    # f.write("MeanShift:" + str(clus)+"\n")


    f.write("FRJA V\n")

    clus = bienClusterise(MatriceClustering=pFRJA_V_KMNI, seuil=seuilBonClustering,  indices=ind[1])
    f.write("kmeansNonInit:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFRJA_V_KMI, seuil=seuilBonClustering, indices=indKmeansInit[1])
    f.write("kmeansInit:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFRJA_V_DBSCAN, seuil=seuilBonClustering, indices=indDBSCAN[1])
    f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pFRJA_V_MeanShift, seuil=seuilBonClustering,indices= indMeanShift[1])
    # f.write("MeanShift:" + str(clus)+"\n")


    f.write("FR  RV\n")
    #
    clus = bienClusterise(MatriceClustering=pFR_RV_KMNI, seuil=seuilBonClustering,  indices=ind[2])
    f.write("kmeansNonInit:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFR_RV_KMI, seuil=seuilBonClustering,indices=indKmeansInit[2])
    f.write("kmeansInit:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFR_RV_DBSCAN, seuil=seuilBonClustering, indices=indDBSCAN[2])
    f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pFR_RV_MeanShift, seuil=seuilBonClustering,  indices= indMeanShift[2])
    # f.write("MeanShift:" + str(clus)+"\n")


    f.write("JA correct/incorrect R\n")

    clus = bienClusterise(MatriceClustering=pCIC_R_KMNI, seuil=seuilBonClustering,  indices=ind[3])
    f.write("kmeansNonInit:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pCIC_R_KMI, seuil=seuilBonClustering, indices=indKmeansInit[3])
    f.write("kmeansInit:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pCIC_R_DBSCAN, seuil=seuilBonClustering,  indices=indDBSCAN[3])
    f.write("DBSCAN:"+ str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pCIC_R_MeanShift, seuil=seuilBonClustering,  indices= indMeanShift[3])
    # f.write("MeanShift:" + str(clus)+"\n")


    f.write("JA correct/incorrect V\n")

    clus = bienClusterise(MatriceClustering=pCIC_V_KMNI, seuil=seuilBonClustering,  indices=ind[4])
    f.write("kmeansNonInit:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pCIC_V_KMI, seuil=seuilBonClustering,indices=indKmeansInit[4])
    f.write("kmeansInit:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pCIC_V_DBSCAN, seuil=seuilBonClustering,  indices=indDBSCAN[4])
    f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pCIC_V_MeanShift, seuil=seuilBonClustering, indices= indMeanShift[4])
    # f.write("MeanShift:" + str(clus)+"\n")

    f.close()




    ##############################################################################
    #appel avec les fichiers deja enregistres
    ##############################################################################


    # f.write("FRJA R")
    #
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansNonInit/pourcentagesFRJA_R.csv", seuil=seuilBonClustering)
    # f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansInit/pourcentagesFRJA_R.csv", seuil=seuilBonClustering)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/DBSCAN/pourcentagesFRJA_R.csv", seuil=seuilBonClustering)
    # f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/MeanShift/pourcentagesFRJA_R.csv", seuil=seuilBonClustering)
    # f.write("MeanShift:" + str(clus)+"\n")
    #
    #
    # f.write("FRJA V")
    #
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansNonInit/pourcentagesFRJA_V.csv", seuil=seuilBonClustering)
    # f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansInit/pourcentagesFRJA_V.csv", seuil=seuilBonClustering)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/DBSCAN/pourcentagesFRJA_V.csv", seuil=seuilBonClustering)
    # f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/MeanShift/pourcentagesFRJA_V.csv", seuil=seuilBonClustering)
    # f.write("MeanShift:" + str(clus)+"\n")
    #
    #
    # f.write("FR  RV")
    #
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansNonInit/pourcentagesRV.csv", seuil=seuilBonClustering)
    # f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansInit/pourcentagesRV.csv")
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/DBSCAN/pourcentagesRV.csv", seuil=seuilBonClustering)
    # f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/MeanShift/pourcentagesRV.csv", seuil=seuilBonClustering)
    # f.write("MeanShift:" + str(clus)+"\n")
    #
    #
    # f.write("JA correct/incorrect R")
    #
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansNonInit/pourcentagesCIC_R.csv", seuil=seuilBonClustering)
    # f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansInit/pourcentagesCIC_R.csv", seuil=seuilBonClustering)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/DBSACN/pourcentagesCIC_R.csv", seuil=seuilBonClustering
    # f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/MeanShift/pourcentagesCIC_R.csv", seuil=seuilBonClustering)
    # f.write("MeanShift:" + str(clus)+"\n")
    #
    #
    # f.write("JA correct/incorrect V")
    #
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansNonInit/pourcentagesCIC_V.csv", seuil=seuilBonClustering)
    # f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansInit/pourcentagesCIC_V.csv", seuil=seuilBonClustering)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/DBSCAN/pourcentagesCIC_V.csv", seuil=seuilBonClustering)
    # f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/MeanShift/pourcentagesCIC_V.csv", seuil=seuilBonClustering)
    # f.write("MeanShift:" + str(clus)+"\n")
    #
    #
    # f.close()
