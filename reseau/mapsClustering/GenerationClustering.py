from mapsAnalysis.utiles import bienClusterise
from mapsClustering.MapsClustering import MapsClustering


def GenerationClustering(couche = "conv1", seuilSuppression = 559, seuilBonClustering = 30, fichier = True):

    vide_KMNI, pFRJA_R_KMNI, pFRJA_V_KMNI, pFR_RV_KMNI, pCIC_R_KMNI, pCIC_V_KMNI, ind = MapsClustering(couche, seuilSuppression, "kmeansNonInit", fichier)
    # vide_KMI, pFRJA_R_KMI, pFRJA_V_KMI, pFR_RV_KMI, pCIC_R_KMI, pCIC_V_KMI, ind = MapsClustering(couche, seuilSuppression, "kmeansInit", fichier)
    # vide_Agglo, pFRJA_R_Agglo, pFRJA_V_Agglo, pFR_RV_Agglo, pCIC_R_Agglo, pCIC_V_Agglo, ind = MapsClustering(couche, seuilSuppression, "agglomerativeClustering", fichier)
    vide_DBSCAN, pFRJA_R_DBSCAN, pFRJA_V_DBSCAN, pFR_RV_DBSCAN, pCIC_R_DBSCAN, pCIC_V_DBSCAN, indDBSCAN = MapsClustering(couche, seuilSuppression, "DBSCAN", fichier)
    vide_MeanShift, pFRJA_R_MeanShift, pFRJA_V_MeanShift, pFR_RV_MeanShift, pCIC_R_MeanShift, pCIC_V_MeanShift, indMeanShift = MapsClustering(couche, seuilSuppression, "MeanShift", fichier)

    print "ind DBSCAN "+ couche + " : ", indDBSCAN
    print "ind MeanShift "+ couche + " : ", indMeanShift


    filename = str("../resultats/" + couche + "/cartes_bon_clustering")
    f = open(filename, "a")

    ##############################################################################
    #appel directement avec les matrices
    ##############################################################################

    f.write("FRJA R\n")

    clus = bienClusterise(MatriceClustering=pFRJA_R_KMNI, seuil=seuilBonClustering, listeVide=vide_KMNI)
    f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pFRJA_R_KMI, seuil=seuilBonClustering, listeVide=vide_KMI)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pFRJA_R_Agglo, seuil=seuilBonClustering, listeVide=vide_Agglo)
    # f.write("Agglomerative Clustering:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFRJA_R_DBSCAN, seuil=seuilBonClustering, listeVide=vide_DBSCAN)
    f.write("DBSCAN:"+ str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFRJA_R_MeanShift, seuil=seuilBonClustering, listeVide=vide_MeanShift)
    f.write("MeanShift:" + str(clus)+"\n")


    f.write("FRJA V\n")

    clus = bienClusterise(MatriceClustering=pFRJA_V_KMNI, seuil=seuilBonClustering, listeVide=vide_KMNI)
    f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pFRJA_V_KMI, seuil=seuilBonClustering, listeVide=vide_KMI)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pFRJA_V_Agglo, seuil=seuilBonClustering, listeVide=vide_Agglo)
    # f.write("Agglomerative Clustering:"+ str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFRJA_V_DBSCAN, seuil=seuilBonClustering, listeVide=vide_DBSCAN)
    f.write("DBSCAN:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFRJA_V_MeanShift, seuil=seuilBonClustering, listeVide=vide_MeanShift)
    f.write("MeanShift:" + str(clus)+"\n")


    f.write("FR  RV\n")

    clus = bienClusterise(MatriceClustering=pFR_RV_KMNI, seuil=seuilBonClustering, listeVide=vide_KMNI)
    f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pFR_RV_KMI, seuil=seuilBonClustering, listeVide=vide_KMI)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pFR_RV_Agglo, seuil=seuilBonClustering, listeVide=vide_Agglo)
    # f.write("Agglomerative Clustering:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFR_RV_DBSCAN, seuil=seuilBonClustering, listeVide=vide_DBSCAN)
    f.write("DBSCAN:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pFR_RV_MeanShift, seuil=seuilBonClustering, listeVide=vide_MeanShift)
    f.write("MeanShift:" + str(clus)+"\n")


    f.write("JA correct/incorrect R\n")

    clus = bienClusterise(MatriceClustering=pCIC_R_KMNI, seuil=seuilBonClustering, listeVide=vide_KMNI)
    f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pCIC_R_KMI, seuil=seuilBonClustering, listeVide=vide_KMI)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pCIC_R_Agglo, seuil=seuilBonClustering, listeVide=vide_Agglo)
    # f.write("Agglomerative Clustering:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pCIC_R_DBSCAN, seuil=seuilBonClustering, listeVide=vide_DBSCAN)
    f.write("DBSCAN:"+ str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pCIC_R_MeanShift, seuil=seuilBonClustering, listeVide=vide_MeanShift)
    f.write("MeanShift:" + str(clus)+"\n")


    f.write("JA correct/incorrect V\n")

    clus = bienClusterise(MatriceClustering=pCIC_V_KMNI, seuil=seuilBonClustering, listeVide=vide_KMNI)
    f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pCIC_V_KMI, seuil=seuilBonClustering, listeVide=vide_KMI)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(MatriceClustering=pCIC_V_Agglo, seuil=seuilBonClustering, listeVide=vide_Agglo)
    # f.write("Agglomerative Clustering:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pCIC_V_DBSCAN, seuil=seuilBonClustering, listeVide=vide_DBSCAN)
    f.write("DBSCAN:" + str(clus)+"\n")
    clus = bienClusterise(MatriceClustering=pCIC_V_MeanShift, seuil=seuilBonClustering, listeVide=vide_MeanShift)
    f.write("MeanShift:" + str(clus)+"\n")

    f.close()




    ##############################################################################
    #appel avec les fichiers deja enregistres
    ##############################################################################


    # f.write("FRJA R")
    #
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansNonInit/pourcentagesFRJA_R.csv", seuil=seuilBonClustering, listeVide=vide_KMNI)
    # f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansInit/pourcentagesFRJA_R.csv", seuil=seuilBonClustering, listeVide=vide_KMI)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/agglomerativeClustering/pourcentagesFRJA_R.csv", seuil=seuilBonClustering, listeVide=vide_Agglo)
    # f.write("Agglomerative Clustering:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/DBSCAN/pourcentagesFRJA_R.csv", seuil=seuilBonClustering, listeVide=vide_DBSCAN)
    # f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/MeanShift/pourcentagesFRJA_R.csv", seuil=seuilBonClustering, listeVide=vide_MeanShift)
    # f.write("MeanShift:" + str(clus)+"\n")
    #
    #
    # f.write("FRJA V")
    #
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansNonInit/pourcentagesFRJA_V.csv", seuil=seuilBonClustering, listeVide=vide_KMNI)
    # f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansInit/pourcentagesFRJA_V.csv", seuil=seuilBonClustering, listeVide=vide_KMI)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/agglomerativeClustering/pourcentagesFRJA_V.csv", seuil=seuilBonClustering, listeVide=vide_Agglo)
    # f.write("Agglomerative Clustering:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/DBSCAN/pourcentagesFRJA_V.csv", seuil=seuilBonClustering, listeVide=vide_DBSCAN)
    # f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/MeanShift/pourcentagesFRJA_V.csv", seuil=seuilBonClustering, listeVide=vide_MeanShift)
    # f.write("MeanShift:" + str(clus)+"\n")
    #
    #
    # f.write("FR  RV")
    #
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansNonInit/pourcentagesRV.csv", seuil=seuilBonClustering, listeVide=vide_KMNI)
    # f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansInit/pourcentagesRV.csv", listeVide=vide_KMI)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/agglomerativeClustering/pourcentagesRV.csv", seuil=seuilBonClustering, listeVide=vide_Agglo)
    # f.write("Agglomerative Clustering:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/DBSCAN/pourcentagesRV.csv", seuil=seuilBonClustering, listeVide=vide_DBSCAN)
    # f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/MeanShift/pourcentagesRV.csv", seuil=seuilBonClustering, listeVide=vide_MeanShift)
    # f.write("MeanShift:" + str(clus)+"\n")
    #
    #
    # f.write("JA correct/incorrect R")
    #
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansNonInit/pourcentagesCIC_R.csv", seuil=seuilBonClustering, listeVide=vide_KMNI)
    # f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansInit/pourcentagesCIC_R.csv", seuil=seuilBonClustering, listeVide=vide_KMI)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/agglomerativeClustering/pourcentagesCIC_R.csv", seuil=seuilBonClustering, listeVide=vide_Agglo)
    # f.write("Agglomerative Clustering:" + str(clus))
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/DBSACN/pourcentagesCIC_R.csv", seuil=seuilBonClustering, listeVide=vide_DBSCAN)
    # f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/MeanShift/pourcentagesCIC_R.csv", seuil=seuilBonClustering, listeVide=vide_MeanShift)
    # f.write("MeanShift:" + str(clus)+"\n")
    #
    #
    # f.write("JA correct/incorrect V")
    #
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansNonInit/pourcentagesCIC_V.csv", seuil=seuilBonClustering, listeVide=vide_KMNI)
    # f.write("kmeansNonInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/kmeansInit/pourcentagesCIC_V.csv", seuil=seuilBonClustering, listeVide=vide_KMI)
    # f.write("kmeansInit:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/agglomerativeClustering/pourcentagesCIC_V.csv", seuil=seuilBonClustering, listeVide=vide_Agglo)
    # f.write("Agglomerative Clustering:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/DBSCAN/pourcentagesCIC_V.csv", seuil=seuilBonClustering, listeVide=vide_DBSCAN)
    # f.write("DBSCAN:" + str(clus)+"\n")
    # clus = bienClusterise(fichierClustering="../resultats/" + couche + "/MeanShift/pourcentagesCIC_V.csv", seuil=seuilBonClustering, listeVide=vide_MeanShift)
    # f.write("MeanShift:" + str(clus)+"\n")
    #
    #
    # f.close()










