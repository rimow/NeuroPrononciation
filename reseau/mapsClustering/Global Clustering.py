from mapsClustering.GenerationClustering import GenerationClustering
from mapsClustering.cartesBienClusterisantes import cartesBienClusterisantes
from mapsAnalysis.utiles import imagesCartesInteressantes


################################################################################
#  Generation de fichiers de clustering
################################################################################
# GenerationClustering(couche = "conv1", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
# GenerationClustering(couche = "conv2", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
# GenerationClustering(couche = "mp2", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
# GenerationClustering(couche = "dense1", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)


################################################################################
# Generation des matrices d'indices de cartes bien clusterisantes
################################################################################
cartesBienClusterisantes(couche = "conv1", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
cartesBienClusterisantes(couche = "conv2", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
cartesBienClusterisantes(couche = "mp2", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
cartesBienClusterisantes(couche = "dense1", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)


#Example of a call to imagesCartesInteressantes
# imagesCartesInteressantes(8, couche='conv1', clustering='3bis')