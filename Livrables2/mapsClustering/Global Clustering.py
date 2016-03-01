from mapsClustering.GenerationClustering import GenerationClustering
from mapsClustering.cartesBienClusterisees import cartesBienClusterisees
from mapsAnalysis.utiles import imagesCartesInteressantes


################################################################################
#  Generation de fichiers de clustering
################################################################################
GenerationClustering(couche = "conv1", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
GenerationClustering(couche = "conv2", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
GenerationClustering(couche = "mp2", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
GenerationClustering(couche = "dense1", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)


################################################################################
# Generation des matrices d'indices de cartes bien clusterisantes
################################################################################
# cartesBienClusterisees(couche = "conv1", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
# cartesBienClusterisees(couche = "conv2", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
# cartesBienClusterisees(couche = "mp2", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)
# cartesBienClusterisees(couche = "dense1", seuilSuppression = 559, seuilBonClustering = 30, fichier = False)


#Example of a call to imagesCartesInteressantes
# imagesCartesInteressantes(8, couche='conv1', clustering='1')