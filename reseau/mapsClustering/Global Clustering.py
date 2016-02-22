from mapsClustering.GenerationClustering import GenerationClustering
from mapsAnalysis.utiles import imagesCartesInteressantes

GenerationClustering(couche = "conv1", seuilSuppression = 559, seuilBonClustering = 30, fichier = True)
GenerationClustering(couche = "conv2", seuilSuppression = 559, seuilBonClustering = 30, fichier = True)
GenerationClustering(couche = "mp2", seuilSuppression = 559, seuilBonClustering = 30, fichier = True)
GenerationClustering(couche = "dense1", seuilSuppression = 559, seuilBonClustering = 30, fichier = True)

#Example of a call to imagesCartesInteressantes
imagesCartesInteressantes(5, 6, couche='conv1', clustering='1')
