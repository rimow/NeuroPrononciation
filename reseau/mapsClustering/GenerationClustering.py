from mapsAnalysis.utiles import bienClusterise
from mapsClustering.MapsClustering import MapsClustering


vide_conv1 = MapsClustering("conv1")
vide_conv2 = MapsClustering("conv2")
vide_mp2 = MapsClustering("mp2")
vide_dense1 = MapsClustering("dense1")

clus = bienClusterise("resultats/conv1_pourcentagesFRJA_R.csv", 30, vide_conv1)
print "FRJA R conv1:", clus
clus = bienClusterise("resultats/conv1_pourcentagesFRJA_V.csv", 30, vide_conv1)
print "FRJA V conv1:", clus
clus = bienClusterise("resultats/conv1_pourcentagesRV.csv", 30, vide_conv1)
print "FR RV conv1:", clus
clus = bienClusterise("resultats/conv1_pourcentagesCIC_R.csv", 30, vide_conv1)
print "JA correct/incorrect R conv1:", clus
clus = bienClusterise("resultats/conv1_pourcentagesCIC_R.csv", 30, vide_conv1)
print "JA correct/incorrect conv1:", clus



clus = bienClusterise("resultats/conv2_pourcentagesFRJA_R.csv", 30, vide_conv2)
print "FRJA R conv2:", clus
clus = bienClusterise("resultats/conv2_pourcentagesFRJA_V.csv", 30, vide_conv2)
print "FRJA V conv2:", clus
clus = bienClusterise("resultats/conv2_pourcentagesRV.csv", 30, vide_conv2)
print "FR RV conv2:", clus
clus = bienClusterise("resultats/conv2_pourcentagesCIC_R.csv", 30, vide_conv2)
print "JA correct/incorrect R conv2:", clus
clus = bienClusterise("resultats/conv2_pourcentagesCIC_R.csv", 30, vide_conv2)
print "JA correct/incorrect conv2:", clus


clus = bienClusterise("resultats/mp2_pourcentagesFRJA_R.csv", 30, vide_mp2)
print "FRJA R mp2:", clus
clus = bienClusterise("resultats/mp2_pourcentagesFRJA_V.csv", 30, vide_mp2)
print "FRJA V mp2:", clus
clus = bienClusterise("resultats/mp2_pourcentagesRV.csv", 30, vide_mp2)
print "FR RV mp2:", clus
clus = bienClusterise("resultats/mp2_pourcentagesCIC_R.csv", 30, vide_mp2)
print "JA correct/incorrect R mp2:", clus
clus = bienClusterise("resultats/mp2_pourcentagesCIC_R.csv", 30, vide_mp2)
print "JA correct/incorrect mp2:", clus


clus = bienClusterise("resultats/dense1_pourcentagesFRJA_R.csv", 30, vide_dense1)
print "FRJA R dense1:", clus
clus = bienClusterise("resultats/dense1_pourcentagesFRJA_V.csv", 30, vide_dense1)
print "FRJA V dense1:", clus
clus = bienClusterise("resultats/dense1_pourcentagesRV.csv", 30, vide_dense1)
print "FR RV dense1:", clus
clus = bienClusterise("resultats/dense1_pourcentagesCIC_R.csv", 30, vide_dense1)
print "JA correct/incorrect R dense1:", clus
clus = bienClusterise("resultats/dense1_pourcentagesCIC_R.csv", 30, vide_dense1)
print "JA correct/incorrect dense1:", clus