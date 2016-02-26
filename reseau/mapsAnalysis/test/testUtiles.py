from mapsAnalysis.utiles import *
import numpy as np
###############    Test ratios   ################
print "\n###################################\n"
print "Test ratios\n"
print "###################################\n"

# Test case 1
Y_Cluster = []
Reference = []

ratio = ratios ( Y_Cluster , Reference, nb_classes=2)
if ratio == []:
    print "Test Successes"
else:
    print "Test Fails"

# Test case 2
Y_Cluster = [0,1,0,0]
Reference = [0,0,0]
ratio = ratios ( Y_Cluster , Reference, nb_classes=2)
if ratio == []:
    print "Test Successes"
else:
    print "Test Fails"

#Test case 3
Y_Cluster = [0,1,0,0]
Reference = [0,1,0,1]
ratio = ratios ( Y_Cluster , Reference, nb_classes=2)
if np.array_equal(ratio,[[100,50],[0,50]]):
    print "Test Successes"
else:
    print "Test Fails"

#Test case 4
Y_Cluster = [0,1,2,3]
Reference = [0,1,0,1]
ratio = ratios ( Y_Cluster , Reference, nb_classes=4)
if np.array_equal(ratio,[[50,0],[0,50],[50,0],[0,50]]):
    print "Test Successes"
else:
    print "Test Fails"

###############    Test bienClusterise  ################
print "\n###################################\n"
print "tests bienClusterise\n"
print "###################################\n"

#test 1 : avec la matrice vide
bon = bienClusterise (fichierClustering = None, MatriceClustering = [],seuil = 30,  indices=[0,1,2])
if (bon == []):
    print "test 1 : avec la matrice vide :  OK"
else:
    print "test 1 : avec la matrice vide :  KO"

#test 2 : test du critere
#ligne 0 : seuil bon et bonne repartition
#ligne 1 : seuil mauvais et les 2 < 50
#ligne 2 : seuil bon et bonne repartition
#ligne 3 : seuil mauvais et les 2 > 50
#ligne 4 : seuil mauvais et bonne repartition
#ligne 5 : seuil bon et les 2 >50
#ligne 6 : seuil bon et les 2 < 50pente translation
matrice = [[0,50],[1,11],[25,76], [65,77],[49,51],[51,99],[1,49]]

bon = bienClusterise (fichierClustering = None, MatriceClustering = matrice, seuil = 30,  indices=[0,1,2,3,4,5,6])
if (bon == [0,2]):
    print "test 2 : test du critere : OK"
else:
    print "test 2 : test du critere : KO"



#test 3 : test du seuil
#ligne 0 : seuil mauvais et bonne repartition
#ligne 1 : seuil mauvais et les 2 < 50
#ligne 2 : seuil bon et bonne repartition
#ligne 3 : seuil mauvais et les 2 > 50
#ligne 4 : seuil mauvais et bonne repartition
#ligne 5 : seuil bon et les 2 >50
#ligne 6 : seuil bon et les 2 < 50
matrice = [[20,50],[1,11],[25,76], [65,77],[49,51],[51,99],[1,49]]
bon = bienClusterise (fichierClustering = None, MatriceClustering = matrice,seuil = 50,  indices=[0,1,2,3,4,5,6])
if(bon == [2]):
    print "test 3 : test du seuil : OK"
else:
    print "test 3 : test du seuil : KO"

#test 4 : avec la matrice du test 2 comme fichier csv
fichier = "test.csv"
bon = bienClusterise (fichierClustering = fichier, MatriceClustering = [], seuil = 30,  indices=[0,1,2,3,4,5,6])
if (bon == [0,2]):
    print "test 4 : avec la matrice du test 2 comme fichier csv : OK"
else:
    print "test 4 : avec la matrice du test 2 comme fichier csv : KO"


#test 5 : de la preseance du fichier sur la matrice
bon = bienClusterise (fichierClustering = fichier, MatriceClustering = matrice, seuil = 30,  indices=[0,1,2,3,4,5,6])
if (bon == [0,2]):
    print "test 5 : de la preseance du fichier sur la matrice: OK"
else:
    print "test 5 : de la preseance du fichier sur la matrice: KO"

#test 6 : test de la matrice indice
bon = bienClusterise (fichierClustering = None, MatriceClustering = matrice,seuil = 30,  indices=[10,20,30,40,50, 60, 70])
if (bon == [10, 30]):
    print "test 6 : test de la matrice indice : OK"
else:
    print "test 6 : test de la matrice indice : KO"
