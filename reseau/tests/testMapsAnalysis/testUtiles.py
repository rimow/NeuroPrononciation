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


########## Tests de initialisation centres ##################
#test 1 : test du format de sortie
map_file_FR = '../maps/BREF80_l_' + couche + '_35maps_th0.500000.pkl'
FR= load_maps(map_file_FR)

type_clustering = 'R_v'
Mat, Reference = pretraitementMatrice([FR],FR.keys(),['R','v'])

centres = initialisation_centres (type_clustering, Mat, Reference)
print(centres.shape)
if (centres.shape==(2,Mat.shape[2])):
    print ('Test 1 initialisation_centres ok : La matrice de sortie a la bonne forme')
else:
    print ('Test 1 initialisation_centres pas ok !! La matrice de sortie n a pas la bonne forme')

#test 2 et 3: existence du centre dans la matrice de depart
ind_centre0 = [index for index,row in enumerate(Mat[0,:,:]) if (row==centres[0,:]).all() ]
if(ind_centre0!=[] and (Mat[0,ind_centre0[0],:]==centres[0,:]).all()):
    print('Test 2 initialisation_centres ok : Le centre 0 appartient bien a la matrice de depart')
else:
    print('Test 2 initialisation_centres pas ok !! Le centre 0 n appartient pas a la matrice de depart')

ind_centre1 = [index for index,row in enumerate(Mat[0,:,:]) if (row==centres[1,:]).all() ]
if(ind_centre1!=[] and (Mat[0,ind_centre1[0],:]==centres[1,:]).all()):
    print('Test 3 initialisation_centres ok : Le centre 1 appartient bien a la matrice de depart')
else:
    print('Test 3 initialisation_centres pas ok !! Le centre 1 n appartient pas a la matrice de depart')


########## Tests de SupprimerCartesVides ##################
map_file='../maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
maps_JAP = load_maps(map_file)

map_file='../maps/BREF80_l_conv1_35maps_th0.500000.pkl'
maps_FR = load_maps(map_file)

maps_entry_list = [maps_FR, maps_JAP]

liste = strategie_trois_l1(maps_entry_list, 559)
if (liste==[62]):
    print 'Test 1 ok : sors bien la bonne liste d elements vides'
else :
    print 'Test 1 pas ok !! Ne sors pas la bonne liste d elements vides'

