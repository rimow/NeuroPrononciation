from mapsAnalysis.utiles import *
import numpy as np
from process_activation_maps import load_maps
from mapsAnalysis.SupprimerCartesVides import *


########################################################################################################################
################################################### TEST RATIOS ########################################################
########################################################################################################################

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

########################################################################################################################
############################################## TEST BIENCLUSTERISE #####################################################
########################################################################################################################

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
fichier = "utilitaireTestUtiles.csv"
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


########################################################################################################################
######################################## TEST INITIALISATION CENTRES ###################################################
########################################################################################################################

#test 1 : test du format de sortie
print "\n###################################\n"
print "tests initialisation_centres\n"
print "###################################\n"
map_file_FR = '../../maps/BREF80_l_conv1_35maps_th0.500000.pkl'
FR= load_maps(map_file_FR)

type_clustering = 'R_v'
Mat, Reference = pretraitementMatrice([FR],FR.keys(),['R','v'])

centres = initialisation_centres (type_clustering, Mat, Reference)

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


########################################################################################################################
######################################### TEST SUPPRIMER CARTES VIDES ##################################################
########################################################################################################################

print "\n###################################\n"
print "tests Supprimer cartes vides\n"
print "###################################\n"
map_file='../../maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
maps_JAP = load_maps(map_file)

map_file='../../maps/BREF80_l_conv1_35maps_th0.500000.pkl'
maps_FR = load_maps(map_file)

maps_entry_list = [maps_FR, maps_JAP]

liste = strategie_trois_l1(maps_entry_list, 559)
if (liste==[62]):
    print 'Test 1 ok : sors bien la bonne liste d elements vides'
else :
    print 'Test 1 pas ok !! Ne sors pas la bonne liste d elements vides'

print('\n')

########################################################################################################################
############################################## TEST PRETRAITEMENTMATRICE ###############################################
########################################################################################################################

print "\n###################################\n"
print "Tests pretraitmentMatrice\n"
print "###################################\n"


#Choix de la couche pour le chargement des cartes correspondantes
couche='conv1'

#chargement des dictionnaires
map_file_FR = "../../maps/BREF80_l_" + couche + "_35maps_th0.500000.pkl"
map_file_JA = "../../maps/PHONIM_l_" + couche + "_35maps_th0.001000.pkl"
FR= load_maps(map_file_FR)
JA = load_maps(map_file_JA)


################################################################################
# TEST ENTREES VIDES, TEST D'ERREURS
################################################################################

# Entrees vide
Mat,Ref = pretraitementMatrice([],[],[])
if len(Mat)==0 and len(Ref)==0:
    print ("TESTS ENTREES VIDES : Success")
else:
    print ("TESTS ENTREES VIDES : Fail")
print("\n")


#Test d'erreurs
print ("TESTS D'ERREURS :" + "\n")

Mat,Ref = pretraitementMatrice([FR],[],[])

Mat,Ref = pretraitementMatrice([],[],['R'])

Mat,Ref = pretraitementMatrice([JA],JA.keys(),[])

print("\n")

################################################################################
# TEST UNITAIRES
################################################################################

print('TESTS UNITAIRES : \n')
################################## TEST 1 ######################################
dict=[FR]
cat = FR.keys()
phone = ['R']
Mat,Ref = pretraitementMatrice(dict,cat,phone)


#recuperation des dimensions pour un dictionnaire
tableau = np.array(FR['correct_OK']['R'])
taille=tableau.shape
if couche == 'dense1':
    expectedmatdimensions = (taille[1], len(dict)*len(cat)*len(phone)*taille[0], 1)
else:
    expectedmatdimensions = (taille[1], len(dict)*len(cat)*len(phone)*taille[0], taille[2]*taille[3])
if (expectedmatdimensions == Mat.shape):
    print("Test unitaire 1 : Success, Mat dimensions are correct")
else:
    print("Test unitaire 1 : Fail(Mat dimensions do not correspond)")

print("-------------------")

expectedrefdimensions = (len(dict)*len(cat)*len(phone)*taille[0],3)
if (expectedrefdimensions == Ref.shape):
    print("Test unitaire 1 : Success, Ref dimensions are correct")
else:
    print("Test unitaire 1 : Fail(Ref dimensions do not correspond)")


print("\n")

################################## TEST 2 ######################################

dict=[FR, JA]
cat = FR.keys()
phone = ['v']
Mat,Ref = pretraitementMatrice(dict,cat,phone)


#recuperation des dimensions pour un dictionnaire
tableau = np.array(FR['correct_OK']['v'])
taille=tableau.shape
if couche == 'dense1':
    expectedmatdimensions = (taille[1], len(dict)*len(cat)*len(phone)*taille[0], 1)
else:
    expectedmatdimensions = (taille[1], len(dict)*len(cat)*len(phone)*taille[0], taille[2]*taille[3])
if (expectedmatdimensions == Mat.shape):
    print("Test unitaire 2 : Success, Mat dimensions are correct")
else:
    print("Test unitaire 2 : Fail(Mat dimensions do not correspond)")

print("-------------------")

expectedrefdimensions = (len(dict)*len(cat)*len(phone)*taille[0],3)
if (expectedrefdimensions == Ref.shape):
    print("Test unitaire 2 : Success, Ref dimensions are correct")
else:
    print("Test unitaire 2 : Fail(Ref dimensions do not correspond)")

print("\n")


################################## TEST 3 ######################################

dict=[JA]
cat = JA.keys()
phone = ['R']
Mat,Ref = pretraitementMatrice(dict,cat,phone)


#recuperation des dimensions pour un dictionnaire
tableau = np.array(JA['correct_OK']['R'])
taille=tableau.shape
if couche == 'dense1':
    expectedmatdimensions = (taille[1], len(dict)*len(cat)*len(phone)*taille[0], 1)
else:
    expectedmatdimensions = (taille[1], len(dict)*len(cat)*len(phone)*taille[0], taille[2]*taille[3])
if (expectedmatdimensions == Mat.shape):
    print("Test unitaire 3 : Success, Mat dimensions are correct")
else:
    print("Test unitaire 3 : Fail(Mat dimensions do not correspond)")

print("-------------------")

expectedrefdimensions = (len(dict)*len(cat)*len(phone)*taille[0],3)
if (expectedrefdimensions == Ref.shape):
    print("Test unitaire 3 : Success, Ref dimensions are correct")
else:
    print("Test unitaire 3 : Fail(Ref dimensions do not correspond)")

print("\n")
