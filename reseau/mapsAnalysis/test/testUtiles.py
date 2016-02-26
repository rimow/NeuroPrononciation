from mapsAnalysis.utiles import *
import numpy as np
###############    Test ratios   ################
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
