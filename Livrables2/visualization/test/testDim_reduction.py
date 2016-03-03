from visualization.dim_reduction import *

###################Test dim_reduction.dim_reduction_PCA##############
#Test case 1
X=[[0,1,1],[1,0,1],[1,2,1],[2,1,4]]
X_red=dim_reduction_PCA(X,2)
if X_red.shape==(4,2):
    print "Test Successes"
else:
    print "Test Fails"

#Test case 2
X=[[0,1,1],[1,0,1],[1,2,1],[2,1,4]]
X_red=dim_reduction_PCA(X,1)
if X_red.shape==(4,1):
    print "Test Successes"
else:
    print "Test Fails"

#Test case 3
X=[]
X_red=dim_reduction_PCA(X,1)
if not X:
    print "Test Successes"
else:
    print "Test Fails"

###################Test dim_reduction.dim_reduction_LDA##############
#Test case 1

X = [[0,1,1],[1,0,1],[1,2,1],[2,1,4],[1,5,6],[0,1,3]]
Y = [0,0,1,2,3,2]
X_red=dim_reduction_LDA(X,Y,2)
if X_red.shape==(6,2):
    print "Test Successes"
else:
    print "Test Fails"

#Test case 2
X=[[0,1,1],[1,0,1],[1,2,1],[2,1,4]]
Y = [0,0,1,2]
X_red=dim_reduction_LDA(X,Y,1)
if X_red.shape==(4,1):
    print "Test Successes"
else:
    print "Test Fails"

#Test case 3
X=[]
Y=[]
X_red=dim_reduction_LDA(X,Y,1)
if not X:
    print "Test Successes"
else:
    print "Test Fails"
