from sklearn.cluster import KMeans

from process_activation_maps import load_maps
import numpy as np

map_file='maps/BREF80_l_conv1_35maps_th0.500000.pkl'
FR = load_maps(map_file)

map_file='maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
JA = load_maps(map_file)






def pretraitementMatrice (liste_dictionnaires = [], liste_categories = [], liste_phonemes = []):


 tableau = np.array(liste_dictionnaires[0][liste_categories[0]][liste_phonemes[0]])
 nb_exemple,nb_carte,lign,col=tableau.shape

 Mat = []
 Reference = []
 for num in range(nb_carte):
     Matinter = []
     for inddict,dict in enumerate(liste_dictionnaires):
        for indcat,cat in enumerate(liste_categories):
            for indpho,pho in enumerate(liste_phonemes):
                for ex in range(nb_exemple):
                    Matinter.append((dict[cat][pho][ex][num]).flatten())
                    if num == 0:
                        Reference.append([inddict,indcat ,indpho])
     Mat.append(Matinter)

 Reference = np.array(Reference)

 Mat = np.array(Mat)

 return Mat, Reference




Mat, Reference = pretraitementMatrice([FR, JA],FR.keys(),['R'])
clus = KMeans(n_clusters=2, init='k-means++')
resCluster = clus.fit(Mat[0])
Y_kmeans = resCluster.labels_

def ratios ( Y_kmeans , Reference, nb_classes=2):

    Y_kmeans = np.array(Y_kmeans)

    # Cree les listes des indices correpondant a chacune des classes
    classes = []
    for cl in range(nb_classes):
        classes.append(np.array([j for (j , i) in enumerate(Y_kmeans) if i == cl]))
    total =[];
    total_class = [];
    for i in range(len(set(Reference))):
        nb = len([j for j in Reference if j == i])
        total.append(nb)
    for m in range(nb_classes):
        nb_cluster =[]
        for i in range(len(set(Reference))):
            nb= len([j for j in Y_kmeans[classes[m]] if j == i])
            nb_cluster.append(nb)
        total_class.append(nb_cluster)
    print total
    print total_class
    ratio = 100.0*np.array(total_class)/np.array(total)

    return ratio

pixi = ratios(Y_kmeans, Reference[:,0])
print pixi
