import numpy as np

# Fichier contenant les fonctions permettant de reduire la matrice de parametres en ne choisissant que certains phonemes
def getPhonemeDict(path):
    """
    :param path: chemin du dictionnaire
    :return: un dictionnaire de la forme : {'phoneme1':[1,0], 'phoneme2':[0,1]...}
    La premiere valeur du tableau vaut 1 si le phoneme est une voyelle, 0 si c'est une consonne, 2 si c'est un silence
    La seconde valeur du tableau vaut 1 si le phoneme est voise, 0 si non voise, 2 si c'est un silence
    """
    dict = {}
    file= open(path)
    lines  = file.readlines()
    file.close()
    for line in lines:
        decomposed_line = line.split(' ')
        liste = []
        for i in range(1,len(decomposed_line)):
            #print decomposed_line[i][0], decomposed_line[i]
            liste.append(int(decomposed_line[i]))
        dict[decomposed_line[0]] = liste
    return dict

def getConsonnes(X,Y,path_aligned):
    """
    :param X: matrice contenant les feature vectors (n_vectors x n_param)
    :param Y: tableau (numpy array) contenant le phoneme correspondant a chaque vecteur
    :param path_aligned: chemin du fichier d'alignement
    :return: une matrice X ne contenant que les consonnes, et le tableau Y qui corresond
    """
    dict = getPhonemeDict(path_aligned)
    Y_consonnes_no = np.array([i for i,j in enumerate(Y) if dict[j][0]==0])
    X_consonnes = X[Y_consonnes_no,:]
    Y_consonnes = Y[np.array(Y_consonnes_no)]
    return X_consonnes, Y_consonnes
