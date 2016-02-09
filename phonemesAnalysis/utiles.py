import numpy as np

# Fichier contenant des fonctions utiles pour l'analyse des resultats

def getPhonemeDict(path):
    """
    :param path: chemin du dictionnaire (qui a une forme bien specifique, voir fichier data/classement). 
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
    
def getY(X,path_aligned,hop_span):
    """
    :param X: chemin du dictionnaire
    :param path_aligned: chemin du fichier d'alignement (forme bien specifique, 'phoneme' temps_debut temps_fin \n 'phonem2' ...)
    :param hop_span: taille de la fenetre (0.01s par exemple)
    :return: Renvoie un vecteur Y faisant la longueur de X et contenant les phonemes correspondant a chaque fenetre
    """
    nb_vectors, nb_features = X.shape
    #Initialisation du tableau contenant les donnees d'alignement
    beginning = []
    end = []
    phonemes = []
    #lecture fichier alignement
    alignement = open(path_aligned)
    lines  = alignement.readlines()
    alignement.close()
    for line in lines:
       decomposed_line = line.split(' ')
       beginning.append(float(decomposed_line[0]))
       end.append(float(decomposed_line[1]))
       ph = decomposed_line[2].split('\n')
       phonemes.append(ph[0])

    #Creation du vecteur Y contenant le phoneme correspondant pour chacun des vecteurs
    phoneme_courant = 0
    Y = []
    for i in range(nb_vectors):
        if (end[phoneme_courant]<hop_span*i and phoneme_courant<len(phonemes)-2): #on enleve le dernier qui ne compte pas
          phoneme_courant = phoneme_courant + 1
          #if not(phoneme_courant==len(phonemes)-2):
        Y.append(phonemes[phoneme_courant])   #On peut faire plus precis sans doute
    #return(Y)
    return(np.array(Y))


def getConsonnes(X,Y,dict):
    """
    :param X: matrice contenant les feature vectors (n_vectors x n_param)
    :param Y: tableau de taille n_vectors (numpy array) contenant les phonemes correspondant a chaque ligne de X. 
    :param dict: le dictionnaire contenant les informations sur les phonemes
    :return: une matrice X ne contenant que les consonnes, et le tableau Y qui corresond
    """
    Y = np.array(Y)
    Y_consonnes_no = np.array([i for i,j in enumerate(Y) if dict[j][0]==0])
    X_consonnes = X[Y_consonnes_no,:]
    Y_consonnes = Y[np.array(Y_consonnes_no)]
    return X_consonnes, Y_consonnes
