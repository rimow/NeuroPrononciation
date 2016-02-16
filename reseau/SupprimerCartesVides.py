import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def load_maps(fname):
    '''loads a map dictionary'''
    maps = pickle.load(open(fname, 'rb'))
    return maps

def intersect(a, b):
     return list(set(a) & set(b))

def strategie_une_l1(maps_entry):
    """
    Supprime une carte si elle est vide sur tous les exemples
    :param maps_entry: un ensemble de cartes d'activation: maps_entry = load_maps(map_path)
    :return: la liste des cartes qui sont vides sur tous les exemples
    """

    nb_exemple,nb_carte,ligne,col = np.array(maps_entry['correct_pasOK']['R']).shape
    liste_phonemes=['R','v']
    liste_vides_precedente= []
    #initialisation de liste_vides_precedent
    for indice_carte in range(nb_carte):
        if(maps_entry['correct_pasOK']['R'][0][indice_carte]==np.zeros([ligne,col])).all():
            liste_vides_precedente.append(indice_carte)
    #calcul
    for pho in liste_phonemes:
        for cat in maps_entry.keys():
            for indice_exemple in range(nb_exemple):
                liste_vides_exemple= []
                for indice_carte in range(nb_carte):
                    if(maps_entry[cat][pho][indice_exemple][indice_carte]==np.zeros([ligne,col])).all():
                        liste_vides_exemple.append(indice_carte)
                liste_vides_precedente = intersect(liste_vides_precedente, liste_vides_exemple)
    return liste_vides_precedente


def strategie_deux_l1(maps_entry):
    '''
    Supprime une carte si elle est vide sur un seul exemple
    :param maps_entry: un ensemble de cartes d'activation: maps_entry = load_maps(map_path)
    :return:la liste des cartes qui sont vides ne serait ce que sur un seul exemple
    '''

    nb_exemples,nb_cartes,ligne,col = np.array(maps_entry['correct_pasOK']['R']).shape
    liste_phonemes=['R','v']
    liste_vides= []
    liste = []
    for pho in liste_phonemes:
        for cat in maps_entry.keys():
            for indice_exemple in range(nb_exemples):
                for indice_carte in range(nb_cartes):
                    if(maps_entry[cat][pho][indice_exemple][indice_carte]==np.zeros([ligne,col])).all():
                        liste_vides.append(indice_carte)
    return set(liste_vides)


def strategie_trois_l1(maps_entry, seuil):
    '''
   Supprime une carte des lors qu elle est vide sur plus de seuil exemples
   :param maps_entry:un ensemble de cartes d'activation: maps_entry = load_maps(map_path)
   :param seuil: si une carte est vide sur plus de 'seuil' exemples on la supprime
   :return: liste des cartes qui sont vides sur plus de 'seuil' exemples
    '''
    nb_exemples, nb_cartes, ligne, col = np.array(maps_L1['correct_pasOK']['R']).shape
    liste_phonemes = ['R', 'v']
    liste_vides = []
    liste = []
    for pho in liste_phonemes:
        for cat in maps_entry.keys():
            for indice_exemple in range(nb_exemples):
                for indice_carte in range(nb_cartes):
                    if (maps_entry[cat][pho][indice_exemple][indice_carte] == np.zeros([ligne, col])).all():
                        liste_vides.append(indice_carte)
    liste_resultat = [i for i in liste_vides if liste_vides.count(i) > seuil]
    return set(liste_resultat)


#exemple de test
if __name__ == '__main__':

    map_file='maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
    maps_L1 = load_maps(map_file)


    print(strategie_une_l1(maps_L1))

    liste = strategie_deux_l1(maps_L1)
    print(len(liste))

    liste = strategie_trois_l1(maps_L1, 250)
    print(liste)
    print(len(liste))



