import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def load_maps(fname):
    '''loads a map dictionary'''
    maps = pickle.load(open(fname, 'rb'))
    return maps

def intersect(a, b):
     return list(set(a) & set(b))

def strategie_une_l1(maps_entry_liste):
    """
    Supprime une carte si elle est vide sur tous les exemples
    :param maps_entry: un ensemble de cartes d'activation: maps_entry = load_maps(map_path)
    :return: la liste des cartes qui sont vides sur tous les exemples
    """

    nb_exemple,nb_carte,ligne,col = np.array(maps_entry_liste[0]['correct_pasOK']['R']).shape
    liste_phonemes=['R','v']
    liste_vides_precedente= []
    #initialisation de liste_vides_precedent
    for indice_carte in range(nb_carte):
        if(maps_entry_liste[0]['correct_pasOK']['R'][0][indice_carte]==np.zeros([ligne,col])).all():
            liste_vides_precedente.append(indice_carte)
    #calcul
    for maps_entry in maps_entry_liste:
        for pho in liste_phonemes:
            for cat in maps_entry.keys():
                for indice_exemple in range(nb_exemple):
                    liste_vides_exemple= []
                    for indice_carte in range(nb_carte):
                        if(maps_entry[cat][pho][indice_exemple][indice_carte]==np.zeros([ligne,col])).all():
                            liste_vides_exemple.append(indice_carte)
                    liste_vides_precedente = intersect(liste_vides_precedente, liste_vides_exemple)
    return liste_vides_precedente


def strategie_deux_l1(maps_entry_liste):
    '''
    Supprime une carte si elle est vide sur un seul exemple
    :param maps_entry: un ensemble de cartes d'activation: maps_entry = load_maps(map_path)
    :return:la liste des cartes qui sont vides ne serait ce que sur un seul exemple
    '''

    nb_exemples,nb_cartes,ligne,col = np.array(maps_entry_liste[0]['correct_pasOK']['R']).shape
    liste_phonemes=['R','v']
    liste_vides= []
    liste = []
    for maps_entry in maps_entry_liste:
        for pho in liste_phonemes:
            for cat in maps_entry.keys():
                for indice_exemple in range(nb_exemples):
                    for indice_carte in range(nb_cartes):
                        if(maps_entry[cat][pho][indice_exemple][indice_carte]==np.zeros([ligne,col])).all():
                            liste_vides.append(indice_carte)
    return list(set(liste_vides))


def strategie_trois_l1(maps_entry_liste, seuil):
    '''
   Supprime une carte des lors qu elle est vide sur plus de seuil exemples
   :param maps_entry:un ensemble de cartes d'activation: maps_entry = load_maps(map_path)
   :param seuil: si une carte est vide sur plus de 'seuil' exemples on la supprime
   :return: liste des cartes qui sont vides sur plus de 'seuil' exemples
    '''
    liste_vides = []
    liste_phonemes = ['R', 'v']
    for maps_entry in maps_entry_liste:
        nb_exemples, nb_cartes, ligne, col = np.array(maps_entry['correct_pasOK']['R']).shape
        for pho in liste_phonemes:
            for cat in maps_entry.keys():
                for indice_exemple in range(nb_exemples):
                    for indice_carte in range(nb_cartes):
                        if (maps_entry[cat][pho][indice_exemple][indice_carte] == np.zeros([ligne, col])).all():
                            liste_vides.append(indice_carte)
    liste_resultat = [i for i in liste_vides if liste_vides.count(i) > seuil]
    return list(set(liste_resultat))


#exemple de test
if __name__ == '__main__':

    map_file='maps/PHONIM_l_conv1_35maps_th0.001000.pkl'
    maps_JAP = load_maps(map_file)

    map_file='maps/BREF80_l_conv1_35maps_th0.500000.pkl'
    maps_FR = load_maps(map_file)

    maps_entry_list = [maps_FR, maps_JAP]



    liste = strategie_trois_l1(maps_entry_list, 559)
    print(liste)

    liste=strategie_une_l1(maps_entry_list)
    print(liste)

    liste = strategie_deux_l1(maps_entry_list)
    print(liste)



