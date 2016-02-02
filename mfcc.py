import librosa
import math
import numpy
from numpy import shape

##############################################
#genere les coefficients cepstraux du fichier son, en utilisant une fenetre glissante
#parametres:
##path(chaine de caracteres) : chemin du fichier son sur la machine utilisateur
##taille_fenetre(secondes) : taille de la fenetre glissante : extraction des parametres pour les fenetres de cette taille
##overlapping(secondes) : deplacement de la fenetre
##nb_mel(int) : nombre de coefficients a generer
#retour:matrice nb_fenetres*nb_mel : les coefficients pour chaque fenetre
##############################################
def mfcc(path, taille_fenetre, overlapping, nb_mel):
    '''
    :genere les coefficients cepstraux du fichier son, en utilisant une fenetre glissante
    :param path: (string) chemin du fichier son sur la machine utilisateur
    :param taille_fenetre: (secondes) taille de la fenetre glissante : extraction des parametres pour les fenetres de cette taille
    :param overlapping: (secondes)  deplacement de la fenetre
    :param nb_mel: (int) nombre de coefficients a generer
    :return: matrice (liste de tableaux) nb_fenetres*nb_mel : les coefficients pour chaque fenetre
    '''

    #acquisition du signal avec le taux d'echantillonage par defaut (22050)
    son, sr  = librosa.core.load(path)
    duree = librosa.core.get_duration(son)

    #exceptions sur les parametres de la fonction
    if taille_fenetre>duree:
        raise Exception
    if overlapping>duree:
        raise Exception

    #calcul du nombre de fenetres total
    nb_ech_fenetre = sr*taille_fenetre
    nb_fenetres = int(math.floor(duree/overlapping))
    if (nb_fenetres-1)*overlapping+taille_fenetre > duree:
        nb_fenetres = nb_fenetres-1

    #pour chacune des fenetres, calcul et stockage de la mfcc
    son_mfcc = [None]*nb_fenetres
    for i in range(nb_fenetres):
        son_inter = son[i*overlapping:nb_ech_fenetre]
        son_inter2 = librosa.feature.mfcc(son_inter,sr,None,nb_mel)
        son_mfcc[i] = son_inter2.T[0]
    #print shape(son_mfcc)
    #print nb_fenetres, nb_mel
    #print son_mfcc

    #enregistrement de la matrice dans un fichier csv
    #a = numpy.asarray(son_mfcc)
    #numpy.savetxt("generation mfcc.csv" , a , delimiter=",")
    return son_mfcc


#piou = mfcc("/home/marianne/Developpement/Bref80_L4M01.wav" , 0.02 , 0.01 , 40)
#print piou