import librosa
import math
import numpy
from numpy import shape
import matplotlib.pyplot as plt

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
    #normalisation du signal
    son_normalized = librosa.util.normalize(son)

    #exceptions sur les parametres de la fonction


    if taille_fenetre>duree:
        raise ValueError("la taille de la fenetre est trop grande pour la dureee de l'enregistrement")



    if overlapping>duree:
        raise ValueError("la taille du hop_lenght est trop grande pour la dure de l'enregistrement")


    #calcul de la mfcc pour les deux sons
    son_mfcc  = librosa.feature.mfcc(son,sr,None,nb_mel, hop_length = int(numpy.floor(overlapping*sr)), n_fft=int(numpy.floor(taille_fenetre*sr)))
    son_mfcc_normalized  = librosa.feature.mfcc(son_normalized,sr,None,nb_mel, hop_length = int(numpy.floor(overlapping*sr)), n_fft=int(numpy.floor(taille_fenetre*sr)))
    #print shape(son_mfcc)
    #print shape(son_mfcc_normalized)

    son2 =  numpy.asarray(son_mfcc)

    son3 =  numpy.asarray(son_mfcc_normalized)


    #affichage des matrices
    plt.figure(0)
    librosa.display.specshow(son2, sr, overlapping, x_axis='frames', y_axis='log', n_xticks = 20, n_yticks = 20, fmin = 50, fmax = 1000)
    plt.savefig("mfcc.jpg")
    plt.savefig("mfcc normalized.png")
    plt.title('MFCC')
    plt.figure(1)
    librosa.display.specshow(son3, sr, overlapping, x_axis='frames', y_axis='log', n_xticks = 20, n_yticks = 20, fmin = 50, fmax = 1000)
    plt.title('MFCC normalized')
    plt.show()

    #enregistrement des matrices sous forme numpyArray avec une taille sr
    numpy.save("mfcc" , numpy.transpose(son2))
    numpy.save("mfcc normalized" , numpy.transpose(son3))



    return  numpy.transpose(son_mfcc)


#piou = mfcc("/home/marianne/Developpement/Bref80_L4M01.wav" , 0.02 , 0.01 , 40)
#print "piou", piou
