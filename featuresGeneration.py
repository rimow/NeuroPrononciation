import pywt
import numpy as np
import scipy as sc
import scipy.io.wavfile
import librosa
from librosa import feature
from librosa import filters
from librosa import util
import matplotlib.pyplot as plt
import math
from numpy import shape

# Fichier contenant les fonctions d'extraction de parametres a partir de signaux


def FourierTransform(signal, sampling_rate, n_fft, hop_length):

    '''
     Fonction de generation des parametres de fourier
    :param signal: C'est le fichier audio a traiter
    :param sampling_rate: La frequence d'echantillonageb faire : signal,sampling_rate = librosa.load(<le signal audio>)
    :param n_fft: La taille de la fenetre
    :param hop_length: La fenetre glissante glisse d'une periode de hop_length
    :return: La matrice D dont les lignes sont des durees de temps de la fenetre et les colonnes contiennent les parametres
    '''



    #S=librosa.feature.melspectrogram(y=s1, sr=sr, S=None, n_fft=441, hop_length=221, n_mels=40)
    #D = scipy.fft(S)
    D=librosa.core.stft(y=signal, n_fft= n_fft, hop_length=hop_length, window=None)
    D = np.abs(D).transpose()
    return D;

#Exemple de fonctionnement : Avec une fenetre de 20ms et un glissement de 10ms
#signal, sampling_rate = librosa.load('1.wav')
#FourierTransform(signal, sampling_rate, int(0.02*sampling_rate), int(0.01*sampling_rate))

def waveletsTransform(audioPath, windowLength, hopLength, fmin,fmax,nBands):
    '''
        Effectue la transformee en ondelettes du signal audio en entree
    :param audioPath: Chemin du signal audio
    :param windowLength: Taille de la fenetre de decoupage (=fonctionEchantillonage*dureeEnSecondes)
    :param hopLength: Saut (=fonctionEchantillonage*dureeEnSecondes)
    :param fmin: frequence minimale
    :param fmax: frequence maximale
    :param nBands: nombre de bandes
    :return: Matrice nBands*xx, ou chaque ligne represente les coefficients pour une fenetre
    :Example: waveletsTransform('1.wav',441 ,221, 50,8000,40)
    '''

    #Load the wav file, y is the data and sr the sampling frequency
    y, sr = librosa.load(audioPath)

    #Fenetres 20 ms et saut de 10ms, 40 bandes
    M=librosa.feature.melspectrogram(y=y, sr=sr, S=None, n_fft=windowLength, hop_length=hopLength, n_mels=nBands, fmin=fmin, fmax=fmax)

    #Transformee en ondelettes sur le signal S
    cA, cD = pywt.dwt(M, 'db2')

    #Utile pour le clustering
    S = np.column_stack((cA,cD))
    S=S.transpose()
    np.save('S.npy',S)


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
        raise Exception
    if overlapping>duree:
        raise Exception

    #calcul de la mfcc pour les deux sons
    son_mfcc  = librosa.feature.mfcc(son,sr,None,nb_mel, hop_length = int(numpy.floor(overlapping*sr)), n_fft=int(numpy.floor(taille_fenetre*sr)))
    son_mfcc_normalized  = librosa.feature.mfcc(son_normalized,sr,None,nb_mel, hop_length = int(numpy.floor(overlapping*sr)), n_fft=int(numpy.floor(taille_fenetre*sr)))
    #print shape(son_mfcc)
    #print shape(son_mfcc_normalized)

    son2 =  numpy.asarray(son_mfcc)

    son3 =  numpy.asarray(son_mfcc_normalized)


    # #affichage des matrices
    # plt.figure(0)
    # librosa.display.specshow(son2, sr, overlapping, x_axis='frames', y_axis='log', n_xticks = 20, n_yticks = 20, fmin = 50, fmax = 1000)
    # plt.title('MFCC')
    # plt.figure(1)
    # librosa.display.specshow(son3, sr, overlapping, x_axis='frames', y_axis='log', n_xticks = 20, n_yticks = 20, fmin = 50, fmax = 1000)
    # plt.title('MFCC normalized')
    # plt.show()

    #enregistrement des matrices sous forme numpyArray avec une taille sr
    numpy.save("mfcc" , numpy.transpose(son2))
    numpy.save("mfcc normalized" , numpy.transpose(son3))
    # plt.savefig("mfcc.png")
    # plt.savefig("mfcc normalized.png")


    return  numpy.transpose(son_mfcc)


def fbank(path, fft_span, hop_span, n_mels, fmin, fmax):
    """

    :param path: emplacement du fichier
    :param fft_span: taille de la fenetre pour la transformee de fourrier en seconde
    :param hop_span: pas entre deux echantillons en seconde
    :param n_mels: nombre de bandes de frequences mel
    :param fmin: frequence minimale de la decomposition
    :param fmax: frequence maximale de la decomposition
    :return: Renvoie les vecteurs fbank representant le signal
             X matrice representant la decomposition fbank au cours du temps (une ligne = une decomposition pour une periode hop_span, de taille n_mels)
    """

    # 1ere facon d ouvrir un fichier
    # wav_signal = scipy.io.wavfile.read(path)
    # wav = np.array(wav_signal[1])
    # s_rate = wav_signal[0]
    # Deuxieme facon d ouvrir un fichier
    wav, s_rate = librosa.load(path)

    X = feature.melspectrogram(util.normalize(wav), s_rate, S=None, n_fft=int(np.floor(fft_span * s_rate)),
                               hop_length=int(np.floor(hop_span * s_rate)), n_mels=n_mels, fmin=fmin, fmax=fmax)
    # #Verification nombre d'echantillons (un toutes les 10ms)
    # size = X.shape
    # print 'Taille de la matrice de sortie',size
    # print 'Taille d un morceau de signal de 10ms que l on obtient' ,len(wav)/size[1]
    # print 'taille theorique d un morceau de signal',0.01*s_rate
    # print 's_rate',s_rate
    # print 'longueur',wav.shape
    # print wav.shape[0]/s_rate
    X = np.log(X)
    return np.transpose(X)

# Exemple
# fft_span = 0.02
# hop_span = 0.01
# n_mels = 40
# fmin = 50
# fmax = 8000
# path = "/home/guery/Documents/n7/ProjetLong/data/Bref80_L4M01.wav"
# X = fbank(path,fft_span,hop_span,n_mels,fmin,fmax)
