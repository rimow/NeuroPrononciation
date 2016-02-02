import pywt
import librosa
import numpy as np
import matplotlib.pyplot as plt

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


