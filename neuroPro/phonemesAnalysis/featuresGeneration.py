# import pywt
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
from Erreurs import initialisationError
from phonemesAnalysis.analyse import *
from phonemesAnalysis.utiles import *
# import mlpy.wavelet as wave

# Fichier contenant les fonctions d'extraction de parametres a partir de signaux
# Specification pour toutes les fonctions:
#    - path des parametres doit etre valide et etre le nom d'un fichier audio
#    - la taille des fenetres doit etre inferieur a la duree des signaux

##########################################################################################################################
############################################ FOURIER TRANSFORM ###########################################################
##########################################################################################################################
def FourierTransform(signal_path, n_fft, hop_length,fmin, fmax, n_mels,affichage=False):

    '''
     Fonction de generation des parametres de fourier
    :param signal_path: C'est le chemin vers le fichier audio a traiter
    :param n_fft: La taille de la fenetre
    :param hop_length: La fenetre glissante glisse d'une periode de hop_length
    :param fmin: frequence minimale
    :param fmax: frequence maximale
    :param nBands: nombre de bandes
    :param affichage: True si on veut afficher le spectrogramme
    :return: La matrice D dont les lignes sont des durees de temps de la fenetre et les colonnes contiennent les parametres
    '''

    #S=librosa.feature.melspectrogram(y=s1, sr=sr, S=None, n_fft=441, hop_length=221, n_mels=40)
    #D = scipy.fft(S)
    signal, sampling_rate = librosa.load(signal_path) #load du fichier audio
    D=librosa.feature.melspectrogram(y=signal, sr=sampling_rate, S=None, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    #D = np.abs(D).transpose()
    D = np.log(D)
    if affichage:
      afficherSpec(D,sampling_rate,hop_length)
    D=D.transpose()
    return D;

#Exemple de fonctionnement : Avec une fenetre de 20ms et un glissement de 10ms
#signal, sampling_rate = librosa.load('1.wav')
#FourierTransform('1.wav', int(0.02*sampling_rate), int(0.01*sampling_rate))

##########################################################################################################################




##########################################################################################################################
############################################ WAVELET TRANSFORM ###########################################################
##########################################################################################################################
def waveletsTransformContinue(signalPath, wf, wf_param, dt, dj, affichageSpectrogram):
    '''
    Calcule la transformee en ondelettes continue du signal
    :param signalPath: Le chemin du signal audio
    :param wf: La fonction de l'ondelette ('morlet', 'paul', 'dog')
    :param wf_param: Parametre de la l'ondelette (8 pour morlet, 2 pour dog et paul)
    :param dt: Pas (10ms par exemple)
    :param dj: Resolution de l'echelle (plus dj est petit plus la resolution est fine)
    :return: la transformee en ondelettes continue du signal, matrice 40*len(signal)
    '''

    # Load the wav file, y is the data and sr the sampling frequency
    signal, fe = librosa.load(signalPath)

    scales = wave.autoscales(len(signal), dt=dt, dj=dj, wf=wf, p=wf_param)
    spec = wave.cwt(signal, dt=dt, scales=scales, wf=wf, p=wf_param)
    spec= np.abs(spec)
    wvtransform=spec.transpose()
    wvtransform= moyennerMatrice(wvtransform) #A decommenter si l'on veut avoir une matrice 40*len(signal)
    if affichageSpectrogram:
        afficherSpec(wvtransform,fe,dt)
    return wvtransform

## AMELIORATION RESULTATS C.W.T
def moyennerMatrice(x):
    '''
    Effectue la moyenne sur les lignes suivant des fenetres de 20ms avec un saut de 10ms
    :param x: Matrice resultante de la transformee en ondelettes continue, 40*len(signal)
    :return: Matrice 3449*40
    '''
    out=[]
    y=np.array(x)
    for i in range(0,len(x)):
        if i % 221 == 0:
            sousMatrice = np.array(y[i:i+441,:])
            moyenne = sousMatrice.mean(0)
            out.append(moyenne)
    out=np.array(out)
    return out


##########################################################################################################################


##########################################################################################################################
############################################### MFCC TRANSFORM ###########################################################
##########################################################################################################################
def mfcc(path, taille_fenetre, hop_span, nb_mel,affichage=False):
    '''
    :genere les coefficients cepstraux du fichier son, en utilisant une fenetre glissante
    :param path: (string) chemin du fichier son sur la machine utilisateur
    :param taille_fenetre: (secondes) taille de la fenetre glissante : extraction des parametres pour les fenetres de cette taille
    :param hop_span: (secondes)  deplacement de la fenetre
    :param nb_mel: (int) nombre de coefficients a generer
    :param affichage: True si on veut afficher le spectrogramme
    :return: matrice (liste de tableaux) nb_fenetres*nb_mel : les coefficients pour chaque fenetre
    '''

    #acquisition du signal avec le taux d'echantillonage par defaut (22050)
    son, sr  = librosa.core.load(path)
    duree = librosa.core.get_duration(son)
    #normalisation du signal
    #son_normalized = librosa.util.normalize(son)

    #exceptions sur les parametres de la fonction

    try:
        taille_fenetre<duree
    except initialisationError:
        print "la fenetre glissante doit etre plus petite que la duree de l'enregistrement"

    try:
        hop_span<duree
    except initialisationError:
        print "la duree du hop_lenght doit etre plus petite que la duree de l'enregistrement"



    #calcul de la mfcc pour les deux sons
    son_mfcc  = librosa.feature.mfcc(son,sr,None,nb_mel, hop_length = int(np.floor(hop_span*sr)), n_fft=int(np.floor(taille_fenetre*sr)))

    # #enregistrement de la matrice sous forme numpyArray avec une taille sr
    # son2 =  numpy.asarray(son_mfcc)
    # numpy.save("data/mfcc" , numpy.transpose(son2))

    # #affichage des matrices
    # plt.figure(0)
    # librosa.display.specshow(son2, sr, overlapping, x_axis='frames', y_axis='log', n_xticks = 20, n_yticks = 20, fmin = 50, fmax = 1000)
    # plt.savefig("mfcc.jpg")
    # plt.title('MFCC')
    # plt.show()
    if affichage:
      afficherSpec(son_mfcc,sr,hop_span)

    return  np.transpose(son_mfcc)

##########################################################################################################################



##########################################################################################################################
############################################## FBANK TRANSFORM ###########################################################
##########################################################################################################################
def fbank(path, fft_span, hop_span, n_mels, fmin, fmax,affichage=False):
    """
    :param path: emplacement du fichier
    :param fft_span: taille de la fenetre pour la transformee de fourrier en seconde
    :param hop_span: pas entre deux echantillons en seconde
    :param n_mels: nombre de bandes de frequences mel
    :param fmin: frequence minimale de la decomposition
    :param fmax: frequence maximale de la decomposition
    :param affichage: True si on veut afficher le spectrogramme
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
    if affichage:
      afficherSpec(X,s_rate,hop_span)
    return np.transpose(X)

#fBank en prenant plusieurs fichiers en entree
def fbankPlus(paths_wav,paths_aligned,fft_span,hop_span,n_mels,fmin,fmax):
  """
    :param paths_wav: tableau des chemins des fichiers sons
    :param paths_aligned: tableau des chemins des fichiers d'alignement
    :param fft_span: fenetre pour la fft
    :param hop_span: pas d'une fenetre a une autre
    :param n_mels: nombre de plage de mels
    :param fmin: frequence miniamale
    :param fmax: frequence maximale
    :return: X (les vecteurs representants le signal, nb_vectors x nb_features)
             et Y (phoneme correspondant a chaque vecteur)
  """
  X = []
  Y = []
  for path,path_a in zip(paths_wav,paths_aligned):
    x = fbank(path,fft_span,hop_span,n_mels,fmin,fmax)
    X.append(x)
    Y.append(getY(x,path_a,hop_span))
  return np.concatenate(np.array(X)),np.concatenate(np.array(Y))

def afficherSpec(X,s_rate,hop_span):
    """
    :param X: matrice dont on veut le spectrogramme
    :param s_rate: frequence d echantillonnage
    :param hop_span: pas d'une fenetre a une autre
    :return: affiche une spectrogram avec librosa.specshow
    """
    plt.figure()
    plt.title('Spectrogrammes : librosa.specshow')
    librosa.display.specshow(X,y_axis='mel', fmax=8000, x_axis='time',sr=s_rate,hop_length=int(np.floor(hop_span * s_rate)))
    plt.colorbar(format='%+2.0f dB')
    plt.show()

##########################################################################################################################


##########################################################################################################################
############################################## TESTS DES TRANSFORMATIONS #################################################
##########################################################################################################################

#Tests des fonction ci-dessus : la verification s'effectue grace aux spectrogrammes
# #Tests
# fft_span = 0.02
# hop_span = 0.01
# n_mels = 40
# fmin = 50
# fmax = 8000
# dt=0.01
# dj=0.5
# path = "./data/Bref80_L4/Bref80_L4M01.wav"
# # Fbank
# X = fbank(path,fft_span,hop_span,n_mels,fmin,fmax,affichage=True)
# # mfcc
# X = mfcc(path, fft_span, hop_span, n_mels,affichage=True)
# #FFT
# FourierTransform(path, 441,221,fmin, fmax, n_mels,affichage=True)
# #WAVELETS
# waveletsTransformContinue(path, 'paul', 2, dt, dj, affichageSpectrogram=True)
