import numpy as np
import scipy as sc
import scipy.io.wavfile
import librosa
from librosa import feature
from librosa import filters
from librosa import util
import matplotlib.pyplot as plt

#Renvoie les vecteurs fbank representant le signal
#path : emplacement du fichier
#fft_span : taille de la fenetre pour la transformee de fourrier en seconde
#hop_span : pas entre deux echantillons en seconde
#n_mels : nombre de bandes de frequences mel
#fmin, fmax : frequences minimales et maximales de la decomposition
#return X : matrice representant la decomposition fbank au cours du temps
# (une colonnes = une decomposition pour une periode hop_span, de taille n_mels)
def fbank(path,fft_span,hop_span,n_mels,fmin,fmax):

  #1ere facon d ouvrir un fichier
  #wav_signal = scipy.io.wavfile.read(path)
  #wav = np.array(wav_signal[1])
  #s_rate = wav_signal[0]
  #Deuxieme facon d ouvrir un fichier
  wav, s_rate = librosa.load(path)

  X = feature.melspectrogram(util.normalize(wav), s_rate, S=None, n_fft=int(np.floor(fft_span*s_rate)), hop_length=int(np.floor(hop_span*s_rate)),n_mels=n_mels,fmin=fmin,fmax=fmax)
  # #Verification nombre d'echantillons (un toutes les 10ms)
  # size = X.shape
  # print 'Taille de la matrice de sortie',size
  # print 'Taille d un morceau de signal de 10ms que l on obtient' ,len(wav)/size[1]
  # print 'taille theorique d un morceau de signal',0.01*s_rate
  # print 's_rate',s_rate
  # print 'longueur',wav.shape
  # print wav.shape[0]/s_rate
  return X

# # TEST
# fft_span = 0.02
# hop_span = 0.01
# n_mels = 40
# fmin = 50
# fmax = 8000
#
# path = "/home/guery/Documents/n7/ProjetLong/data/Bref80_L4M01.wav"
# X = fbank(path,fft_span,hop_span,n_mels,fmin,fmax)
# #Plot
# librosa.display.specshow(X,y_axis='mel', fmax=8000, x_axis='time')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.show()
# print 'colonne 20 juste pour voir:', X[20,:]
# print X.shape


############################################################################################################
#Comparaison avec .mat
# mat = scipy.io.loadmat('/home/guery/Documents/n7/ProjetLong/data/Bref80_L4M01.mat')
# data = mat['d1']
# print data.shape
# mean_mat = np.mean(data,2)
# mean_fct = np.mean(X,2)
# print mean_mat-mean_fct

#### OLD but good
#ouverture fichier wav
# wav_signal = scipy.io.wavfile.read("/home/guery/Documents/n7/ProjetLong/data/Bref80_L4M01.wav")
# wav = np.array(wav_signal[1])
# s_rate = wav_signal[0]
# #wav, s_rate = librosa.load("/home/guery/Documents/n7/ProjetLong/data/Bref80_L4M01.wav") #marche bof
#
# #exemple
# #    sp1 = mirspectrum(s, 'Frame', 0.020, 's', 0.010, 's', 'Min', fMin, 'Max', fMax,
# # 'Bands', nbBands, 'Window', 'hamming', 'NormalInput', 'Mel', 'Log');
# #filters.logfrequency(sample_rate, n_fft, n_bins=84, bins_per_octave=12, tuning=0.0, fmin=None, spread=0.125)
#
# #Calcul des vecteurs
# X = feature.melspectrogram(util.normalize(wav), s_rate, S=None, n_fft=0.02*s_rate, hop_length=0.01*s_rate,n_mels=40,fmin=50,fmax=8000)
# #X = feature.melspectrogram(wav, s_rate, S=None, n_fft=0.02*s_rate, hop_length=0.01*s_rate,n_mels=40,fmin=50,fmax=8000)
#
# #Verification nombre d'echantillons (un toutes les 10ms)
# size = X.shape
# print 'Taille de la matrice de sortie',size
# print 'Taille d un morceau de signal de 10ms que l on obtient' ,len(wav)/size[1]
# print 'taille theorique d un morceau de signal',0.01*s_rate
#
# #Plot
# #librosa.display.specshow(librosa.logamplitude(X,ref_power=np.max),y_axis='mel', fmax=8000, x_axis='time')
# librosa.display.specshow(X,y_axis='mel', fmax=8000, x_axis='time')
# #plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.show()
# print X[20,:]


# Comparaison des deux ouvertures de fichier
# wav, s_rate = librosa.load("/home/guery/Documents/n7/ProjetLong/data/Bref80_L4M01.wav") #marche bof
# print 'rate',s_rate
# print 'taille',len(wav)
# print len(wav)/s_rate
# wav_signal = scipy.io.wavfile.read("/home/guery/Documents/n7/ProjetLong/data/Bref80_L4M01.wav")
# wav = np.array(wav_signal[1])
# s_rate = wav_signal[0]
# print 'rate',s_rate
# print 'taille',len(wav)
# print len(wav)/s_rate
