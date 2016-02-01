import pywt
import librosa
import numpy as np
import matplotlib.pyplot as plt

#Load the wav file, y is the data and sr the sampling frequency
y, sr = librosa.load('1.wav')
# print(y)
# print(sr)

#Fenetres 20 ms et saut de 10ms, 40 bandes
S=librosa.feature.melspectrogram(y=y, sr=sr, S=None, n_fft=441, hop_length=221, n_mels=40, fmin=50, fmax=8000)
print(S.shape)



#Transformee en ondelettes sur le signal S
cA, cD = pywt.dwt(S, 'db2')
print(cA)
print(cD)
# print(len(cA[1]))


Time=np.linspace(0, len(y)/sr, num=len(y))

librosa.display.specshow(librosa.logamplitude(np.abs(cA)**2,ref_power=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()


#################################################
############       CLUSTERING    ################
#################################################

