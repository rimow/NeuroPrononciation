import numpy as np
import math
import scipy
import wave
import librosa

from matplotlib import pyplot as plt
from numpy.fft import fft
from scipy.io import wavfile

#lecture du fichier wav 
s1, sr = librosa.load('1.wav')
#sr, s1 = wavfile.read('1.wav')
duree = librosa.core.get_duration(y=s1, sr=sr, S=None, n_fft=441, hop_length=221)
Time=np.linspace(0, duree/sr, num=duree)


#application de la fonction melspectrogram puis fft
S=librosa.feature.melspectrogram(y=s1, sr=sr, S=None, n_fft=441, hop_length=221, n_mels=40)
D = scipy.fft(S)

#je garde les valeurs absolues de D
D = np.abs(D)
print(np.abs(D))
np.save('D.npy', D)

#pour dessiner le spectrogramme mais c pas encore bon
#librosa.display.specshow(librosa.logamplitude(np.abs(D)**2,ref_power=np.max), y_axis='log', x_axis='time')
#plt.plot(Time, np.abs(D)**2 )
#plt.title('Power spectrogram')
#plt.colorbar(format='%+2.0f dB')
#plt.tight_layout()
#plt.show()
