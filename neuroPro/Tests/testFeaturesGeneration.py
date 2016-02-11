from phonemesAnalysis.featuresGeneration import *

#Tests des fonction ci-dessus : la verification s'effectue grace aux spectrogrammes
#Tests
fft_span = 0.02
hop_span = 0.01
n_mels = 40
fmin = 50
fmax = 8000
dt=0.01
dj=0.5
path = "../data/Bref80_L4/Bref80_L4M01.wav"
# Fbank
X = fbank(path,fft_span,hop_span,n_mels,fmin,fmax,affichage=True)
# mfcc
X = mfcc(path, fft_span, hop_span, n_mels,affichage=True)
#FFT
FourierTransform(path, 441,221,fmin, fmax, n_mels,affichage=True)
#WAVELETS
waveletsTransformContinue(path, 'paul', 2, dt, dj, affichageSpectrogram=True)