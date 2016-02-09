from phonemesAnalysis.utiles import *
from phonemesAnalysis.analyse import *
from phonemesAnalysis.featuresGeneration import *

#Script faisant l'histogramme des coefficients suite a une transformation fbank
# un histogramme par parametre ici seulement 15 s'affiche, voir documentation

fft_span = 0.02
hop_span = 0.01
n_mels = 40
fmin = 50
fmax = 8000
nb_classes = 3
path = "./data/Bref80_L4/Bref80_L4M01.wav"
path_aligned = "./data/Bref80_L4/Bref80_L4M01.txt"
path_dict = "./data/classement"

dict = getPhonemeDict(path_dict)

X = fbank(path,fft_span,hop_span,n_mels,fmin,fmax)
Y = getY(X,path_aligned,hop_span=hop_span)
Y_v_non_v = getY_v_non_v(Y,dict,1)

CoeffsHistogrammes(X , 20 , Y_v_non_v ,0 , 15)
