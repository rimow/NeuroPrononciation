from sklearn.cluster import KMeans, AgglomerativeClustering
from phonemesAnalysis.analyse import *
from phonemesAnalysis.utiles import *
from phonemesAnalysis.featuresGeneration import *

# Script faisant le clustering (Kmeans, Kmeans supervise, agglomerative clustering) de phonemes prononces par un francais et un japonais
# Pour un phoneme donne, recherche de ceux prononces par le francais et par le japonais, ce qui donne un ensemble a clusteriser (en 2 classes)
# On fait cela pour tous les phonemes
# Ecriture des resultats dans le fichier ./data/japonais_resultats.txt



def getPhoneme(X,Y,ph):
    y_ph = [i for i,j in enumerate(Y) if j==ph]
    return X[y_ph,:]

paths_wav = ['../data/Bref80_L4/Bref80_L4M01.wav',
              '../data/Bref80_L4/Bref80_L4M02.wav']
              #'../data/Bref80_L4/Bref80_L4M03.wav',
              #'../data/Bref80_L4/Bref80_L4M04.wav',
              #'../data/Bref80_L4/Bref80_L4M05.wav',
              #'../data/Bref80_L4/Bref80_L4M06.wav',
              #'../data/Bref80_L4/Bref80_L4M07.wav',
              #'../data/Bref80_L4/Bref80_L4M08.wav',
              #'../data/Bref80_L4/Bref80_L4M09.wav']

paths_aligned = ['../data/Bref80_L4/Bref80_L4M01.txt',
              '../data/Bref80_L4/Bref80_L4M02.txt']
              #'../data/Bref80_L4/Bref80_L4M03.txt',
              # '../data/Bref80_L4/Bref80_L4M04.txt',
              # '../data/Bref80_L4/Bref80_L4M05.txt',
              # '../data/Bref80_L4/Bref80_L4M06.txt',
              # '../data/Bref80_L4/Bref80_L4M07.txt',
              # '../data/Bref80_L4/Bref80_L4M08.txt',
              # '../data/Bref80_L4/Bref80_L4M09.txt']


paths_wav_jap = ['../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_01.wav',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_02.wav',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_03.wav',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_04.wav',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_05.wav',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_06.wav',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_07.wav',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_08.wav',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_09.wav']

paths_aligned_jap = ['../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_01.txt',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_02.txt',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_03.txt',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_04.txt',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_05.txt',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_06.txt',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_07.txt',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_08.txt',
                 '../data/fichiers_AYA_WASAKA_MOTS/AYA-WAKASA-production_phon-imFINAL-MOTS_09.txt']

fft_span = 0.02
hop_span = 0.01
n_mels = 40
fmin = 50
fmax = 8000
nb_classes = 2 # (2 classes pour deux langues)
path_dict = "../data/classement"
f_res = open('../resultats/resultatsClustering/japonais_resultats.txt', 'w')

dict = getPhonemeDict(path_dict)

# On recupere les enregistrements en francais et japonais, transformation en fbank, generation des vecteurs d'alignement
X_jap,Y_jap = fbankPlus(paths_wav_jap,paths_aligned_jap,fft_span,hop_span,n_mels,fmin,fmax)
phonemes_presents = list(set(Y_jap))
X,Y = fbankPlus(paths_wav,paths_aligned,fft_span,hop_span,n_mels,fmin,fmax)

clus1= KMeans(nb_classes)
clus2 = KMeans(nb_classes)
clus3 = AgglomerativeClustering(nb_classes)

cluss = [clus1,clus2,clus3]
nb_cluss = len(cluss)

#pour tous les phonemes presents dans les enregistrements japonais
for ph in phonemes_presents:
  f_res.write(ph+'\n')
  #on recupere le bon phoneme
  X_b_jap = getPhoneme(X_jap,Y_jap,ph)
  X_b_fr = getPhoneme(X,Y,ph)
  X_b = np.concatenate([X_b_jap,X_b_fr])
  Y_b = np.concatenate([np.array(np.ones(X_b_fr.shape[0])),np.array(np.zeros(X_b_jap.shape[0]))])
  #Pour tout les types de clustering
  f_res.write('nb phonemes jap : '+str(X_b_jap.shape[0])+' nb phonemes fr :'+str(X_b_fr.shape[0])+'\n')
  for iclus in range(nb_cluss):
    f_res.write('Methode'+str(iclus)+':\n')
    clus = cluss[iclus]
    if iclus==1:  # le kmeans supervise est place en 2eme position
        means = getMeanVectors(X_b,Y_b)
        centres = np.array([means[0],means[1]])
        clus.set_params(init=centres)
    Y_cluster = clus.fit_predict(X_b)
    # Au cas ou :
    Y_cluster = np.array(Y_cluster)
    # Cree les listes des indices correpondant a chacune des classes
    classes = []
    f_res.write('            francais         japonais\n')
    for cl in range(nb_classes):
        classes.append(np.array([j for (j , i) in enumerate(Y_cluster) if i == cl]))
    nb_b_jap = len([i for i in Y_b if i == 0])
    nb_b = len([i for i in Y_b if i == 1])
    print 'Pourcentage des ',ph,' francais et japonais : :'
    for cl in range(nb_classes):
        p1 = 100. * len([i for i in Y_b[classes[cl]] if i == 0]) / nb_b_jap  # % classe cl et japonais
        p2 = 100. * len([i for i in Y_b[classes[cl]] if i == 1]) / nb_b  # % classe cl et francais
        print '   Classe ' , cl , ':'
        print '      francais :' , p2 , '\n     japonais :' , p1
        f_res.write('classe '+str(cl)+'    '+str(p2)+'    '+str(p1)+'\n')
    f_res.write('\n')

f_res.close()
