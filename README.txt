Description générale:
Ces fichiers ont pour but de réaliser des clustering de phonemes et de fournir des résultats d'analyse. On peut représenter un signal audio de différentes manières, aligner cette transformation le fichier d'alignement correspondant, faire des clustering et analyser la répartion de différentes catégories de phonemes (voisés/non voisés, voyelles/consonnes, occlusives/fricatives/nasales/voyelles/semi-consonnes)
Analyse annexes : clutering de phonemes francais/japonais, représentation en histogrammes des coefficients d'une matrice transformation d'un signal 

Fichiers:
histogrammesCoefficients.py : script affichant des histogrammes de coefficients, après une transformation fbank
clustering_pythonFbank.py : clustering sur une transformation python en fbank d'un signal audio, affichages de resultats
clustering_phonemes_fr_jap : clustering sur les differentes categories de phonemes, analyse de la separation francais/japonais
phonemesAnalysis/
   featuresGeneration.py : fonctions permettant de transformer un signal
   analyse.py : fonctions faisant l'interprétation de résultats
   utiles.py : fonctions utiles pour l'analyse
resultats/
   resultats_clustering_phonemes_fr_jap.txt : resultats du script clustering_phonemes_fr_jap.py


Bibliothèques utilisées:
-librosa pour le traitement du signal
-pywavelet pour la transformée en ondelettes
-scikit-learn pour le clustering
-scipy, numpy pour le traitement de données en python

