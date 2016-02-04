import numpy as np
from matplotlib import pyplot as plt

def printRatios(X,nb_classes,Y_cluster,y_voise_non_voise):

  """
    :param X:matrice contenant les feature vectors (n_vectors x n_param)
    :param nb_classes: le nombre de classes resultant du clustering
    :param Y_cluster: tableau contenat les classes attribuees a chaque fenetre de X
    :param y_voise_non_voise: vecteur contenant les classes : 0 pour non voise, 1 pour voise, 2 pour silence
    :return: Pour chacune des classes, ecris le pourcentage de voyelles, consonnes et silences, et le pourcentage de
            voises, non voises et silences (pourcentage du total des phoneme)
  """
  #Au cas ou :
  Y_cluster = np.array(Y_cluster)

  #Cree les listes des indices correpondant a chacune des classes
  classes = []
  for cl in range(nb_classes):
      classes.append(np.array([j for (j,i) in enumerate(Y_cluster) if i==cl]))


  # print 'Voyelles et consonnes (en pourcentage du total des phonemes) :'
  # y_cons_voy = [] #contient pour chaque phoneme de Y sa classe en tant que voyelle ou consonne ou rien
  # for ph in Y:
  #     y_cons_voy.append(dict[ph][0])
  # y_cons_voy = np.array(y_cons_voy)
  # for cl in range(nb_classes):
  #   p1 = 100.*len([ i for i in y_cons_voy[classes[cl]] if i == 0 ])/len(Y_cluster) # % classe cl et consonne
  #   p2 = 100.*len([ i for i in y_cons_voy[classes[cl]] if i == 1 ])/len(Y_cluster) # % classe cl et voyelle
  #   p3 = 100.*len([ i for i in y_cons_voy[classes[cl]] if i == 2 ])/len(Y_cluster) # % classe cl et silence
  #   print '   Classe ',cl,':'
  #   print '     Consonnes :',p1,'\n     Voyelles :',p2,'\n     Silences :',p3,'\n'


  print 'Voises et non voises (en pourcentage du total des phonemes) :'
  for cl in range(nb_classes):
    p1 = 100.*len([ i for i in y_voise_non_voise[classes[cl]] if i == 0 ])/len(Y_cluster) # % classe cl et non voise
    p2 = 100.*len([ i for i in y_voise_non_voise[classes[cl]] if i == 1 ])/len(Y_cluster) # % classe cl et voise
    p3 = 100.*len([ i for i in y_voise_non_voise[classes[cl]] if i == 2 ])/len(Y_cluster) # % classe cl et silence
    print '   Classe ',cl,':'
    print '     Non voises :',p1,'\n    Voises :',p2,'\n   Silences :',p3,'\n'


def getY_v_non_v(Y,dict):
  """
    :param Y: tableau contenant les phonemes correspondant a chaque ligne de X
    :param dict: le dictionnaire contenant les informations sur les phonemes
    :return: un vecteur contenant, respectivement a chaque phoneme de Y, 0 si non voise, 1 si voise, 2 si silence
  """
  y_voise_non_voise = [] #contient pour chaque phoneme de Y sa classe en tant que voise ou non voise ou rien
  for ph in Y:
      y_voise_non_voise.append(dict[ph][1])
  y_voise_non_voise = np.array(y_voise_non_voise)
  return y_voise_non_voise
  
  
  def CoeffsHistogrammes(X,bins,y_voise_non_voise,ind_min,ind_max):
  """
    :param X: matrice contenant les feature vectors (n_vectors x n_param)
    :param bins: nombre de bandes dans les histogrammes
    :param y_voise_non_voise: vecteur contenant les classes : 0 pour non voise, 1 pour voise, 2 pour silence
    :param ind_min: indice correspondant au premier parametre dont on veut l'histogramme des coefficients
    :param ind_max: indice correspondant au dernier parametre dont on veut l'histogramme des coefficients
    :return: affiche, pour chaque parametre entre min et max, l'histogramme des coefficients pour les phonemes voises et non voises
             et les silences
  """
  X_shape = X.shape
  if ind_max>=ind_min and ind_max>0 and ind_min >=0 and ind_max<X_shape[1]:
     indices_non_voise = [i for (i,j) in enumerate(y_voise_non_voise) if j==0]
     indices_voise = [i for (i,j) in enumerate(y_voise_non_voise) if j==1]
     indices_silence = [i for (i,j) in enumerate(y_voise_non_voise) if j==2]
     for i in range(ind_min,ind_max+1):
       ax = plt.figure()
       bins1 = plt.hist(X[indices_non_voise,i],bins=bins,alpha=0.5,label='non voises')
       plt.legend()
       bins2 = plt.hist(X[indices_voise,i],bins=bins,alpha=0.5,label='voises')
       plt.legend()
       bins3 = plt.hist(X[indices_silence,i],bins=bins,alpha=0.5,label='silences')
       plt.legend()
       plt.title('Parametre (indice) : '+str(i))
       plt.show()
  else:
    print 'Erreur : ind_max doit etre superieur ou egal a ind_min, et ind_max et ind_min doivent etre des indices corrects.'
