"""
#Cree un dictionnaire en lisant le fichier path
# renvoie un dictionnaire de la forme : {'phoneme1':[1,0], ...}
# La premiere valeur du tableau vaut 1 si le phoneme est une voyelle, 0 si c'est une consonne, 2 si c'est un silence
# La seconde valeur du tableau vaut 1 si le phoneme est voise, 0 si non voise, 2 si c'est un silence"""
def getPhonemeDict(path):
    dict = {}
    file= open(path)
    lines  = file.readlines()
    file.close()
    for line in lines:
        decomposed_line = line.split(' ')
        liste = []
        for i in range(1,len(decomposed_line)):
            liste.append(int(decomposed_line[i][0]))
        dict[decomposed_line[0]] = liste
    return dict

#Test
# path = "/home/guery/Documents/n7/ProjetLong/data/classement"
# d = getPhonemeDict(path)
# print d
