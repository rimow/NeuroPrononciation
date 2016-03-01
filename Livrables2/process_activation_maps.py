import cPickle as pickle

import matplotlib.pyplot as plt
from theano.compile import shape


def load_maps(fname):
    '''loads a map dictionary'''
    maps = pickle.load(open(fname, 'rb'))
    return maps

def plot_128maps(maps):
    '''pour la L1 et MP1'''
    f, ax = plt.subplots(8, 16, sharex=True, sharey=True)
    for i in range(8):
        for j in range(16):
            ax[i, j].imshow(maps[i*8+j], aspect='auto')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

def plot_256maps(maps):
    '''pour la L2 et MP2'''
    f, ax = plt.subplots(8, 32, sharex=True, sharey=True)
    for i in range(8):
        for j in range(32):
            ax[i, j].imshow(maps[i*8+j], aspect='auto')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

if __name__ == '__main__':

    map_file='maps/BREF80_l_conv1_35maps_th0.500000.pkl'
    maps_L1 = load_maps(map_file)
    print "map 1:", shape(maps_L1)

    print 'maps_L1.keys()', maps_L1.keys()

    map_file='maps/BREF80_l_conv2_35maps_th0.500000.pkl'
    maps_L2 = load_maps(map_file)

    map_file='maps/BREF80_l_mp2_35maps_th0.500000.pkl'
    maps_MP2 = load_maps(map_file)

    dense_file='maps/BREF80_l_dense1_35maps_th0.500000.pkl'
    act_DENSE = load_maps(dense_file)

    # les cles du dictionnaire maps:
    # print maps_L1.keys()

    # exemple pour dessiner des cartes de la couche de conv 1
    phone='R'
    indice_exemple = 2
    map = maps_L1['correct_pasOK'][phone][indice_exemple]
    plot_128maps(map)

    # exemple pour dessiner des cartes de la couche de conv 2
    map = maps_L2['correct_pasOK'][phone][indice_exemple]
    print 'shape map', map.shape
    plot_256maps(map)


    # exemple pour dessiner les activations de la couche dense
    act = act_DENSE['correct_pasOK'][phone][indice_exemple]
    print act
    plt.figure()
    plt.plot(act)


    plt.show()


