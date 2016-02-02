""" Create reference phoneme for the samples."""

import numpy as np

def drange(start, stop, step):
    """ range() function for float number """

    r = start
    while r < stop:
        yield r
        r += step


def include_period(p1,p2):
    """Function to decide the overlap status of two time periods.

    Args:
        p1:A time period with format like [float,float]
        p2:A time period with format like [float,float]

    Returns:
        int: 1 if p1 completely includes p2
             2 if p1 partially includes p2
             3 if p1 isn't overlap with p2

    """
    if((p1[0]<=p2[0]) and (p1[1]>=p2[1])):
        return 1
    elif((p1[0]<=p2[0]) and (p1[1]<p2[1])):
        return 2
    else:
        return 3


def create_reference(X,alignfile):
    """ This fucntion is used to create reference for the datas

    Args:
        X(ndarray):n*40 matrix,each line represent one sample
        alignfile(str): algin filename
    Returns:
        pho(array):n*1 vector, each element represent the reference phoneme for each sample
    """
    n_samples, n_features = X.shape
    #get the time periods
    period = np.loadtxt(alignfile,delimiter=' ',usecols=(0,1))[0:-1]
    # get the phoneme for different time periods
    phoneme = np.loadtxt(alignfile,dtype= str ,delimiter=' ',usecols=[2])[0:-1]
    n_phoneme = len(np.unique(phoneme))

    #define time period for each sample
    cut_period = [[round(start,2),round(end,2)] for start,end in zip(drange(0,(n_samples+1)*0.01,0.01),drange(0.02,(n_samples+2)*0.01,0.01))]

    #get the reference phoneme for each sample
    pho = [None]*n_samples
    i=0;j=0
    while (i<n_samples and j<len(period)-1):
        if(include_period(period[j],cut_period[i])==1):
            pho[i] = phoneme[j]
            i+=1
        elif(include_period(period[j],cut_period[i])==2):
            pho[i] = phoneme[j]#+'+'+phoneme[j+1]
            j+=1
            i+=1
        elif (include_period(period[j],cut_period[i])==3):
            pho[i-1]=phoneme[j]
            j+=1
    while(i<n_samples):
        pho[i]=phoneme[-1]
        i+=1
    return pho