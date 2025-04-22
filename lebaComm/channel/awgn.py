# channel model
#   1) AWGN : Additive white Gaussian noise

import numpy as np

def awgn(X,pNoise):
    if np.iscomplexobj(X):  # for complex input
        noise = np.sqrt(pNoise / 2) * (np.random.normal(0, 1, size=X.shape)
                                       + np.random.normal(0, 1, size=X.shape)*1j)
    else:   # for real input
        noise = np.sqrt(pNoise) * np.random.normal(0, 1, size=X.size)
    
    return X+noise
