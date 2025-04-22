# calculate llr for different modulation.

import numpy as np

# llr = log[p(x=0|Y)/p(x=1|Y)]
# for bpsk: mappting table is 0 to -1, 1 to 1. 
#         Some person may want to use 0 to 1, 1 to -1,
#         the llr calculation should be changed to: 2*np.real(Y)/pNoise
def llr_bpsk(Y,pNoise):
    return -2*np.real(Y)/pNoise