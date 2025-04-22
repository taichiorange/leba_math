# modulation and demodulation functions

from modemod.llr import llr_bpsk

# now only support BPSK： 0 to -1, 1 to 1
def pskmod(X,M=2,phaseoffset=0):
    return 2*X-1

# pNoise [in] : power of noise
# Notes: assume the power of signal is 1.
def pskdemod(Y,pNoise,M=2,phaseoffset=0,OutputType="llr"):
    
    if M == 2 and OutputType == "llr":
        retDem = llr_bpsk(Y,pNoise)
    else:
        retDem = 0
    
    return retDem
