#!/usr/bin/env python3

# 这个执行的速度比较慢，没有并行。如果为了提高速度，请参考类似名称的文件，是多进程并行执行的版本。

import numpy as np
import matplotlib.pyplot as plt
import time

from modemod.modem import pskmod
from modemod.modem import pskdemod
from channel.awgn import awgn
from polarcode.polarconstruct import polarConBhatBound
from polarcode.polarencoder import polarEncode
from polarcode.polarencoder import polarCreateMatrixG
from polarcode.polardecoder import polarScDecode

n = 9
N = 2**n
K = 232

# mothercode construction
design_SNR  = 5.0
NSample = int(1e+3)
realSnr = np.arange(-2,15,1)

# construct polar code according to the design SNR.
reliabilities, frozen, FERest = polarConBhatBound(N,K,(10**(design_SNR/10))*K/N)
print(frozen)

# G matrix, only demo, no use.
G = polarCreateMatrixG(N)

start_time = time.time()  # Record the start time

ber = np.zeros(len(realSnr))
for i in range(len(realSnr)):
    snr = realSnr[i]
    pNoise = np.power(10,-snr/10)

    errN = 0
    for j in range(1,NSample):
        # create message data
        message = np.random.randint(2, size=K)

        # polar encode
        xEncoded = polarEncode(message,N,frozen)
        xMod = pskmod(xEncoded)
        txData = awgn(xMod,pNoise)
        pcLlr = pskdemod(txData,pNoise,M=2,OutputType="llr")

        messageReceived = polarScDecode(pcLlr,N,frozen)
        errN += np.sum(messageReceived != message)

    ber[i] = errN/(NSample*K)
    print(ber[i])

end_time = time.time()  # Record the end time

print(f"Execution time: {end_time - start_time} seconds")

plt.semilogy(realSnr,ber)
plt.grid()
plt.ylim(1e-6, 0.6)
plt.show(block=True)
