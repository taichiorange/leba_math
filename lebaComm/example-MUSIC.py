import numpy as np
import matplotlib.pyplot as plt
from beam.MUSIC import MusicGetSpectrum

# configure
N = 8  # number of antennas
M = 5   # number of beams
k_true = np.array([0.3, 0.4,0.5,0.6,0.7])  # beam angles
SNR = 10  # 信噪比(dB)

# create a manifold vector
def steering_vector(k, N):
    n = np.arange(N)
    return np.exp(1j * 2 * np.pi * k * n)

NSamples = 100

# generate signals
X = np.zeros((N, NSamples), dtype=complex)
for i in range(M):
    a_k = steering_vector(k_true[i], N)
    s = np.exp(1j * 2 * np.pi * np.random.rand(NSamples))  # random signals
    X += np.outer(a_k, s)

# add noise
noise = (np.random.randn(N, NSamples) + 1j * np.random.randn(N, NSamples)) / np.sqrt(2)
X += noise * (10 ** (-SNR / 20))

k_scan,P_music = MusicGetSpectrum(X,N,M,500)

# 绘制 MUSIC 频谱
plt.plot(k_scan, P_music)
plt.xlabel("wave beam index (normalized) ")
plt.ylabel("MUSIC Spectrum (dB)")
plt.title("MUSIC High-Resolution f-k Spectrum")
plt.grid()
plt.xticks(np.arange(0, 1, 0.1))
plt.show()
