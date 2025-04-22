import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from beam.MUSIC import rootMUSIC


def simulate_ula_signals(num_sensors, num_snapshots, doa_deg, d=0.5, wavelength=1.0, SNR_dB=20):
    """
    Parameters:
        num_sensors: number of sensors (antennas)
        num_snapshots: number of samples
        doa_deg: actual angles (degrees), list or numpy array
        d: distance between adjacent antennas (unit: wavelength, default 0.5, half wavelength)
        wavelength: wavelength (default is 1.0)
        SNR_dB: SNR in dB
    
    Returns:
        Y: array of received signals (num_sensors x num_snapshots)
    """
    doa_rad = np.deg2rad(doa_deg)
    num_sources = len(doa_deg)
    sensor_idx = np.arange(num_sensors)
    
    # Construct the array manifold matrix
    # Steering vector: a(θ) = [1, exp(j*2π*d*sinθ), exp(j*2π*2*d*sinθ), ...]^T
    A = np.exp(1j * 2 * np.pi * d * np.sin(doa_rad)[np.newaxis, :] * sensor_idx[:, np.newaxis] / wavelength)
    
    # Generate signals: complex Gaussian random signals (normalized to unit power)
    S = (np.random.randn(num_sources, num_snapshots) + 1j * np.random.randn(num_sources, num_snapshots)) / np.sqrt(2)
    
    # Received signals without noise
    Y_clean = A @ S  # num_sensors x num_snapshots
    
    # Add noise
    signal_power = np.mean(np.abs(Y_clean)**2)
    noise_power = signal_power / (10**(SNR_dB/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(num_sensors, num_snapshots) + 1j * np.random.randn(num_sensors, num_snapshots))
    
    Y = Y_clean + noise
    return Y

if __name__ == "__main__":
    # Simulation parameters
    num_sensors = 8           # Number of array sensors
    num_snapshots = 1000      # Number of snapshots; recommended to increase to improve covariance matrix estimation accuracy
    doa_true = [-20, 20, 30]   # True DOAs (in degrees)
    SNR_dB = 20               # SNR in dB
    d = 0.5                   # Sensor spacing (in wavelengths)
    wavelength = 1.0          # Wavelength
    
    # Generate simulated signal data (with specified beam/directions)
    Y = simulate_ula_signals(num_sensors, num_snapshots, doa_true, d, wavelength, SNR_dB)
    
    # Estimate the DOAs using the Root-MUSIC algorithm
    num_sources = len(doa_true)
    doa_est,roots_all = rootMUSIC(Y, num_sources, d, wavelength)
    
    print("True DOAs (degrees):", doa_true)
    print("Estimated DOAs (degrees):", doa_est)
    
    plt.figure(figsize=(6,6))
    theta = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', label='unit circle')
    plt.scatter(np.real(roots_all), np.imag(roots_all), marker='o', color='b', label='roots of polynomial')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Root-MUSIC Root Distribution')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()
