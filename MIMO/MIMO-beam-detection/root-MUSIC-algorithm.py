import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

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

def root_music(R, num_sources, d=0.5, wavelength=1.0):
    """
    Root-MUSIC DOA estimation (for ULA).
    
    Parameters:
        R: sample covariance matrix of the received signals (num_sensors x num_sensors)
        num_sources: number of signals (number of DOAs to be estimated)
        d: sensor spacing (in wavelengths, default 0.5)
        wavelength: signal wavelength (default 1.0)
    
    Returns:
        doa_estimates_deg: estimated DOAs (in degrees, sorted in ascending order)
    """
    num_sensors = R.shape[0]
    
    # Perform eigenvalue decomposition of the covariance matrix
    eigvals, eigvecs = eigh(R)
    
    # Select the noise subspace: use the (num_sensors - num_sources) eigenvectors corresponding to the smallest eigenvalues
    En = eigvecs[:, :num_sensors - num_sources]
    
    # Construct the noise subspace projection matrix
    Pn = En @ En.conj().T  # Size: (num_sensors x num_sensors)
    
    # Extract polynomial coefficients using the Toeplitz structure:
    # For a ULA, each diagonal of Pn should theoretically be equal,
    # Here, sum the elements on each diagonal to obtain the coefficients c[k] (k from -M+1 to M-1)
    c = np.array([np.sum(np.diag(Pn, k)) for k in range(-num_sensors+1, num_sensors)])
    
    # Normalize: set the coefficient for k=0 (the main diagonal) to 1, which does not change the root locations
    c = c / c[num_sensors - 1]
    
    # Construct the polynomial coefficients; note that np.roots requires the coefficients in descending order
    poly_coeffs = c[::-1]
    
    # Solve for all roots of the polynomial
    roots_all = np.roots(poly_coeffs)
    
    # Consider only the roots inside the unit circle (theoretically, the signal-related roots should lie near the unit circle)
    roots_inside = roots_all[np.abs(roots_all) < 1]
    
    # Sort the roots by their distance from the unit circle and select the num_sources roots closest to the unit circle
    distances = np.abs(np.abs(roots_inside) - 1)
    sorted_indices = np.argsort(distances)
    selected_roots = roots_inside[sorted_indices][:num_sources]
    
    # According to theory, the phase of the root satisfies: angle(z) = -2π*d*sin(θ)/wavelength
    # beta = 2π*d/wavelength
    beta = 2 * np.pi * d / wavelength
    phi = np.angle(selected_roots)
    
    doa_estimates_rad = np.arcsin(phi / beta)
    doa_estimates_deg = np.rad2deg(doa_estimates_rad)
    
    return np.sort(doa_estimates_deg),roots_all

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
    
    # Estimate the sample covariance matrix
    R = Y @ Y.conj().T / num_snapshots
    
    # Estimate the DOAs using the Root-MUSIC algorithm
    num_sources = len(doa_true)
    doa_est,roots_all = root_music(R, num_sources, d, wavelength)
    
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
