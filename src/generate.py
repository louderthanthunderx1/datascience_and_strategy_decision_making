import numpy as np
from typing import Tuple

def generate_training_data(k: int, m: int, n_points: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data (xk, yk) from interval [0.1, k] with steps.
    Data is contaminated with zero-mean Gaussian noise.
    
    Parameters:
    -----------
    k : int
        Number of alphabets in surname (k=7)
    m : int
        Number of alphabets in nickname (m=3)
    n_points : int
        Number of data points (default: 30)
    
    Returns:
    --------
    xk : np.ndarray
        Training data x values
    yk : np.ndarray
        Training data y values (with noise)
    """
    # Generate x from interval [0.1, k] with uniform spacing
    xk = np.linspace(0.1, k, n_points)
    
    # Generate true polynomial (using a known polynomial for demonstration)
    # y = a1*x + a2*x^2 + a3*x^3 + ... + b
    # For demonstration, let's use: y = 2*x + 0.5*x^2 - 0.1*x^3 + 1
    true_coeffs = np.array([2.0, 0.5, -0.1, 1.0])  # a1, a2, a3, b
    
    # Calculate true y values
    y_true = (true_coeffs[0] * xk + 
              true_coeffs[1] * xk**2 + 
              true_coeffs[2] * xk**3 + 
              true_coeffs[3])
    
    # Add zero-mean Gaussian noise
    # Noise variance could be related to m, using m/10 as standard deviation
    noise_std = m / 10.0
    noise = np.random.normal(0, noise_std, size=xk.shape)
    yk = y_true + noise
    
    return xk, yk