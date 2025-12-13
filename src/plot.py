import numpy as np
import matplotlib.pyplot as plt


def plot_results(xk: np.ndarray, yk: np.ndarray, 
                 x_plot: np.ndarray, y_pred: np.ndarray,
                 degree: int, title: str, save_path: str = None):
    """
    Plot the original raw data and the estimated polynomial.
    
    Parameters:
    -----------
    xk : np.ndarray
        Training data x values
    yk : np.ndarray
        Training data y values (noisy)
    x_plot : np.ndarray
        X values for smooth plotting
    y_pred : np.ndarray
        Predicted y values
    degree : int
        Degree of the polynomial
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(xk, yk, color='red', alpha=0.6, label='Noisy Training Data', s=50)
    plt.plot(x_plot, y_pred, 'b-', linewidth=2, label=f'Estimated Polynomial (degree {degree})')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()