import numpy as np


def build_data_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Build the data matrix A for polynomial regression.
    
    For degree n, the matrix A has columns: [x, x^2, x^3, ..., x^n, 1]
    
    Parameters:
    -----------
    x : np.ndarray
        Input data points
    degree : int
        Degree of the polynomial
    
    Returns:
    --------
    A : np.ndarray
        Data matrix of shape (n_points, degree+1)
    """
    n_points = len(x)
    A = np.zeros((n_points, degree + 1))
    
    # Fill columns: x, x^2, x^3, ..., x^degree
    for i in range(degree):
        A[:, i] = x**(i + 1)
    
    # Last column is constant term (bias)
    A[:, degree] = 1.0
    
    return A



def least_squares_solution(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve the least squares problem: X* = (A^T A)^(-1) A^T B = A†B
    
    Parameters:
    -----------
    A : np.ndarray
        Data matrix
    B : np.ndarray
        Target vector (yk)
    
    Returns:
    --------
    X : np.ndarray
        Optimal coefficients [a1, a2, ..., an, b]
    """
    # X* = (A^T A)^(-1) A^T B
    ATA = A.T @ A
    ATB = A.T @ B
    X = np.linalg.solve(ATA, ATB)
    
    return X