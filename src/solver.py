import numpy as np


def ridge_regression_solution(A: np.ndarray, B: np.ndarray, rho: float) -> np.ndarray:
    """
    Solve the Ridge regression problem: X^Ridge = (A^T A + ρI)^(-1) A^T B
    
    Parameters:
    -----------
    A : np.ndarray
        Data matrix
    B : np.ndarray
        Target vector (yk)
    rho : float
        Regularization parameter
    
    Returns:
    --------
    X : np.ndarray
        Optimal coefficients [a1, a2, ..., an, b]
    """
    # X^Ridge = (A^T A + ρI)^(-1) A^T B
    ATA = A.T @ A
    I = np.eye(ATA.shape[0])
    ATB = A.T @ B
    X = np.linalg.solve(ATA + rho * I, ATB)
    
    return X


def predict_polynomial(x: np.ndarray, coeffs: np.ndarray, degree: int) -> np.ndarray:
    """
    Predict y values using polynomial coefficients.
    
    Parameters:
    -----------
    x : np.ndarray
        Input data points
    coeffs : np.ndarray
        Polynomial coefficients [a1, a2, ..., an, b]
    degree : int
        Degree of the polynomial
    
    Returns:
    --------
    y_pred : np.ndarray
        Predicted y values
    """
    y_pred = np.zeros_like(x)
    
    # Add polynomial terms: a1*x + a2*x^2 + ... + an*x^n
    for i in range(degree):
        y_pred += coeffs[i] * (x**(i + 1))
    
    # Add bias term
    y_pred += coeffs[degree]
    
    return y_pred