"""
Polynomial Regression Assignment
Surname: sukchok (k=7)
Nickname: lok (m=3)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.generate import generate_training_data
from src.metrics import build_data_matrix, least_squares_solution
from src.solver import ridge_regression_solution, predict_polynomial
from src.plot import plot_results


def main():
    """
    Main function to execute all tasks.
    """
    # Step 1: Set up parameters
    k = 7  # Number of alphabets in surname "sukchok"
    m = 3  # Number of alphabets in nickname "lok"
    n_points = 30  # Number of training points
    
    print("=" * 60)
    print("Polynomial Regression Assignment")
    print(f"Surname: sukchok (k={k})")
    print(f"Nickname: lok (m={m})")
    print("=" * 60)
    
    # Step 2: Generate training data
    print("\nStep 1: Generating training data...")
    np.random.seed(42)  # For reproducibility
    xk, yk = generate_training_data(k, m, n_points)
    print(f"Generated {n_points} data points from interval [0.1, {k}]")
    
    # Create smooth x values for plotting
    x_plot = np.linspace(0.1, k, 200)
    
    # Task 1: Polynomial of degree 1
    print("\n" + "=" * 60)
    print("Task 1: Polynomial model of degree 1")
    print("=" * 60)
    degree = 1
    A1 = build_data_matrix(xk, degree)
    X1 = least_squares_solution(A1, yk)
    y_pred1 = predict_polynomial(x_plot, X1, degree)
    print(f"Coefficients: {X1}")
    plot_results(xk, yk, x_plot, y_pred1, degree, 
                 "Task 1: Polynomial Regression (Degree 1)", 
                 "output/task1_degree1.png")
    
    # Task 2: Polynomial of degree 3
    print("\n" + "=" * 60)
    print("Task 2: Polynomial model of degree 3")
    print("=" * 60)
    degree = 3
    A3 = build_data_matrix(xk, degree)
    X3 = least_squares_solution(A3, yk)
    y_pred3 = predict_polynomial(x_plot, X3, degree)
    print(f"Coefficients: {X3}")
    plot_results(xk, yk, x_plot, y_pred3, degree, 
                 "Task 2: Polynomial Regression (Degree 3)", 
                 "output/task2_degree3.png")
    
    # Task 3: Polynomial of degree 7
    print("\n" + "=" * 60)
    print("Task 3: Polynomial model of degree 7")
    print("=" * 60)
    degree = 7
    A7 = build_data_matrix(xk, degree)
    X7 = least_squares_solution(A7, yk)
    y_pred7 = predict_polynomial(x_plot, X7, degree)
    print(f"Coefficients: {X7}")
    plot_results(xk, yk, x_plot, y_pred7, degree, 
                 "Task 3: Polynomial Regression (Degree 7)", 
                 "output/task3_degree7.png")
    
    # Task 4: Ridge Regression
    print("\n" + "=" * 60)
    print("Task 4: Ridge Regression for degree 7")
    print("=" * 60)
    
    # Task 4.1: Ridge regression with ρ = 10^-6
    print("\nTask 4.1: Ridge regression with ρ = 10^-6")
    rho1 = 1e-6
    X7_ridge1 = ridge_regression_solution(A7, yk, rho1)
    y_pred7_ridge1 = predict_polynomial(x_plot, X7_ridge1, degree)
    print(f"Coefficients: {X7_ridge1}")
    plot_results(xk, yk, x_plot, y_pred7_ridge1, degree, 
                 f"Task 4.1: Ridge Regression (Degree 7, ρ = {rho1})", 
                 "output/task4_1_ridge_1e-6.png")
    
    # Task 4.2: Ridge regression with ρ = 0.1
    print("\nTask 4.2: Ridge regression with ρ = 0.1")
    rho2 = 0.1
    X7_ridge2 = ridge_regression_solution(A7, yk, rho2)
    y_pred7_ridge2 = predict_polynomial(x_plot, X7_ridge2, degree)
    print(f"Coefficients: {X7_ridge2}")
    plot_results(xk, yk, x_plot, y_pred7_ridge2, degree, 
                 f"Task 4.2: Ridge Regression (Degree 7, ρ = {rho2})", 
                 "output/task4_2_ridge_0.1.png")
    
    # Summary comparison plot - ALL METHODS
    print("\n" + "=" * 60)
    print("Creating comparison plot (ALL METHODS)...")
    print("=" * 60)
    
    plt.figure(figsize=(14, 8))
    
    # Training data
    plt.scatter(xk, yk, color='red', alpha=0.6, label='Noisy Training Data', s=50, zorder=5)
    
    # All polynomial fits
    plt.plot(x_plot, y_pred1, 'c-', linewidth=2, label='Degree 1 (Linear)', alpha=0.8)
    plt.plot(x_plot, y_pred3, 'orange', linewidth=2, label='Degree 3', alpha=0.8)
    plt.plot(x_plot, y_pred7, 'b-', linewidth=2, label='Degree 7 (Least Squares)', alpha=0.8)
    plt.plot(x_plot, y_pred7_ridge1, 'g--', linewidth=2, label=f'Degree 7 Ridge (ρ = {rho1})', alpha=0.8)
    plt.plot(x_plot, y_pred7_ridge2, 'm--', linewidth=2, label=f'Degree 7 Ridge (ρ = {rho2})', alpha=0.8)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Comparison: All Polynomial Regression Methods', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/comparison_all_methods.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("All tasks completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
