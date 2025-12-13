# Polynomial Regression Assignment

This project implements polynomial regression using least squares and Ridge regression methods.

## Student Information
- **Surname**: sukchok (k = 7)
- **Nickname**: lok (m = 3)

## Problem Description

The assignment involves:
1. Generating training data `(xk, yk)` from interval `[0.1, k]` with zero-mean Gaussian noise
2. Estimating polynomial models of different degrees using least squares
3. Applying Ridge regression to regularize high-degree polynomials

## Mathematical Formulation

### Polynomial Model
The general polynomial model is:
```
y = a₁x + a₂x² + a₃x³ + ... + aₙxⁿ + b
```

### Least Squares Solution
For the overdetermined system `AX = B`, the optimal solution is:
```
X* = (AᵀA)⁻¹AᵀB = A†B
```

### Ridge Regression Solution
The regularized solution is:
```
X^Ridge = (AᵀA + ρI)⁻¹AᵀB
```

## Tasks

1. **Task 1**: Fit polynomial of degree 1 using least squares
2. **Task 2**: Fit polynomial of degree 3 using least squares
3. **Task 3**: Fit polynomial of degree 7 using least squares
4. **Task 4**: Apply Ridge regression to degree 7 polynomial
   - **Task 4.1**: ρ = 10⁻⁶
   - **Task 4.2**: ρ = 0.1

## Installation

Install the required dependencies:

```bash
pip install numpy matplotlib
```

Or using the project file:

```bash
pip install -e .
```

## Usage

Run the main script:

```bash
python main.py
```

The script will:
1. Generate training data with noise
2. Fit polynomials of degrees 1, 3, and 7
3. Apply Ridge regression with different regularization parameters
4. Display plots for each task
5. Save plots as PNG files

## Output

The script generates the following plots:
- `task1_degree1.png`: Polynomial regression (degree 1)
- `task2_degree3.png`: Polynomial regression (degree 3)
- `task3_degree7.png`: Polynomial regression (degree 7)
- `task4_1_ridge_1e-6.png`: Ridge regression (ρ = 10⁻⁶)
- `task4_2_ridge_0.1.png`: Ridge regression (ρ = 0.1)
- `comparison_degree7.png`: Comparison of all degree 7 methods

## Code Structure

- `generate_training_data()`: Generates noisy training data
- `build_data_matrix()`: Constructs the data matrix A for polynomial regression
- `least_squares_solution()`: Solves the least squares problem
- `ridge_regression_solution()`: Solves the Ridge regression problem
- `predict_polynomial()`: Predicts y values using polynomial coefficients
- `plot_results()`: Visualizes results

