import numpy as np

# samples for class 1
class_1 = np.array([
    [0.42, -0.087, 0.58],
    [-0.2, -3.3, -3.4],
    [1.3, -0.32, 1.7],
    [0.39, 0.71, 0.23],
    [-1.6, -5.3, -0.15],
    [-0.029, 0.89, -4.7],
    [-0.23, 1.9, 2.2],
    [0.27, -0.3, -0.87],
    [-1.9, 0.76, -2.1],
    [0.87, -1, -2.6]
])

# samples for class 1
class_2 = np.array([
    [-0.4, 0.58, 0.089],
    [-0.31, 0.27, -0.04],
    [0.38, 0.055, -0.035],
    [-0.15, 0.53, 0.011],
    [-0.35, 0.47, 0.034],
    [0.17, 0.69, 0.1],
    [-0.011, 0.55, -0.18],
    [-0.27, 0.61, 0.12],
    [-0.065, 0.49, 0.0012],
    [-0.12, 0.054, -0.063]
])

# Question a
print("Question a:")
print("Means (μ):", np.mean(class_1, axis=0))
print("Variances (σ^2):", np.var(class_1, axis=0, ddof=0))
print("\n\n\n")

# Question b
combinations = [(0, 1), (1, 2), (0, 2)]
print("Question b:")
for combination in combinations:
    print(f"Characteristics x{combination[0]+1}, x{combination[1]+1}")
    print("Mean vector (μ) for 2D:", np.mean(class_1[:, combination], axis=0))
    print("Covariance matrix (Σ) for 2D:\n", np.cov(class_1[:, combination], rowvar=False, ddof=0))
    print("\n")
print("\n\n\n")

# Question c
print("Question c:")
print("Mean vector (μ) for 3D:", np.mean(class_1, axis=0))
print("Covariance matrix (Σ) for 3D:\n", np.cov(class_1, rowvar=False, ddof=0))
print("\n\n\n")

# Question d
print("Question d:")
print("Mean vector (μ) for 3D:", np.mean(class_2, axis=0))
print("Diagonal covariance matrix (Σ) for 3D:\n", np.diag(np.var(class_2, axis=0, ddof=0)))
print("\n\n\n")
