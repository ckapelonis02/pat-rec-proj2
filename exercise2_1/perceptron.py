import numpy as np
import matplotlib.pyplot as plt

# Given 2D samples per class
class_1 = np.array(
    [[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2, 2.7], [4.1, 2.8], [3.1, 5], [-0.8, -1.3], [0.9, 1.2], [5, 6.4],
     [3.9, 4.0]])
class_2 = np.array(
    [[7.1, 4.2], [-1.4, -4.3], [4.5, 0], [6.3, 1.6], [4.2, 1.9], [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7],
     [4.1, -2.2]])
class_3 = np.array(
    [[-3, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4, 2.2], [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6],
     [1.9, 5.1]])
class_4 = np.array(
    [[-2, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0], [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4],
     [-7.1, -9.7], [-8.0, -6.3]])


def perceptron(X, y, max_iter=1000):
    w = np.zeros(X.shape[1])  # initially zero vector
    b = 0
    reps = 0
    for reps in range(max_iter):
        errors = 0
        for xi, yi in zip(X, y):
            if yi * (np.dot(xi, w) + b) <= 0:
                w += yi * xi
                b += yi
                errors += 1
        if errors == 0:
            break
    return w, b, reps


def plot_decision_boundary(w, b, ax):
    x_values = np.linspace(-10, 10, 200)
    y_values = -(w[0] * x_values + b) / w[1]
    ax.plot(x_values, y_values, 'k--')


# Question a
plt.figure(figsize=(10, 8))
plt.scatter(class_1[:, 0], class_1[:, 1], c='red', label='class 1')
plt.scatter(class_2[:, 0], class_2[:, 1], c='blue', label='class 2')
plt.scatter(class_3[:, 0], class_3[:, 1], c='green', label='class 3')
plt.scatter(class_4[:, 0], class_4[:, 1], c='purple', label='class 4')
plt.title('Scatter Plot for Different Classes')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.show()

# Question b
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# class_1 vs class_2
X1 = np.vstack((class_1, class_2))
y1 = np.hstack((np.ones(len(class_1)), -np.ones(len(class_2))))
w1, b1, reps = perceptron(X1, y1)
plot_decision_boundary(w1, b1, axs[0])
axs[0].scatter(class_1[:, 0], class_1[:, 1], c='red', label='class 1')
axs[0].scatter(class_2[:, 0], class_2[:, 1], c='blue', label='class 2')
axs[0].set_title(f'class_1 vs class_2 took {reps} repetitions')
axs[0].set_xlim(-10, 10)
axs[0].set_ylim(-10, 10)
axs[0].legend()

# class_2 vs class_3
X2 = np.vstack((class_2, class_3))
y2 = np.hstack((np.ones(len(class_2)), -np.ones(len(class_3))))
w2, b2, reps = perceptron(X2, y2)
plot_decision_boundary(w2, b2, axs[1])
axs[1].scatter(class_2[:, 0], class_2[:, 1], c='blue', label='class 2')
axs[1].scatter(class_3[:, 0], class_3[:, 1], c='green', label='class 3')
axs[1].set_title(f'class_2 vs class_3 took {reps} repetitions')
axs[1].set_xlim(-10, 10)
axs[1].set_ylim(-10, 10)
axs[1].legend()

# class_3 vs class_4
X3 = np.vstack((class_3, class_4))
y3 = np.hstack((np.ones(len(class_3)), -np.ones(len(class_4))))
w3, b3, reps = perceptron(X3, y3)
plot_decision_boundary(w3, b3, axs[2])
axs[2].scatter(class_3[:, 0], class_3[:, 1], c='green', label='class 3')
axs[2].scatter(class_4[:, 0], class_4[:, 1], c='purple', label='class 4')
axs[2].set_title(f'class_3 vs class_4 took {reps} repetitions')
axs[2].set_xlim(-10, 10)
axs[2].set_ylim(-10, 10)
axs[2].legend()

plt.show()
