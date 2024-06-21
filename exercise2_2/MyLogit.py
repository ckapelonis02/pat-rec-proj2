import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def plotData(X, y):
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k', label='Admitted')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y', label='Not admitted')


def sigmoid(z):
    sigmoid_function = 1 / (1 + np.exp(-z))
    return sigmoid_function


# Calculate the cost function
def costFunction(theta, X, y):
    y_exp = sigmoid(np.dot(X, theta))
    J = (-1 / len(y)) * (np.dot(y.T, np.log(y_exp)) + np.dot((1 - y).T, np.log(1 - y_exp)))
    return J


# Calculate the gradient of the cost function
def gradient(theta, X, y):
    y_exp = sigmoid(np.dot(X, theta))
    grad = (1 / len(y)) * np.dot(X.T, (y_exp - y))
    return grad


def plotDecisionBoundary(theta, X, y):
    plotData(X[:, 1:3], y)
    x_value = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    y_value = -(theta[0] + theta[1] * x_value) / theta[2]
    plt.plot(x_value, y_value, label='Decision Boundary')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()


# Class Prediction
def predict(theta, X):
    pred_class = sigmoid(np.dot(X, theta)) >= 0.5
    return pred_class


# Initialization
np.set_printoptions(suppress=True)

# Load Data
data = np.loadtxt('exam_scores_data1.txt', delimiter=',')
X = data[:, [0, 1]]
y = data[:, 2]

# ==================== Part 1: Plotting ====================
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
plotData(X, y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()
input('\nProgram paused. Press enter to continue.\n')

# ============ Part 2: Compute Cost and Gradient ============
m, n = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1)
initial_theta = np.zeros(n + 1)
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Gradient at initial theta (zeros):', grad)
input('\nProgram paused. Press enter to continue.\n')

# ============= Part 3: Optimizing using minimize  =============
options = {'gtol': 1e-6}  # Set tolerance to 1e-6
res = minimize(fun=costFunction, x0=initial_theta, args=(X, y), jac=gradient, method='TNC', options=options)

theta = res.x
cost = res.fun
print('Cost at theta found by minimize:', cost)
print('theta:', theta)
plotDecisionBoundary(theta, X, y)
plt.show()
input('\nProgram paused. Press enter to continue.\n')

# ============== Part 4: Predict and Accuracies ==============
# Check the result for a student with marks 45 and 85.
scores = np.array([1, 45, 85])  # intercept term 1
prob = sigmoid(np.dot(scores, theta))
print('For a student with scores 45 and 85, we predict an admission probability of', prob)

# Calculate training accuracy
p = predict(theta, X)
train_accuracy = np.mean(p == y) * 100
print(f'Train Accuracy: {train_accuracy}%')
