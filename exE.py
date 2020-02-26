from main_functions import *


def compute_confusion_matrix(X_tilde, w, y):
    """Computes the confusion matrix"""
    sigmoid_values = predict(X_tilde, w)
    thresholded_values = sigmoid_values > 0.5
    confusion = [0, 0, 0, 0]
    num_ones = np.count_nonzero(y)
    num_zeros = len(y) - num_ones
    for i in range(len(y)):
        # True negatives
        if y[i] == 0 and thresholded_values[i] == 0:
            confusion[0] += 1 / num_zeros
        # False positives
        elif y[i] == 0 and thresholded_values[i] == 1:
            confusion[1] += 1 / num_zeros
        # False negatives
        elif y[i] == 1 and thresholded_values[i] == 0:
            confusion[2] += 1 / num_ones
        # True positives
        elif y[i] == 1 and thresholded_values[i] == 1:
            confusion[3] += 1 / num_ones
    return confusion


# Import dataset
X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')

# Randomly permute dataset
permutation = np.random.permutation(X.shape[0])
X = X[permutation, :]
y = y[permutation]

# Split the data into train and test sets
n_train = 800
X_train = X[0: n_train, :]
X_test = X[n_train:, :]
y_train = y[0: n_train]
y_test = y[n_train:]

# Define hyperparameters
alpha = 0.001
n_steps = 100

# Train classifier
X_tilde_train = get_x_tilde(X_train)
X_tilde_test = get_x_tilde(X_test)
w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha)

print(compute_confusion_matrix(X_tilde_test, w, y_test))
