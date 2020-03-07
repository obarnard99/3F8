import numpy as np
import matplotlib.pyplot as plt

def plot_data_internal(X, y):
    """Function that plots the points in 2D together with their labels"""
    x_min, x_max = X[ :, 0 ].min() - .5, X[ :, 0 ].max() + .5
    y_min, y_max = X[ :, 1 ].min() - .5, X[ :, 1 ].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(None), xx.max(None))
    plt.ylim(yy.min(None), yy.max(None))
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label = 'Class 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Class 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot data')
    plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
    return xx, yy


def plot_data(X, y):
    """Function that plots the data without returning anything by calling "plot_data_internal"""
    xx, yy = plot_data_internal(X, y)
    plt.show()
    plt.savefig('C:\\Users\\obarn\\Google Drive\\Cambridge\\Part IIA\\3F8\\Lab\\assets\\plot1.png')


def logistic(x):
    """The logistic function"""
    return 1.0 / (1.0 + np.exp(-x))


def predict(X_tilde, w):
    """Function that makes predictions with a logistic classifier"""
    return logistic(np.dot(X_tilde, w)) # Operates over the rows of X_tilde and outputs a vector # np.dot() and np.matmul() are the same here


def compute_average_ll(X_tilde, y, w):
    """Function that computes the average loglikelihood of the logistic classifier on some data"""
    output_prob = predict(X_tilde, w)
    #print(output_prob)
    return np.mean(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))


def get_x_tilde(X):
    """Function that expands a matrix of input features by adding a column equal to 1"""
    return np.concatenate((np.ones((X.shape[ 0 ], 1 )), X), 1)


def fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha):
    """Function that finds the model parameters by optimising the likelihood using gradient descent"""
    w = np.random.randn(X_tilde_train.shape[ 1 ])
    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)
    for i in range(n_steps):
        sigmoid_value = predict(X_tilde_train, w)
        grad = np.matmul(np.transpose(X_tilde_train),y_train - sigmoid_value)
        w = w + alpha*grad
        ll_train[ i ] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[ i ] = compute_average_ll(X_tilde_test, y_test, w)
        #print(ll_train[ i ], ll_test[ i ])

    return w, ll_train, ll_test


def plot_ll(ll):
    """Function that plots the average log-likelihood returned by "fit_w"""
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
    ax.plot(np.arange(1, len(ll) + 1), ll, 'r-')
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title('Plot Average Log-likelihood Curve')
    plt.show()


def plot_predictive_distribution(X, y, w, map_inputs = lambda x : x):
    """Function that plots the predictive probabilities of the logistic classifier"""
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_tilde = get_x_tilde(map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)))
    Z = predict(X_tilde, w)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.show()


def evaluate_basis_functions(l, X, Z):
    """Function that replaces initial input features by evaluating Gaussian basis functions on a grid of points"""
    X2 = np.sum(X**2, 1)
    Z2 = np.sum(Z**2, 1)
    ones_Z = np.ones(Z.shape[ 0 ])
    ones_X = np.ones(X.shape[ 0 ])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / l**2 * r2)


def compute_confusion_matrix(X_tilde, w, y):
    """Computes the confusion matrix"""
    sigmoid_values = predict(X_tilde, w)
    thresholded_values = sigmoid_values > 0.5
    num_ones = np.count_nonzero(y)
    num_zeros = len(y) - num_ones
    confusion = [0, 0, 0, 0]
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

