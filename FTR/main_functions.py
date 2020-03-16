import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


class BayesianLogisticClassifier:
    def __init__(self, sigma02):
        self.sigma02 = sigma02
        self.count = 0

    def plot_data_internal(self, X, y):
        """Function that plots the points in 2D together with their labels"""
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        plt.figure()
        plt.xlim(xx.min(None), xx.max(None))
        plt.ylim(yy.min(None), yy.max(None))
        ax = plt.gca()
        ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label='Class 1')
        ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label='Class 2')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Plot data')
        plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
        return xx, yy

    def plot_data(self, X, y):
        """Function that plots the data without returning anything by calling "plot_data_internal"""
        xx, yy = self.plot_data_internal(X, y)
        plt.show()
        plt.savefig('C:\\Users\\obarn\\Google Drive\\Cambridge\\Part IIA\\3F8\\Lab\\assets\\plot1.png')

    def logistic(self, x):
        """The logistic function"""
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, X_tilde, w):
        """Function that makes predictions with a logistic classifier"""
        mu = np.dot(X_tilde, w)
        AN = self.compute_AN(X_tilde, w)
        sigma2 = np.diag(X_tilde @ AN @ np.transpose(X_tilde))
        prediction = self.logistic(np.divide(mu, np.sqrt(np.ones(
            mu.shape[0]) + np.pi * sigma2 / 8)))  # Operates over the rows of X_tilde and outputs a vector
        return prediction

    def compute_average_ll(self, X_tilde, y, w):
        """Function that computes the average loglikelihood of the logistic classifier on some data"""
        output_prob = self.predict(X_tilde, w)
        # print(output_prob)
        return np.mean(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))

    def get_x_tilde(self, X):
        """Function that expands a matrix of input features by adding a column equal to 1"""
        return np.concatenate((np.ones((X.shape[0], 1)), X), 1)

    def fit_w(self, X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha):
        """Function that finds the model parameters by optimising the likelihood using gradient descent"""
        w = np.random.randn(X_tilde_train.shape[1])
        ll_train = np.zeros(n_steps)
        ll_test = np.zeros(n_steps)
        for i in range(n_steps):
            sigmoid_value = self.logistic(np.dot(X_tilde_train, w))
            grad = np.transpose(X_tilde_train) @ (y_train - sigmoid_value) - w / self.sigma02
            w = w + alpha * grad
            ll_train[i] = self.compute_average_ll(X_tilde_train, y_train, w)
            ll_test[i] = self.compute_average_ll(X_tilde_test, y_test, w)

        return w, ll_train, ll_test

    def plot_ll(self, ll):
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

    def plot_predictive_distribution(self, X, y, w, map_inputs=lambda x: x):
        """Function that plots the predictive probabilities of the logistic classifier"""
        xx, yy = self.plot_data_internal(X, y)
        ax = plt.gca()
        X_tilde = self.get_x_tilde(map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)))
        Z = self.predict(X_tilde, w)
        Z = Z.reshape(xx.shape)
        cs2 = ax.contour(xx, yy, Z, cmap='RdBu', linewidths=2)
        plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=14)
        plt.show()

    def evaluate_basis_functions(self, l, X, Z):
        """Function that replaces initial input features by evaluating Gaussian basis functions on a grid of points"""
        X2 = np.sum(X ** 2, 1)
        Z2 = np.sum(Z ** 2, 1)
        ones_Z = np.ones(Z.shape[0])
        ones_X = np.ones(X.shape[0])
        r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
        return np.exp(-0.5 / l ** 2 * r2)

    def compute_confusion_matrix(self, X_tilde, w, y):
        """Computes the confusion matrix"""
        sigmoid_values = self.predict(X_tilde, w)
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

    def compute_AN(self, X_tilde, w):
        A0 = 1 / self.sigma02 * np.identity(len(w))
        sigmoid_value = self.logistic(np.dot(X_tilde, w))
        AN = A0 + np.transpose(X_tilde) @ np.diag(np.multiply(sigmoid_value, np.ones(sigmoid_value.shape[0]) - sigmoid_value)) @ X_tilde
        return AN

    def f(self, w, X_tilde):
        """Returns the unnormalised posterior"""
        A0 = 1 / self.sigma02 * np.identity(w.shape[0])
        prior = np.exp(-0.5 * np.transpose(w) @ A0 @ w)
        return np.prod(self.predict(X_tilde, w))*prior

    def compute_ll(self, w, *args):
        """Function that computes the loglikelihood of the logistic classifier on some data, multiplied by -1"""
        ErrorHandler = False
        self.count += 1
        if self.count % 10 == 0:
            print('Iter: {}'.format(self.count))

        X_tilde = args[0]
        y = args[1]
        A0 = 1 / self.sigma02 * np.identity(w.shape[0])
        output_prob = self.predict(X_tilde, w)
        grad = -1 * (np.transpose(X_tilde) @ (y - output_prob) - w / self.sigma02)
        if ErrorHandler:
            with np.errstate(divide='raise'):
                # Not sure this works because code has been vectorised
                try:
                    log_f = -1 * (np.sum(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob)) - (
                                0.5 * np.transpose(w) @ A0 @ w))

                except FloatingPointError:
                    log_f - 1 * (np.sum(y * (X_tilde @ w)) - (0.5 * np.transpose(w) @ A0 @ w))
        else:
            log_f = -1 * (np.sum(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob)) - (
                        0.5 * np.transpose(w) @ A0 @ w))
        return log_f, grad

    def compute_wmap(self, X_tilde_train, y, maxfun=100000):
        w0 = np.random.randn(X_tilde_train.shape[1])
        wmap = scipy.optimize.fmin_l_bfgs_b(self.compute_ll, w0, args=tuple([X_tilde_train, y]), maxfun=maxfun)
        return wmap[0], -wmap[1] # Note sign to ensure fmap is negative

    def compute_evidence(self, fmap, AN):
        return fmap*(2*np.pi)**(AN.shape[0]/2)*np.linalg.det(AN)**(-0.5)

    def f(self, w, X_tilde):
        """Returns the unnormalised posterior"""
        A0 = 1 / self.sigma02 * np.identity(w.shape[0])
        prior = np.exp(-0.5 * np.transpose(w) @ A0 @ w)
        return np.prod(self.predict(X_tilde, w))*prior

    def fit_w(self, X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha):
        """Function that finds the model parameters by optimising the likelihood using gradient descent"""
        w = np.random.randn(X_tilde_train.shape[1])
        ll_train = np.zeros(n_steps)
        ll_test = np.zeros(n_steps)
        for i in range(n_steps):
            sigmoid_value = self.logistic(np.dot(X_tilde_train, w))
            grad = np.transpose(X_tilde_train) @ (y_train - sigmoid_value) - w / self.sigma02
            w = w + alpha * grad
            ll_train[i] = self.compute_average_ll(X_tilde_train, y_train, w)
            ll_test[i] = self.compute_average_ll(X_tilde_test, y_test, w)

        return w, ll_train, ll_test