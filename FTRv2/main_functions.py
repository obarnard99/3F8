import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


class BayesianLogisticClassifier:
    # Dataset Initialisation
    def __init__(self, X, y, mode='train', predictor='laplace'):
        self.X0 = np.loadtxt(X)
        self.y0 = np.loadtxt(y)

        # Randomly permute dataset
        permutation = np.random.permutation(self.X0.shape[0])
        X = self.X0[permutation, :]
        y = self.y0[permutation]

        # Split the data into train and test sets
        n_train = int(0.8 * X.shape[0])
        self.X_train = X[0: n_train, :]
        self.X_test = X[n_train:, :]
        self.y_train = y[0: n_train]
        self.y_test = y[n_train:]

        self.mode = mode
        self.set_predictor(predictor)

    def generate_datasets(self, sigma02=1, l=None):
        """Function that expands a matrix of input features by adding a column equal to 1 and with radial basis
        functions if necessary """
        self.sigma02 = sigma02
        self.l = l
        if self.l is not None:
            expanded_X_train = self._evaluate_basis_functions(l, self.X_train, self.X_train)
            self.X_tilde_train = self._get_x_tilde(expanded_X_train)
            expanded_X_test = self._evaluate_basis_functions(l, self.X_test, self.X_train)
            self.X_tilde_test = self._get_x_tilde(expanded_X_test)
        else:
            self.X_tilde_train = self._get_x_tilde(self.X_train)
            self.X_tilde_test = self._get_x_tilde(self.X_test)
        self.set_mode(self.mode) # Updates internal datasets accordingly

    def set_mode(self, mode):
        """Sets the mode of operation to training or testing"""
        self.mode = mode
        if mode.lower() == 'train':
            self.X_tilde = self.X_tilde_train
            self.y = self.y_train
        elif mode.lower() == 'test':
            self.X_tilde = self.X_tilde_test
            self.y = self.y_test

    def set_predictor(self, fn):
        self.predictor = fn.lower()
        if fn.lower() == 'laplace':
            self._predict = self._compute_laplace_prediction
        elif fn.lower() == 'map':
            self._predict = self._compute_MAP_prediction

    def _get_x_tilde(self, X):
        """Function that expands a matrix of input features by adding a column equal to 1"""
        return np.concatenate((np.ones((X.shape[0], 1)), X), 1)

    def _evaluate_basis_functions(self, l, X, Z):
        """Function that replaces initial input features by evaluating Gaussian basis functions on a grid of points"""
        X2 = np.sum(X ** 2, 1)
        Z2 = np.sum(Z ** 2, 1)
        ones_Z = np.ones(Z.shape[0])
        ones_X = np.ones(X.shape[0])
        r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
        return np.exp(-0.5 / l ** 2 * r2)


    # Training
    def _logistic(self, x, NaNflag=False):
        """The logistic function. If function overflows, returns 0 for that value or NaN if NaNflag set"""
        f = np.array([])
        with np.errstate(all='raise'):
            for i in x:
                try:
                    val = 1.0 / (1.0 + np.exp(-i))
                except FloatingPointError:
                    if NaNflag:
                        val = np.nan
                    else:
                        val = 0
                f = np.append(f, val)
        return f

    def _compute_laplace_prediction(self, X_tilde, w):
        """Function that makes predictions with a Laplacian approximation"""
        AN = self._compute_AN(w)
        mu = np.dot(X_tilde, w)
        sigma2 = np.diag(X_tilde @ AN @ np.transpose(X_tilde))
        exponent = np.divide(mu, np.sqrt(1 + np.pi * sigma2 / 8))
        return self._logistic(exponent)

    def _compute_MAP_prediction(self, X_tilde, w):
        """Function that makes predictions with a logistic classifier and the MAP weights"""
        return self._logistic(np.dot(X_tilde, w))

    def _compute_ll(self, w):
        """Function that computes the loglikelihood of the logistic classifier on some data, multiplied by -1"""
        A0 = 1 / self.sigma02 * np.identity(w.shape[0])
        exponent = np.dot(self.X_tilde, w)
        sigmoid_value = self._logistic(exponent, NaNflag=True)
        pointwise_ll = self.y * np.log(sigmoid_value) + (1. - self.y) * np.log(1.0 - sigmoid_value)
        inds = np.where(np.isnan(pointwise_ll))
        pointwise_ll[inds] = np.take(exponent, inds)
        if self.predictor == 'map':
            ll = -1. * np.sum(pointwise_ll)
            sigmoid_value[inds] = 0
            grad = -1 * np.transpose(self.X_tilde) @ (self.y - sigmoid_value)
        elif self.predictor == 'laplace':
            ll = -1. * (np.sum(pointwise_ll) - (0.5 * np.transpose(w) @ A0 @ w) - w.shape[0]/2 * np.log(2*np.pi*self.sigma02))
            sigmoid_value[inds] = 0
            grad = -1 * (np.transpose(self.X_tilde) @ (self.y - sigmoid_value) - w / self.sigma02)
        return ll, grad

    def compute_wmap(self):
        w0 = np.zeros(self.X_tilde.shape[1])
        wmap = scipy.optimize.fmin_l_bfgs_b(self._compute_ll, w0)
        return wmap[0], -wmap[1] # Note sign to ensure log(fmap) is negative

    def _compute_AN(self, w):
        A0 = 1 / self.sigma02 * np.identity(len(w))
        sigmoid_value = self._logistic(np.dot(self.X_tilde, w))
        return A0 + np.transpose(self.X_tilde) @ np.diag(np.multiply(sigmoid_value, 1 - sigmoid_value)) @ self.X_tilde


    # Plotting
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

    def plot_predictive_distribution(self, w):
        """Function that plots the predictive probabilities of the logistic classifier"""
        xx, yy = self._plot_data_internal()
        ax = plt.gca()
        if self.l is not None:
            map_inputs = lambda x: self._evaluate_basis_functions(self.l, x, self.X_train)
            mapped_inputs = map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1))
        else:
            mapped_inputs = np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)
        X_tilde = self._get_x_tilde(mapped_inputs)
        Z = self._predict(X_tilde, w)
        Z = Z.reshape(xx.shape)
        cs2 = ax.contour(xx, yy, Z, cmap='RdBu', linewidths=2)
        plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=14)
        plt.show()

    def _plot_data_internal(self):
        """Function that plots the points in 2D together with their labels"""
        x_min, x_max = self.X0[:, 0].min() - .5, self.X0[:, 0].max() + .5
        y_min, y_max = self.X0[:, 1].min() - .5, self.X0[:, 1].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        plt.figure()
        plt.xlim(xx.min(None), xx.max(None))
        plt.ylim(yy.min(None), yy.max(None))
        ax = plt.gca()
        ax.plot(self.X0[self.y0 == 0, 0], self.X0[self.y0 == 0, 1], 'ro', label='Class 1')
        ax.plot(self.X0[self.y0 == 1, 0], self.X0[self.y0 == 1, 1], 'bo', label='Class 2')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Plot data')
        plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
        return xx, yy

    def plot_data(self):
        """Function that plots the data without returning anything by calling "plot_data_internal"""
        xx, yy = self._plot_data_internal()
        plt.show()
        #plt.savefig('C:\\Users\\obarn\\Google Drive\\Cambridge\\Part IIA\\3F8\\Lab\\assets\\plot1.png')


    # Performance Metrics
    def compute_average_ll(self, w):
        """Function that computes the average loglikelihood of the logistic classifier on some data"""
        return -self._compute_ll(w)[0]/self.X_tilde.shape[0]

    def compute_confusion_matrix(self, w):
        """Computes the confusion matrix"""
        prediction = self._predict(self.X_tilde, w)
        thresholded_values = prediction > 0.5
        num_ones = np.count_nonzero(self.y)
        num_zeros = len(self.y) - num_ones
        confusion = [0, 0, 0, 0]
        for i in range(len(self.y)):
            # True negatives
            if self.y[i] == 0 and thresholded_values[i] == 0:
                confusion[0] += 1 / num_zeros
            # False positives
            elif self.y[i] == 0 and thresholded_values[i] == 1:
                confusion[1] += 1 / num_zeros
            # False negatives
            elif self.y[i] == 1 and thresholded_values[i] == 0:
                confusion[2] += 1 / num_ones
            # True positives
            elif self.y[i] == 1 and thresholded_values[i] == 1:
                confusion[3] += 1 / num_ones
        return confusion

    def compute_log_evidence(self, w, log_fmap):
        """Computes the natural logarithm of the model evidence. Note that we throw away the constant term as this
        adds unnecessary computation """
        AN = self._compute_AN(w)
        sign, logdet = np.linalg.slogdet(AN)
        return log_fmap - 0.5 * logdet #+ (AN.shape[0]/2) * np.log(2*np.pi)

