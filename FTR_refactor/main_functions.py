import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


class BayesianLogisticClassifier:
    # Dataset Initialisation
    def __init__(self, X, y, l, sigma02, RBF=False, mode='train'):
        self.sigma02 = sigma02
        self.count = 0
        self.RBF = RBF
        self.l = l
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

        self.generate_datasets()
        self.set_mode(mode)

    def generate_datasets(self):
        """Function that expands a matrix of input features by adding a column equal to 1 and with radial basis
        functions if necessary """
        if self.RBF:
            expanded_X_train = self._evaluate_basis_functions(self.l, self.X_train, self.X_train)
            self.X_tilde_train = np.concatenate((np.ones((expanded_X_train.shape[0], 1)), expanded_X_train), 1)
            expanded_X_test = self._evaluate_basis_functions(self.l, self.X_test, self.X_train)
            self.X_tilde_test = np.concatenate((np.ones((expanded_X_test.shape[0], 1)), expanded_X_test), 1)

        else:
            self.X_tilde_train = np.concatenate((np.ones((self.X_train.shape[0], 1)), self.X_train), 1)
            self.X_tilde_test = np.concatenate((np.ones((self.X_train.shape[0], 1)), self.X_train), 1)

    def set_mode(self, mode):
        """Sets the mode of operation to training or testing"""
        if mode.lower() == 'train':
            self.X_tilde = self.X_tilde_train
            self.y = self.y_train
        elif mode.lower() == 'test':
            self.X_tilde = self.X_tilde_test
            self.y = self.y_test

    def _evaluate_basis_functions(self, l, X, Z):
        """Function that replaces initial input features by evaluating Gaussian basis functions on a grid of points"""
        X2 = np.sum(X ** 2, 1)
        Z2 = np.sum(Z ** 2, 1)
        ones_Z = np.ones(Z.shape[0])
        ones_X = np.ones(X.shape[0])
        r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
        return np.exp(-0.5 / l ** 2 * r2)


    # Training
    def _logistic(self, x):
        """The logistic function. Returns NaN if function overflows"""
        f = np.array([])
        with np.errstate(all='raise'):
            for i in x:
                try:
                    val = 1.0 / (1.0 + np.exp(-i))
                except FloatingPointError:
                    val = np.nan
                f = np.append(f, val)
        return f

    def _predict(self, X_tilde, w, ll=False):
        """Function that makes predictions with a logistic classifier. If the logistic function overflows,
        only the exponent is returned as per the coursework hint """
        mu = np.dot(X_tilde, w)
        sigma2 = np.diag(X_tilde @ self.AN @ np.transpose(X_tilde))
        exponent = np.divide(mu, np.sqrt(1 + np.pi * sigma2 / 8))
        prediction = self._logistic(exponent)
        inds = np.where(np.isnan(prediction))
        if ll:
            prediction[inds] = -np.take(exponent, inds)
        else:
            prediction[inds] = 1
        return prediction

    def update_AN(self, w):
        A0 = 1 / self.sigma02 * np.identity(len(w))
        sigmoid_value = self._logistic(np.dot(self.X_tilde, w))
        self.AN = A0 + np.transpose(self.X_tilde) @ np.diag(np.multiply(sigmoid_value, np.ones(sigmoid_value.shape[0]) - sigmoid_value)) @ self.X_tilde
        #print(self.AN)

    def _compute_ll(self, w):
        """Function that computes the loglikelihood of the logistic classifier on some data, multiplied by -1"""
        ErrorHandler = True
        iterCount = False
        if iterCount:
            self.count += 1
            if self.count % 10 == 0:
                print('Iter: {}'.format(self.count))

        A0 = 1 / self.sigma02 * np.identity(w.shape[0])
        self.update_AN(w)
        output_prob = self._predict(self.X_tilde, w, ll=True)
        grad = -1 * (np.transpose(self.X_tilde) @ (self.y - output_prob) - w / self.sigma02)
        log_f = np.array([])
        if ErrorHandler:
            for idx, prob in enumerate(output_prob):
                if prob < 0.:
                    val = -prob
                else:
                    val = self.y[idx] * np.log(prob) + (1. - self.y[idx]) * np.log(1.0 - prob)
                log_f = np.append(log_f, val)
            log_f = -1 * (np.sum(log_f) - (0.5 * np.transpose(w) @ A0 @ w))
        else:
            log_f = -1. * (np.sum(self.y * np.log(output_prob) + (1. - self.y) * np.log(1.0 - output_prob)) - (
                        0.5 * np.transpose(w) @ A0 @ w))
        return log_f, grad

    def compute_wmap(self):
        w0 = np.zeros(self.X_tilde.shape[1])
        wmap = scipy.optimize.fmin_l_bfgs_b(self._compute_ll, w0)
        return wmap[0], -wmap[1] # Note sign to ensure fmap is negative


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
        if self.RBF:
            map_inputs = lambda x: self._evaluate_basis_functions(self.l, x, self.X_train)
            mapped_inputs = map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1))
        else:
            mapped_inputs = np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)
        X_tilde = np.concatenate((np.ones((mapped_inputs.shape[0], 1)), mapped_inputs), 1)
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
        output_prob = self._predict(self.X_tilde, w)
        return np.mean(self.y * np.log(output_prob) + (1 - self.y) * np.log(1.0 - output_prob))

    def compute_confusion_matrix(self, w):
        """Computes the confusion matrix"""
        sigmoid_values = self._predict(self.X_tilde, w)
        thresholded_values = sigmoid_values > 0.5
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

    def compute_evidence(self, fmap):
        return fmap*(2*np.pi)**(self.AN.shape[0]/2)*np.linalg.det(self.AN)**(-0.5)



