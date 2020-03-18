import numpy as np
import matplotlib.pyplot as plt
from FTRv1.main_functions import BayesianLogisticClassifier

# Initialise dataset
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
n_steps = 1000
sigma02 = 1

# Initialise classifier
BLC = BayesianLogisticClassifier(sigma02)

# Train the classifier
X_tilde_train = BLC.get_x_tilde(X_train)
X_tilde_test = BLC.get_x_tilde(X_test)

wmap, log_fmap = BLC.compute_wmap(X_tilde_train, y_train)
AN = BLC.compute_AN(X_tilde_train, wmap)
Z = BLC.compute_evidence(np.exp(log_fmap), AN)

# Plot the predictive distribution
BLC.plot_predictive_distribution(X, y, wmap)

# Average LL
ll_train = BLC.compute_average_ll(X_tilde_train, y_train, wmap)
ll_test = BLC.compute_average_ll(X_tilde_test, y_test, wmap)
print('Train: {0:.3}'.format(ll_train))
print('Test: {0:.3}'.format(ll_test))

# Compute confusion matrix
confusion = BLC.compute_confusion_matrix(X_tilde_test, wmap, y_test)
print('[{0:.3}, {1:.3}]'.format(confusion[0], confusion[1]))
print('[{0:.3}, {1:.3}]'.format(confusion[2], confusion[3]))

'''
print(np.exp(log_fmap))
print(Z)

# Plot the training and test log likelihoods
#BLC.plot_ll(ll_train)
#BLC.plot_ll(ll_test)


'''