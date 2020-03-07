from FTR.main_functions import *

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
alpha = 0.0001
n_steps = 10000
l = 1
sigma02 = 1

# Train the classifier
X_tilde_train = get_x_tilde(evaluate_basis_functions(l, X_train, X_train))
X_tilde_test = get_x_tilde(evaluate_basis_functions(l, X_test, X_train))
w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha, sigma02)

# Plot the training and test log likelihoods
#plot_ll(ll_train)
#plot_ll(ll_test)

# Plot the predictive distribution
plot_predictive_distribution(X, y, w, lambda x : evaluate_basis_functions(l, x, X_train))

# Average LL
print('Train: {0:.3}'.format(ll_train[-1]))
print('Test: {0:.3}'.format(ll_test[-1]))

# Compute confusion matrix
confusion = compute_confusion_matrix(X_tilde_test, w, y_test)
print('[{0:.3}, {1:.3}]'.format(confusion[0], confusion[1]))
print('[{0:.3}, {1:.3}]'.format(confusion[2], confusion[3]))
