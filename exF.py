from main_functions import *

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

# Expand the data
l = # XXX Width of the Gaussian basis function. To be completed by the student

X_tilde_train = get_x_tilde(evaluate_basis_functions(l, X_train, X_train))
X_tilde_test = get_x_tilde(evaluate_basis_functions(l, X_test, X_train))

# We train the new classifier on the feature expanded inputs

alpha = # XXX Learning rate for gradient-based optimisation with basis functions. To be completed by the student
n_steps = # XXX Number of steps of gradient-based optimisation with basis functions. To be completed by the student

w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha)
