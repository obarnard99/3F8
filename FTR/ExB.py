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
alpha = 0.001
n_steps = 100

# Train classifier
X_tilde_train = get_x_tilde(X_train)
X_tilde_test = get_x_tilde(X_test)

w = np.random.randn(X_tilde_train.shape[1])
print(w.shape)
prior = init_prior(w, 1)
print(prior(w))
print(predict(X_tilde_train, w))
print(f(w, X_tilde_train, prior))
