from FTR_refactor.main_functions import BayesianLogisticClassifier
import numpy as np


# Define dataset
X = 'X.txt'
y = 'y.txt'

# Define hyperparameters
sigma02 = 1
RBF = True
l = 0.1

# Initialise classifier
BLC = BayesianLogisticClassifier(X, y, l, sigma02, RBF=RBF)

# Train model
wmap, log_fmap = BLC.compute_wmap()
BLC.update_AN(wmap)

# Plot predictive distribution
BLC.plot_predictive_distribution(wmap)

# Compute confusion matrix
BLC.set_mode('train')
confusion = BLC.compute_confusion_matrix(wmap)
print('[{0:.3}, {1:.3}]'.format(confusion[0], confusion[1]))
print('[{0:.3}, {1:.3}]'.format(confusion[2], confusion[3]))
