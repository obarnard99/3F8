from FTR_refactor.main_functions import BayesianLogisticClassifier

# Define dataset
X = 'X.txt'
y = 'y.txt'

# Define hyperparameters
sigma02 = 0.6
l = 0.5

# Initialise classifier
BLC = BayesianLogisticClassifier(X, y)
BLC.generate_datasets(sigma02=sigma02, l=l)

# Train model
wmap, log_fmap = BLC.compute_wmap()

# Calculate test metrics

# Set model used for prediction
print('Laplace:')
BLC.set_predictor('laplace')

# Plot predictive distribution
BLC.set_mode('test')
BLC.plot_predictive_distribution(wmap)

# Average LL
BLC.set_mode('train')
print('Train: {0:.3}'.format(BLC.compute_average_ll(wmap)))
BLC.set_mode('test')
print('Test: {0:.3}'.format(BLC.compute_average_ll(wmap)))

# Compute confusion matrix
confusion = BLC.compute_confusion_matrix(wmap)
print('[{0:.3}, {1:.3}]'.format(confusion[0], confusion[1]))
print('[{0:.3}, {1:.3}]'.format(confusion[2], confusion[3]))

'''
# Set model used for prediction
print('\nMAP:')
BLC.set_predictor('MAP')

# Plot predictive distribution
BLC.plot_predictive_distribution(wmap)

# Average LL
BLC.set_mode('train')
print('Train: {0:.3}'.format(BLC.compute_average_ll(wmap)))
BLC.set_mode('test')
print('Test: {0:.3}'.format(BLC.compute_average_ll(wmap)))

# Compute confusion matrix
confusion = BLC.compute_confusion_matrix(wmap)
print('[{0:.3}, {1:.3}]'.format(confusion[0], confusion[1]))
print('[{0:.3}, {1:.3}]'.format(confusion[2], confusion[3]))
'''
