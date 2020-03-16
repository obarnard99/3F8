from FTR_refactor.main_functions import BayesianLogisticClassifier
import numpy as np
import matplotlib.pyplot as plt

# Define dataset
X = 'X.txt'
y = 'y.txt'

# Define hyperparameters
l = np.linspace(0.09, 0.11, 10)
sigma02 = np.linspace(0.9, 1.1, 10)

# Initialise classifier
BLC = BayesianLogisticClassifier(X, y)

# Optimise evidence
evidences = {}
heatmap = []
for idx, i in enumerate(sigma02):
    heatmap.append([])
    for j in l:
        print("({}, {})".format(i,j))
        # Update datasets
        BLC.generate_datasets(sigma02=i, l=j)

        # Train model
        wmap, log_fmap = BLC.compute_wmap()

        # Compute evidence
        evidence = BLC.compute_log_evidence(wmap, log_fmap)
        evidences[(i,j)] = evidence
        heatmap[idx].append(evidence)

max_params = max(evidences, key=evidences.get)
print("{}: {}".format(max_params, evidences[max_params]))
min_params = min(evidences, key=evidences.get)
print("{}: {}".format(min_params, evidences[min_params]))

plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.show()