from FTR_refactor.main_functions import BayesianLogisticClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Define dataset
X = 'X.txt'
y = 'y.txt'

# Define hyperparameters
sigma02 = np.around(np.linspace(0.3, 1.2, 10), 3)
l = np.around(np.linspace(0.3, 0.9, 10), 3)

# Initialise classifier
BLC = BayesianLogisticClassifier(X, y)

# Optimise evidence
evidences = {}
heatmap = []
for i in sigma02:
    heatmap.append([])
    for j in l:
        # Update datasets
        BLC.generate_datasets(sigma02=i, l=j)

        # Train model
        wmap, log_fmap = BLC.compute_wmap()

        # Compute evidence
        evidence = BLC.compute_log_evidence(wmap, log_fmap)
        evidences[(j,i)] = evidence
        heatmap[-1].append(evidence)
        print("({0}, {1}): {2:.5}".format(j, i, evidence))

ax = sns.heatmap(heatmap, linewidth=0.5, xticklabels=l, yticklabels=sigma02)
plt.ylabel('$\sigma_0^2$')
plt.xlabel('$l$')
plt.show()

max_params = max(evidences, key=evidences.get)
print("Max: ({0}, {1}): {2:.5}".format(max_params[0], max_params[1], evidences[max_params]))
min_params = min(evidences, key=evidences.get)
print("Min: ({0}, {1}): {2:.5}".format(min_params[0], min_params[1], evidences[min_params]))