import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 2)
X = np.r_[X, np.random.randn(100, 2) + [2, 2]]  # Add some outliers

# Fit the model
clf = IsolationForest(contamination=0.1, random_state=42)
y_pred = clf.fit_predict(X)

# Separate the predictions
normal = X[y_pred == 1]
anomaly = X[y_pred == -1]

# Plot the results
plt.figure(figsize=(10, 7))
plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
plt.scatter(anomaly[:, 0], anomaly[:, 1], c='red', label='Anomaly')
plt.title('Anomaly Detection with Isolation Forest')
plt.legend()
plt.show()

print(f"Number of anomalies detected: {len(anomaly)}")
