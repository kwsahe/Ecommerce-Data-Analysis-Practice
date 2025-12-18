import numpy as np
from collections import Counter

from sklearn.tree import DecisionTreeClassifier

class MyRandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifier(max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        print(f"{self.n_estimators}그루 나무 훈련 완료")

    def predict(self, X):
        tree_predictions = []
        for tree in self.trees:
            pred = tree.predict(X)
            tree_predictions.append(pred)
        
        tree_predictions = np.array(tree_predictions).T
        final_results = []
        for votes in tree_predictions:
            most_common = Counter(votes).most_common(1)[0][0]
            final_results.append(most_common)

        return np.array(final_results)