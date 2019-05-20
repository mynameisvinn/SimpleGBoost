import numpy as np
from sklearn import tree

class SimpleGBoost(object):
    
    def __init__(self, n_trees=3):
        self.n_trees = n_trees


    def fit(self, X, y):
        """fit estimator with data.
        """
        self.X = X
        self.y = y
        self.ls_trees = []  # list of shallow trees
    
        f = self._initial_guess()  # initial prediction is just the mean
        
        for _ in range(self.n_trees):
            f = self._fit(f)
        
        return self
    
    def _initial_guess(self):        
        f0 = np.zeros(self.y.shape[0])  # initial predictions is the naive average y
        f0[:] = np.mean(self.y)
        self.f0 = f0.astype(int)
        return self.f0
    
    def _fit(self, f_initial):
        """given f_i, return f_i+1. 
        
        each call creates and appends a new shallow model to the
        tree.
        """

        # first, calculate residuals
        residuals = self.y - f_initial

        # fit a shallow tree on residuals
        model = tree.DecisionTreeClassifier(max_features=1, max_depth=1)
        model.fit(self.X, residuals)
        h = model.predict(self.X)
        
        # append each fitted model so we can use it later
        self.ls_trees.append(model)
        
        # new prediction is f_initial + h
        f_new = f_initial + h
        new_residuals = self.y - f_new
        return f_new
    
    def predict(self):
        """
        final prediction is (1) initial prediction, which is the mean, plus
        (2) all predicted residuals
        """
        predictions = np.copy(self.f0)  # start with the naive average
        for clf in self.ls_trees:
            predictions += clf.predict(self.X)
        return predictions