import numpy as np

class LinearRegression:
    def __init__(self, 
                 fit_intercept: bool=True,
                 copy_X: bool=True):
        """
            Initialize LinearRegression object.
            Parameters:
                - fit_intercept: whether to calculate the intercept for this model. 
                                 If set to False, no intercept will be used in calculations 
                                 (e.g. data is expected to be already centered).
                - copy_X: If True, X will be copied; else, it may be overwritten.
        """
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Fit linear model.
            Parameters:
                - X: training data
                - y: target values
            Returns:
                - self: returns an instance of self.
        """
        # TODO: check X and y
        # If copy_X is True, X_train is a copy of X
        if self.copy_X:
            X_train = X.copy()
        else:
            X_train = X
        # If fit_intercept is True, add a column of ones to X_train
        if self.fit_intercept:
            X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        # Apply SVD to X_train to avoid calculating the inverse of a singular matrix
        U, S, VT = np.linalg.svd(X_train, full_matrices=False)
        # Create an array of zeros with the same shape as X
        S_inv = np.zeros((VT.shape[0], VT.shape[0]))
        # Fill the diagonal of S_inv with the reciprocal of S
        np.fill_diagonal(S_inv, 1 / S)
        # Calculate the coefficients
        self.coef_ = VT.T @ S_inv @ U.T @ y
        # If fit_intercept is True, the last element of coef_ is the intercept
        if self.fit_intercept:
            # Then, extract the intercept and the coefficients
            self.intercept_ = self.coef_[-1]
            self.coef_ = self.coef_[:-1]
        else:
            # Otherwise, set the intercept to 0
            self.intercept_ = 0
        return self
    
    def predict(self, X: np.ndarray):
        """
            Predict using the linear model.
            Parameters:
                - X: samples
            Returns:
                - y: returns predicted values.
        """
        return np.dot(X, self.coef_) + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray):
        """
            Return the coefficient of determination of the prediction.
            Parameters:
                - X: samples
                - y: true values
        """
        u = ((y - self.predict(X)) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v

class LogisticRegression:
    def fit(self, X, y):
        return self

class SVM:
    # NOTE: Linear, Gaussian
    pass

class DecisionTree:
    pass

class RandomForest:
    pass

class XGBoost:
    pass

class LightGBM:
    pass

class PCA:
    pass

class Kmeans:
    pass

class FastRandomForest:
    pass

class FastXGBoost:
    pass

class FastLightGBM:
    pass

def test_linear_regression():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    clf = LinearRegression()
    clf.fit(X, y)
    assert clf.score(X, y) > 0.999
    assert np.allclose(clf.coef_, [1., 2.])
    assert np.allclose(clf.intercept_, 3.)
    assert np.allclose(clf.predict(np.array([[3, 5]])), [16.])

if __name__ == "__main__":
    test_linear_regression()