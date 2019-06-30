from sklearn.svm import SVC # Import SVM
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB # Import Naive Bayes
from abc import ABCMeta, abstractmethod # Import for abstract class

# Interface Class
class Algorithm(object):
    __metaclass__= ABCMeta

    @abstractmethod
    def __init__(self, params=None): pass

    @abstractmethod
    def fit(self, X, Y, sample_weight=None): pass

    @abstractmethod
    def set_params(self, **params): pass

    @abstractmethod
    def predict(self, X): pass

    @abstractmethod
    def get_params(self, deep=True): pass

    @abstractmethod
    def score(self, X, y, sample_weight=None): pass


# SVM Class, "extends" Algorithm
class SVM(Algorithm):
    def __init__(self, params=None):
        gamma = 'auto'
        if params == None:
            params = {'gamma':gamma, 'C':1.0, 'kernel':'rbf', 'degree':3,
                      'coef0':0.0, 'shrinking':True, 'probability':False,
                      'tol':1e-3, 'cache_size':200, 'class_weight':None,
                      'verbose':False, 'max_iter':-1, 'decision_function_shape':'ovr',
                      'random_state':None}
        self.clf = SVC()
        self.clf.set_params(**params)
        super(SVM, self).__init__()
    def fit(self, X, y, sample_weight=None):
        self.clf.fit(X, y, sample_weight)
    def predict(self, X):
        print(self.clf.predict(X))
    def get_params(self, deep=True):
        return self.clf.get_params(deep)
    def set_params(self, **params):
        return self.clf.set_params(**params)
    def decision_function(self, X):
        return self.clf.decision_function(X)
    def score(self, X, y, sample_weight=None):
        return self.clf.score(X, y, sample_weight)

# Bayes Class, "extends" Algorithm
# paramns, Puede ser None, pero en caso de ser un parámetro debe ser un dict
class Bayes(Algorithm):
    def __init__(self, params=None):
        if params == None:
            # params = {'priors':None, 'var_smoothing':1e-9}
            params = {'alpha':1.0, 'fit_prior':True, 'class_prior':None}
        self.clf = MultinomialNB()
        self.clf.set_params(**params)
    def fit(self, X, y, sample_weight=None):
        self.clf.fit(X, y, sample_weight)
    def predict(self, X):
        pred = self.clf.predict(X)
        # print(self.clf.predict(X))
        return pred
    def get_params(self, deep=True):
        return self.clf.get_params(deep)
    def set_params(self, **params):
        return self.clf.set_params(**params)
    def score(self, X, y, sample_weight=None):
        return self.clf.score(X, y, sample_weight)
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        return self.clf.partial_fit(X, y, classes, sample_weight)
    def predict_log_proba(self, X):
        return self.clf.predict_log_proba(X)
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
