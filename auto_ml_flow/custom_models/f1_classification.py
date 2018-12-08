import numpy as np
from scipy.optimize import minimize
from scipy.sparse import hstack

from auto_ml_flow.prediction.f1_classification import f1_loss


class F1LogisticRegression(object):

    def __init__(self, reg_lambda=0.01, beta_init=None):
        self.reg_lambda = reg_lambda
        self.beta = None
        self.beta_init=beta_init
        self.x = None
        self.y = None

    @staticmethod
    def _sigmoid_funct(z):
        return 1.0 / (1.0 + np.exp(-1*z))

    def model_loss(self, x, y, w):
        return f1_loss(y, self._sigmoid_funct(x.dot(w))) + (self.reg_lambda * np.sum(w**2.0))

    def fit(self, x, y):

        funct_opt = lambda w: self.model_loss(hstack((x, np.ones((x.shape[0],)))), y, w)

        if self.beta_init is None:
            beta_init = np.ones((x.shape[1] + 1,))
        else:
            beta_init = self.beta_init

        res = minimize(funct_opt, beta_init,
                       method='BFGS', options={'maxiter': 500})

        self.beta = res.x

        return self

    def predict_proba(self, x):
        return self._sigmoid_funct(hstack((x, np.ones((x.shape[0],)))).dot(self.beta))

    def predict(self, x):
        return (self.predict_proba(x) > 0.5).astype(int)
