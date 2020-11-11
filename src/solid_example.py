import numpy as np, pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as statsimport
import statsmodels.api as sm


class Precission():
    def precission(y, yhat):
        res = []
        for i in range(yhat):
            res.append((y[i] - yhat[i]) ** 2)

        return res

class Base_Builder():
    """Creating base class for different builders
    """
    x = 0
    y = 0

    def MLERegression(self, params):
        intercept, beta, sd = params[0], params[1], params[2] # inputs are guesses at our parameters
        yhat = intercept + beta*self.x # predictions# next, we flip the Bayesian question
        # compute PDF of observed values normally distributed around mean (yhat)
        # with a standard deviation of sd
        negLL = -np.sum(scipy.stats.norm.logpdf(self.y, loc=yhat, scale=sd))# return negative LL
        return(negLL)


class Graph_builder(Base_Builder):
    """Subclass for generating data and building graph, depending on dots amount
    """
    def __init__(self, dots):
        self.dots = dots

# define likelihood function
    def gen_data(self):
        self.x = np.linspace(0, 20, self.dots)
        self.y = 3 * self.x + 1 + self.get_noise()
        df = pd.DataFrame({'y':self.y, 'x':self.x})
        df['constant'] = 0
        return df

    def print_graph(self):
        """Single responsibility principle"""
        df = self.gen_data()
        print(df.head(15))
        plt.scatter(df.x, df.y)
        plt.show()
        return self

    def get_noise(self):
        """Open-closed principle
        """
        return np.random.normal(loc=0.0, scale=2.0, size=self.dots)


results = minimize(Graph_builder(20).print_graph().MLERegression, np.array([5, 5, 2]), method='Nelder-Mead',
                                                                                       options={'disp': True})
print(results['x'])
print(results['x'][0])
print(results['x'][1])
print(results['x'][2])