import numpy as np
from itertools import product
from notebook_utils import *
from rich.progress import track
from numba import jit, cuda

'''
           _                                 _   _               _
          | |                               | | | |             | |
 ___  ___ | |_   _____ _ __   _ __ ___   ___| |_| |__   ___   __| |
/ __|/ _ \| \ \ / / _ \ '__| | '_ ` _ \ / _ \ __| '_ \ / _ \ / _` |
\__ \ (_) | |\ V /  __/ |    | | | | | |  __/ |_| | | | (_) | (_| |
|___/\___/|_| \_/ \___|_|    |_| |_| |_|\___|\__|_| |_|\___/ \__,_|
'''


# @jit(target_backend='cuda')
def linear_regression(A0, y0, A1):
    debug(f'implementing linear regression solver...')
    # transpose our data for matrix multiplication
    A0['data'] = A0['data'].T
    y0['data'] = y0['data'].T
    A1['data'] = A1['data'].T

    a = np.dot(A0['data'].T, A0['data'])
    b = np.dot(A0['data'].T, y0['data'])
    x = np.dot(np.linalg.inv(a), b)

    y1 = create_matrix(y0['width'], A1['height'])

    y1['data'] = np.dot(A1['data'], x).T

    # debug(f"predicted y1 shape: {y1['data'].shape}")

    return y1

def multivariate_regression_(A0, y0, A1):
    debug(f'implementing multivariate regression solver...')
    A0['data'] = A0['data'].T
    y0['data'] = y0['data'].T
    A1['data'] = A1['data'].T

    #> Add bias term (column of ones) to X
    X_bias = np.c_[np.ones(A0['width']), A0['data']]

    #> Calculate beta using the normal equation
    beta = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y0['data'])

    y1 = create_matrix(y0['width'], A1['height'])
    y1['data'] = X_bias.dot(beta)

    return y1

def multivariate_regression(A0, y0, A1):
    debug(f'implementing multivariate regression solver...')

    A0['data'] = A0['data'].T
    y0['data'] = y0['data'].T
    A1['data'] = A1['data'].T

    y1 = create_matrix(y0['width'], A1['height'])

    max_iterations = 1000
    learning_rate  = 0.01
    bias = 0
    weights = np.zeros((A0['height'],1))

    def mean_squared_error(y0, y1):
        # debug(f" {y1.shape} minus {y0.shape} ...")
        return ((y1 - y0) ** 2).mean()

    def gradient_descent(A0, y0, y1, learning_rate):
        gradient = np.dot(A0.T, (y1 - y0)) / len(y1)
        return gradient * learning_rate

    # perform gradient descent here
    for i in range(max_iterations):
        y1['data'] = np.dot(A0['data'], weights) + bias
        cost = mean_squared_error(y0['data'], y1['data'])

        # debug(f"Iteration {i + 1}, Loss: {cost}")

        gradient = gradient_descent(A0['data'], y0['data'], y1['data'], learning_rate)

        # debug(f"weights shape is {weights.shape} and gradient is {gradient.shape}")

        weights = weights - gradient
        bias -= np.sum(y1['data'] - y0['data']) / y0['height']

    # perform predict here
    y1['data'] = np.dot(A0['data'], weights) + bias

    return y1

'''
                    _ _      _                                   _      _
                   | (_)    | |                                 | |    | |
 _ __  _ __ ___  __| |_  ___| |_ ___  _ __   _ __ ___   ___   __| | ___| |
| '_ \| '__/ _ \/ _` | |/ __| __/ _ \| '__| | '_ ` _ \ / _ \ / _` |/ _ \ |
| |_) | | |  __/ (_| | | (__| || (_) | |    | | | | | | (_) | (_| |  __/ |
| .__/|_|  \___|\__,_|_|\___|\__\___/|_|    |_| |_| |_|\___/ \__,_|\___|_|
| |
|_|
'''

class Model():
    def __init__(self):
        self.debug_mode = False

        self.test_cases = []
        self.scores     = []

        self.x_train  = None
        self.y_train  = None
        self.x_test   = None
        self.y_test   = None # note: we return this in our predict method
        self.y_golden = None # note: used in debug mode

    def configure(self, debug_mode = False, test_cases = []):
        self.debug_mode = debug_mode

        if self.debug_mode:
            configure_logging(logging.DEBUG)
        else:
            configure_logging(logging.INFO)

        debug('running in DEBUG mode')

        self.test_cases = [list(case) for case in product(*test_cases)]
        self.scores     = [0.0 for _ in range(len(self.test_cases))]

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        if self.debug_mode:
            info(f'fitted our model with x_train dimensions: {x_train["data"].shape}, y_train dimensions: {y_train["data"].shape}')

    def predict(self, x_test, y_golden = None):

        self.x_test   = x_test
        self.y_golden = y_golden
        self.y_test   = create_matrix(self.y_train['height'], self.x_test['width'])

        if self.debug_mode:
            info(f'fitted our model with x_test dimensions: {self.x_test["data"].shape}, y_test dimensions: {self.y_test["data"].shape}')
            if self.y_golden is None:
                error_out('running debug mode MUST include golden data!')
            else:
                info(f'fitted our model with y_golden dimensions: {self.y_golden["data"].shape}')

            for i, case in enumerate(self.test_cases):
                info(f'running case {case}')

                #> pre-process our data
                self.x_train = preprocess(case[0], self.x_train)
                self.y_train = preprocess(case[0], self.y_train)
                self.x_test  = preprocess(case[0], self.x_test)

                #> solve our multivariate system
                self.y_test = case[1](self.x_train, self.y_train, self.x_test)

                #> post-process our prediction
                self.y_test = postprocess(case[2], self.y_test)

                #> cross-validate our solution
                self.scores[i] = self.validate()
                info(f'case score: {self.scores[i]}')

            info(f'highest score: {self.scores[np.argmax(self.scores)]} ran with {self.test_cases[np.argmax(self.scores)]}')
        else:
            for i, case in enumerate(self.test_cases):
                info(f'running case {case}')
                self.y_test = case[1](self.x_train, self.y_train, self.x_test)

        return self.y_test

    def validate(self):
        debug(f"performing cross-validation...")
        x = self.y_test['data']
        y = self.y_golden['data']

        if len(x) != len(y):
            error_out(f'cross-validated predictions MUST have same length! {x.shape} and {y.shape}')

        #> calculate the mean values
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        #> numerator: sum of product deviations
        numerator = np.sum((x - x_mean) * (y - y_mean))

        #> denominator: product of square roots of the sum of squares of deviations
        denominator = np.sqrt(np.sum((x - x_mean) ** 2)) * np.sqrt(np.sum((y - y_mean) ** 2))

        return numerator / denominator