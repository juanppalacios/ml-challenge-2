import numpy as np
from itertools import product
from notebook_utils import *
from rich.progress import track
from numba import jit, cuda
import matplotlib.pyplot as plt


'''
           _                                 _   _               _
          | |                               | | | |             | |
 ___  ___ | |_   _____ _ __   _ __ ___   ___| |_| |__   ___   __| |
/ __|/ _ \| \ \ / / _ \ '__| | '_ ` _ \ / _ \ __| '_ \ / _ \ / _` |
\__ \ (_) | |\ V /  __/ |    | | | | | |  __/ |_| | | | (_) | (_| |
|___/\___/|_| \_/ \___|_|    |_| |_| |_|\___|\__|_| |_|\___/ \__,_|
'''


# @jit(target_backend='cuda')
def linear_regression(A0, y0, A1, learning_rates = None):
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

    A0['data'] = A0['data'].T
    y0['data'] = y0['data'].T
    A1['data'] = A1['data'].T

    return y1

# @jit(target_backend='cuda')
def multivariate_regression(A0, y0, A1, learning_rates):
    debug(f'implementing multivariate regression solver...')

    A0['data'] = A0['data'].T
    y0['data'] = y0['data'].T
    A1['data'] = A1['data'].T

    x_train = np.hstack([np.ones((A0['width'], 1)), A0['data']])
    x_test  = np.hstack([np.ones((A1['width'], 1)), A1['data']])

    y1 = create_matrix(y0['height'], A1['width'])

    max_iterations = learning_rates[0]
    learning_rate  = learning_rates[1]

    @jit(target_backend='cuda')
    def gradient_descent(A0, y0, learning_rate, iterations):
        m, n  = A0.shape
        theta = np.zeros(n)

        for _ in range(iterations):
            error = A0 @ theta - y0
            gradient = A0.T @ error
            theta -= learning_rate * gradient / m
        return theta

    # perform gradient descent here
    for i in track(range(y0['height'])):
        y_train = y0['data'][:, i]
        theta = gradient_descent(x_train, y_train, learning_rate, max_iterations)
        y1['data'].T[:,i] = x_test @ theta

    A0['data'] = A0['data'].T
    y0['data'] = y0['data'].T
    A1['data'] = A1['data'].T

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

        self.parameters = []
        self.scores     = []

        self.x_train  = None
        self.y_train  = None
        self.x_test   = None
        self.y_test   = None

        self.y_golden = None # note: used in debug mode

    def configure(self, debug_mode = False, parameters = []):
        self.debug_mode = debug_mode

        if self.debug_mode:
            configure_logging(logging.DEBUG)
        else:
            configure_logging(logging.INFO)

        debug('running in DEBUG mode')

        self.parameters = [list(parameter) for parameter in product(*parameters)]
        self.scores     = [0.0  for _ in range(len(self.parameters))]
        self.y_test     = [None for _ in range(len(self.parameters))]

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        if self.debug_mode:
            info(f'fitted our model with x_train dimensions: {x_train["data"].shape}, y_train dimensions: {y_train["data"].shape}')

    def predict(self, x_test, y_golden = None):

        self.x_test   = x_test
        self.y_golden = y_golden
        self.y_test   = [create_matrix(self.y_train['height'], self.x_test['width']) for _ in range(len(self.parameters))]
        iterations    = [0 for _ in range(len(self.parameters))]

        if self.debug_mode:
            info(f'fitted our model with x_test dimensions: {self.x_test["data"].shape}, y_test dimensions: {self.y_test[0]["data"].shape}')
            if self.y_golden is None:
                error_out('running debug mode MUST include golden data!')
            else:
                info(f'fitted our model with y_golden dimensions: {self.y_golden["data"].shape}')

            #> run through all of our hyper-parameter lists, resets each iteration
            for i, parameter in enumerate(self.parameters):
                info(f'running parameter {parameter}')
                iterations[i] = parameter[1][1][0]

                #> pre-process our data
                debug('pre-processing training and testing data sets...')
                self.x_train = preprocess(parameter[0], self.x_train)
                self.y_train = preprocess(parameter[0], self.y_train)
                self.x_test  = preprocess(parameter[0], self.x_test)

                #> solve our multivariate system
                self.y_test[i] = parameter[1][0](self.x_train, self.y_train, self.x_test, parameter[1][1])

                #> post-process our prediction
                debug('post-processing predicted result...')
                self.y_test[i] = postprocess(parameter[2], self.y_test[i])

                #> cross-validate our solution
                self.scores[i] = self.validate(i)
                info(f'parameter score: {self.scores[i]}')

            #> visualize our trend line
            plt.plot(iterations, self.scores, marker = 'o', linestyle = '-')
            # plt.ylim(0.60, 0.90)

            #> Add labels and title
            plt.xlabel('Iterations')
            plt.ylabel('Test Scores')
            plt.title('Test Scores vs. Iterations')

            #> Save plot to out folder
            plt.savefig('../out/performance.png')

            info(f'highest score: {self.scores[np.argmax(self.scores)]} ran with {self.parameters[np.argmax(self.scores)]}')
            return self.y_test[np.argmax(self.scores)]
        else:
            for i, parameter in enumerate(self.parameters):
                info(f'running parameter {parameter}')
                self.y_test = parameter[1][0](self.x_train, self.y_train, self.x_test, parameter[1][1])
            return self.y_test

    def validate(self, i):
        debug(f"performing cross-validation...")
        x = self.y_test[i]['data']
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