import csv
import logging
import numpy as np
from itertools import product
from numba import jit, cuda

# todo: most of these notes are in Lecture 4 Notes
#> 1. implement cross-validation (Wold's works for NMF)
#> 1. implement SVD, NMF, gradient_descent
#> 1. implement pre-processing
#> 1. post-processing

# note: added this to suppress numba deprecation warnings
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
 _                   _
| |                 (_)
| | ___   __ _  __ _ _ _ __   __ _
| |/ _ \ / _` |/ _` | | '_ \ / _` |
| | (_) | (_| | (_| | | | | | (_| |
|_|\___/ \__, |\__, |_|_| |_|\__, |
          __/ | __/ |         __/ |
         |___/ |___/         |___/
'''

logger = logging.getLogger('debugger')

def configure_logging(level):
    logging.basicConfig(format='%(message)s\n', encoding = 'utf-8', level = level)

def debug(message : str):
    logger.debug(f'debug - {message}')

def info(message : str):
    logger.info(f'info - {message}')

def error_out(message : str):
    logger.error(f'error - {message}')
    exit(1)


'''
     _       _                                            _
    | |     | |                                          (_)
  __| | __ _| |_ __ _   _ __  _ __ ___   ___ ___  ___ ___ _ _ __   __ _
 / _` |/ _` | __/ _` | | '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
| (_| | (_| | || (_| | | |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
 \__,_|\__,_|\__\__,_| | .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
                       | |                                         __/ |
                       |_|                                        |___/
'''

def create_matrix(width, height, data = None):
    if data is None:
        data = np.zeros((width, height), dtype = float)

    matrix = {
        'data'   : data,
        'height' : data.shape[0],
        'width'  : data.shape[1],
        'number of columns' : range(data.shape[0]),
        'number of rows'    : range(data.shape[1]),
    }

    return matrix

def preprocess(method, data):
    debug('pre-processing training and testing data sets...')
    if method == 'logarithm':
        return data
    elif method == 'clip':
        return data
    elif method == 'normalize':
        return data
    else:
        error_out(f'\"{method}\" method not recognized!')

def postprocess(method, data):
    debug('post-processing predicted result...')
    if method == 'logarithm':
        return data
    elif method == 'clip':
        data['data'] = np.clip(data['data'], a_min=0, a_max=None)
        return data
    elif method == 'normalize':
        return data
    else:
        error_out(f'\"{method}\" method not recognized!')

# @jit(target_backend='cuda')
def linear_regression(A0, y0, A1):
    debug(f'implementing linear regression solver...')

    a = A0['data'].T @ A0['data']
    b = A0['data'].T @ y0['data']
    x = np.dot(np.linalg.inv(a), b)

    y1 = create_matrix(y0['height'], A1['width'])

    y1['data'] = A1['data'] @ x

    debug(f"predicted y1 shape: {y1['data'].shape}")

    return y1

def nmf(A0, y0, A1):

    '''
        algorithm
        1. initialize W (randomly)
        2. initialize H (zeros)
        3. alternating least squares
        4. update H:
        5. update W:
        6.
    '''

    y1 = create_matrix(y0['height'], A1['width'])
    return y1

def svd(A0, y0, A1):
    y1 = create_matrix(y0['height'], A1['width'])
    return y1

'''
                    _     __             _ _
                   | |   / /            (_) |
 _ __ ___  __ _  __| |  / /_      ___ __ _| |_ ___
| '__/ _ \/ _` |/ _` | / /\ \ /\ / / '__| | __/ _ \
| | |  __/ (_| | (_| |/ /  \ V  V /| |  | | ||  __/
|_|  \___|\__,_|\__,_/_/    \_/\_/ |_|  |_|\__\___|
'''

def read_input(path : str):

    data = np.delete(np.genfromtxt(path, delimiter = ',', dtype = float, skip_header = 1), obj = 0, axis = 1).T

    return create_matrix(data.shape[1], data.shape[0], data)

def write_output(path, data):
    data = np.transpose(data)
    data = data.flatten('C')
    data = np.atleast_2d(data).T

    with open(path, mode = 'w', newline = '') as file:
        writer = csv.writer(file, delimiter = ',', lineterminator = '\r\n', quotechar = "'")
        writer.writerow(['\"Id\"', '\"Expected\"'])

        for i in range(len(data)):
            writer.writerow([f'\"ID_{i + 1}\"', f'{data[i][0]}'])


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

        self.x_train = None
        self.y_train = None
        self.x_test  = None
        self.y_test  = None  # note: we return this in our predict method

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

    def predict(self, x_test, golden_data = None):
        self.x_test = x_test
        self.y_test = create_matrix(self.x_train['height'], self.x_test['width'])
        if self.debug_mode:
            if golden_data is None:
                error_out('running debug mode MUST include golden data!')

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
        '''
            source: http://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/
        '''
        # todo: Wold's method of cross-validation
        # todo: take a subset of A0, subset of A1
        return 1.0

'''
     _      _                                _
    | |    (_)                              | |
  __| |_ __ ___   _____ _ __    ___ ___   __| | ___
 / _` | '__| \ \ / / _ \ '__|  / __/ _ \ / _` |/ _ \
| (_| | |  | |\ V /  __/ |    | (_| (_) | (_| |  __/
 \__,_|_|  |_| \_/ \___|_|     \___\___/ \__,_|\___|
'''

# read in our training and testing data sets
train_rna = read_input('../train/training_set_rna.csv')
train_adt = read_input('../train/training_set_adt.csv')
test_rna  = read_input('../test/test_set_rna.csv')

# note: use only with DEBUG_MODE
gold_adt  = read_input('../test/test_set_rna.csv')

#> hyper-parameter test case lists
parameter_0 = ['logarithm', 'clip', 'normalize'] #> pre-process methods
parameter_1 = [linear_regression, nmf, svd] #> solution methods
parameter_2 = ['normalize'] #> post-process methods

parameter_0 = ['logarithm'] #> pre-process methods
parameter_1 = [linear_regression] #> solution methods
parameter_2 = ['clip'] #> post-process methods

model = Model()

model.configure(debug_mode = True, test_cases = [parameter_0, parameter_1, parameter_2])

model.fit(train_rna, train_adt)

test_adt = model.predict(test_rna, gold_adt)

write_output('../out/debug/kaggle_challenge_2.csv', test_adt['data'])