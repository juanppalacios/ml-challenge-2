
import time
import numpy as np
from numba import jit, cuda

from rich.progress import track
from notebook_utils import *

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
parameter_1 = [linear_regression, gradient_descent, nmf] #> solution methods

model = Model()

model.configure(debug_mode = True, test_cases = [parameter_0, parameter_1])

model.fit(train_rna, train_adt)

test_adt = model.predict(test_rna, gold_adt)

write_output('../out/debug/kaggle_challenge_2.csv', test_adt['data'])