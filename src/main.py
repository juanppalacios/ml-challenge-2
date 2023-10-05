
import time
import numpy as np
from numba import jit, cuda

from rich.progress import track
from notebook_utils import *

# message level set to debugging mode
configure_logging(logging.DEBUG)

# read in our training and testing data sets
train_rna = read_input('../train/training_set_rna.csv')
train_adt = read_input('../train/training_set_adt.csv')
test_rna  = read_input('../test/test_set_rna.csv')

# creating our predicted testing ADT set
test_adt  = create_matrix(train_adt['height'], test_rna['width'])

# x = linear_solve(train_rna, train_adt)

#> selecting our model hyper-parameters
models    = ['linear_regression', 'NMF', 'SVD', ]
parameters = [1, 2, 3]

test_cases = [{} for _ in range(len(models) * len(parameters))]

for test_case in track(test_cases):
    # debug(f"\nrunning test case {test_case}\n")
    # debug(f"fitting our model with x_train y_train")
    # debug(f"evaluate model and save score to test case list")
    time.sleep(2)

#> model.fit(train_rna, train_adt)

# predict
# test_adt = predict_using(test_rna)

# write our output
write_output('../out/debug/kaggle_challenge_2.csv', test_adt['data'])