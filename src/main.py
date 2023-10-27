from notebook_utils  import *
from predictor_model import *

'''
                                        _      _
                                       | |    | |
 _ __ _   _ _ __    _ __ ___   ___   __| | ___| |
| '__| | | | '_ \  | '_ ` _ \ / _ \ / _` |/ _ \ |
| |  | |_| | | | | | | | | | | (_) | (_| |  __/ |
|_|   \__,_|_| |_| |_| |_| |_|\___/ \__,_|\___|_|
'''

DEBUG_MODE = True

if DEBUG_MODE:
    # note: select which cross-validation data set to run
    select_set = 0

    # note: read in our cross-validation training and testing data sets
    train_rna = read_input(f'../train/cv_train_rna/cv_training_set_rna_{select_set}.csv', trim_header = False)
    train_adt = read_input(f'../train/cv_train_adt/cv_training_set_adt_{select_set}.csv', trim_header = False)
    test_rna  = read_input(f'../test/cv_test_rna/cv_test_set_rna_{select_set}.csv', trim_header = False)

    # note: use only with DEBUG_MODE
    gold_adt  = read_input(f'../test/cv_golden_adt/cv_golden_set_adt_{select_set}.csv', trim_header = False)

else:
    #> read in our training and testing data sets
    train_rna = read_input('../train/training_set_rna.csv', trim_header = True)
    train_adt = read_input('../train/training_set_adt.csv', trim_header = True)
    test_rna  = read_input('../test/test_set_rna.csv', trim_header = True)

    gold_adt = None

    #> visualizing our input data
    visualize('../train/training_data.png', train_rna, train_adt)

#> hyper-parameter test case lists
preprocess_methods  = ['none'] #> pre-process methods
prediction_methods  = [
    [linear_regression, None],

    #> varying our iterations
    [multivariate_regression, [1500, 0.00099]],
    [multivariate_regression, [1600, 0.00099]],
    [multivariate_regression, [1700, 0.00099]],
    [multivariate_regression, [1800, 0.00099]],
    [multivariate_regression, [1900, 0.00099]],
    [multivariate_regression, [2000, 0.00099]],
    [multivariate_regression, [2100, 0.00099]],
    [multivariate_regression, [2200, 0.00099]],
    [multivariate_regression, [2300, 0.00099]],
]
postprocess_methods = ['clip'] #> post-process methods

#> run our model...

model = Model()

model.configure(debug_mode = DEBUG_MODE, parameters = [preprocess_methods, prediction_methods, postprocess_methods])

model.fit(train_rna, train_adt)

test_adt = model.predict(test_rna, gold_adt)

write_output('../out/debug/test_set_adt.csv', test_adt)