import logging
import csv
import numpy as np

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

def linear_solve(A, y):

    x = create_matrix(1, A['width'])

    for i in A['number of columns']:
        a = np.dot(A['data'][:,i], A['data'][:,i].T)
        b = np.dot(A['data'][:,i].T, y['data'][:,i])
        x[i] =  np.divide(b, a)

    return x

def gradient_descent(a, b, max_iterations = 1000, min_tolerance = 1e-9):
    raise NotImplementedError

def nmf(a, b):
    raise NotImplementedError

def svd(a, b):
    raise NotImplementedError

'''
                    _     __             _ _
                   | |   / /            (_) |
 _ __ ___  __ _  __| |  / /_      ___ __ _| |_ ___
| '__/ _ \/ _` |/ _` | / /\ \ /\ / / '__| | __/ _ \
| | |  __/ (_| | (_| |/ /  \ V  V /| |  | | ||  __/
|_|  \___|\__,_|\__,_/_/    \_/\_/ |_|  |_|\__\___|
'''

def read_input(path : str) -> dict[str, any]:

    data = np.delete(np.genfromtxt(path, delimiter = ',', dtype = float, skip_header = 1), obj = 0, axis = 1)

    return create_matrix(data.shape[1], data.shape[0], data)

def write_output(path, data) -> None:
    data = np.transpose(data)
    data = data.flatten('C')
    data = np.atleast_2d(data).T

    with open(path, mode = 'w', newline = '') as file:
        writer = csv.writer(file, delimiter = ',', lineterminator = '\r\n', quotechar = "'")
        writer.writerow(['\"Id\"', '\"Expected\"'])

        for i in range(len(data)):
            writer.writerow([f'\"row_{i + 1}\"', f'{data[i][0]}'])
