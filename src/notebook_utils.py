import csv
import logging
import numpy as np
import matplotlib.pyplot as plt

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
    # debug('pre-processing training and testing data sets...')
    if method == 'none':
        return data
    elif method == 'logarithm':
        # data = np.log(data)
        return data
    elif method == 'clip':
        return data
    elif method == 'normalize':
        minimum, maximum = np.min(data), np.max(data)
        data = (data - minimum) / (maximum - minimum)
        return data
    else:
        error_out(f'\"{method}\" method not recognized!')

def postprocess(method, data):
    # debug('post-processing predicted result...')
    if method == 'none':
        return data
    elif method == 'logarithm':
        data = np.log(data)
        return data
    elif method == 'clip':
        data['data'] = np.clip(data['data'], a_min=0, a_max=None)
        return data
    elif method == 'normalize':
        minimum, maximum = np.min(data), np.max(data)
        data = (data - minimum) / (maximum - minimum)
        return data
    else:
        error_out(f'\"{method}\" method not recognized!')

'''
                    _     __             _ _
                   | |   / /            (_) |
 _ __ ___  __ _  __| |  / /_      ___ __ _| |_ ___
| '__/ _ \/ _` |/ _` | / /\ \ /\ / / '__| | __/ _ \
| | |  __/ (_| | (_| |/ /  \ V  V /| |  | | ||  __/
|_|  \___|\__,_|\__,_/_/    \_/\_/ |_|  |_|\__\___|
'''

def read_input(path : str, trim_header = False):

    if trim_header:
        data = np.delete(np.genfromtxt(path, delimiter = ',', dtype = float, skip_header = 1), obj = 0, axis = 1)
    else:
        data = np.genfromtxt(path, delimiter = ',', dtype = float)

    data = create_matrix(data.shape[1], data.shape[0], data)

    # print(f"checking how our data unravels: {data['data'].shape}\n{data['data']}")

    return data

def visualize(path : str, x_train, y_train):
    plt.figure()

    plt.imshow(y_train['data'], cmap='viridis', alpha=0.5)
    plt.colorbar()

    plt.imshow(x_train['data'], cmap='plasma', alpha=0.5)

    plt.savefig(path, format='png', dpi=300)  # Specify the file name, format, and dpi

def write_output(path, data):
    data = data['data']

    # data = np.transpose(data)
    data = data.flatten('C')
    data = np.atleast_2d(data).T

    with open(path, mode = 'w', newline = '') as file:
        writer = csv.writer(file, delimiter = ',', lineterminator = '\r\n', quotechar = "'")
        writer.writerow(['\"Id\"', '\"Expected\"'])

        for i in range(len(data)):
            writer.writerow([f'\"ID_{i + 1}\"', f'{data[i][0]}'])