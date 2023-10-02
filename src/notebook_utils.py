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

def configure_logging(level) -> None:
    logging.basicConfig(format='%(message)s\n', encoding = 'utf-8', level = level)

def debug(message : str) -> None:
    logger.debug(f'debug - {message}')

def info(message : str) -> None:
    logger.info(f'info - {message}')


'''
    data processing
'''
def gradient_descent(a, b, max_iterations = 1000, min_tolerance = 1e-9):
    
    x = np.zeros(len(b))
    # debug()
    
    #>  x   <- 0
    #>  for i in range(0, max_iterations):
    #>      tol <- 0
    #>      for i in range(0, b'length):
    #>          beta <- b[i] / a[i][j]
    #>          x[i] <- x[i] + beta
    #>          b    <- b - beta * a[i][j]
    #>          tol  <- tol + abs(beta / x[i])
    #>      if tol < min_tolerance: break

def create_matrix(width, height):

    data = np.zeros((width, height), dtype = float)

    matrix = {
        'data'   : data,
        'height' : data.shape[0],
        'width'  : data.shape[1],
        'number of columns' : range(data.shape[0]),
        'number of rows'    : range(data.shape[1]),
    }

    return matrix


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

    matrix = {
        'data'   : data,
        'height' : data.shape[0],
        'width'  : data.shape[1],
        'number of columns' : range(data.shape[0]),
        'number of rows'    : range(data.shape[1]),
    }

    return matrix

def write_output(path, data) -> None:
    data = np.transpose(data)
    data = data.flatten('C')
    data = np.atleast_2d(data).T

    with open(path, mode = 'w', newline = '') as file:
        writer = csv.writer(file, delimiter = ',', lineterminator = '\r\n', quotechar = "'")
        writer.writerow(['\"Id\"', '\"Expected\"'])

        for i in range(len(data)):
            writer.writerow([f'\"row_{i + 1}\"', f'{data[i][0]}'])
