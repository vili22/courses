import numpy as np
import csv


def read_data(filename):
    with open(filename) as f:
        array2d = [[float(digit) for digit in line.split(',')] for line in f]
        return np.array(array2d)


def write_csv_accuracy(filename, headers, data):
    with open(filename, 'w+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(headers)
        writer.writerows(data)


def write_csv_logloss(filename, headers, data):
    with open(filename, 'w+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(headers)
        writer.writerows(map(lambda t: ("%d" % t[0], "%.6f" % t[1], "%.6f" % t[2], "%.6f" % t[3], "%.6f" % t[4], "%.6f" % t[5], "%.6f" % t[6], "%.6f" % t[7], "%.6f" % t[8], "%.6f" % t[9], "%.6f" % t[10]), data))
