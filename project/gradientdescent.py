import numpy as np


def gradient_descent(r, X, alpha):
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X = np.c_[np.ones([X.shape[0], 1]), X]

    wjs = np.random.uniform(-1e-2, 1e-2, X.shape[1])
    max_iter = 10000
    prev_error = 100

    for iteration in range(0, max_iter):

        sum_factor = r - 1.0 / (1.0 + np.exp(-(np.matmul(wjs, np.transpose(X)))))
        dwjs = np.matmul(sum_factor, X)
        wjs = wjs + alpha * dwjs

        sigmoid = 1.0 / (1.0 + np.exp(-(np.matmul(wjs, np.transpose(X)))))
        error_training = -np.sum(r * np.log(sigmoid) + (1 - r) * np.log(1-sigmoid))
        error_change = np.abs((error_training-prev_error)/prev_error)
        prev_error = error_training
        #print([iteration, error_change])
        if error_change < 1e-4:
            print('converged after ' + str(iteration) + ' iterations')
            return wjs
    print('gradient descent unable to converge')
    return wjs
