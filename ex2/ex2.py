import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    with open(filename) as f:
        array2d = [[float(digit) for digit in line.split()] for line in f]
        return np.array(array2d)

train_data = read_data('/home/vvirkkal/Documents/kurssit/machine_learning_basic_principles/ex2/ex2_data/ex2_traindata.txt')
test_data = read_data('/home/vvirkkal/Documents/kurssit/machine_learning_basic_principles/ex2/ex2_data/ex2_testdata.txt')

train_data = train_data[train_data[:,0].argsort()]
x_train = train_data[:, 0]
y_train = train_data[:, 1]

x_validation = x_train[1::3]
y_validation = y_train[1::3]

x_train = np.delete(x_train, np.arange(1, x_train.size, 3))
y_train = np.delete(y_train, np.arange(1, y_train.size, 3))
print [train_data.size, x_train.size, x_validation.size]

test_data = test_data[test_data[:, 1].argsort()]
x_test = test_data[:, 0]
y_test = test_data[:, 1]

min_val = float('inf');
min_ind = 1;

for k in range(1,30):
    phi = np.vander(x_train, k +1 , increasing=True)
    w = np.dot(np.linalg.pinv(phi), y_train)
    y_interp = np.dot(np.vander(x_validation, k+1, increasing=True), w)

    norm = np.linalg.norm(y_validation - y_interp)**2
    print [k +1 , norm]
    if norm < min_val:
        min_ind = k
        min_val = norm

print [min_ind, min_val]
phi = np.vander(x_train, min_ind + 1, increasing=True)
w = np.dot(np.linalg.pinv(phi), y_train)
y_interp = np.dot(np.vander(x_test, min_ind + 1, increasing=True), w)

plt.plot(x_test, y_test, '.b')
plt.plot(x_test, y_interp, '.r')
plt.show()


