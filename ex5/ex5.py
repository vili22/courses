import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def read_data(filename):
    with open(filename) as f:
        array2d = [[float(digit) for digit in line.split(',')] for line in f]
        return np.array(array2d)

train_data = read_data('/home/vvirkkal/Documents/kurssit/machine_learning_basic_principles/ex5/spect_training.txt')
test_data = read_data('/home/vvirkkal/Documents/kurssit/machine_learning_basic_principles/ex5/spect_test.txt')

r = train_data[:, 0]
features = train_data[:, 1:(train_data.shape[1])]

r_test = test_data[:, 0];
features_test = test_data[:, 1:(test_data.shape[1])]

p0j = np.ones(features.shape[1])
p1j = np.ones(features.shape[1])

p1j_divisor = 2 + np.sum(r)
p0j_divisor = 2 + r.size - p1j_divisor

pr = p1j_divisor/r.size

for index1 in range(0, p0j.size):
    for index2 in range(0, r.size):

        factor = 1 if r[index2] == 1 else 0
        p0j[index1] += (1 - factor) * features[index2, index1]
        p1j[index1] += factor * features[index2, index1]

    p0j[index1] = p0j[index1] / p0j_divisor
    p1j[index1] = p1j[index1] / p1j_divisor

wj = np.zeros(p0j.size)

for index in range(0, p0j.size):
    wj[index] = np.log(p0j[index]) - np.log(1-p0j[index]) - np.log(p1j[index]) + np.log(1-p1j[index])

w0 = np.sum(np.log(1 - p0j)) + np.log(1 - pr) - np.sum(np.log(1 - p1j)) - np.log(pr)

px1_train = 1/(1 + np.exp(w0 + np.matmul(wj, np.transpose(features))))
estimate_train = np.rint(px1_train)
train_error = np.linalg.norm(r-estimate_train)**2

px1_test = 1/(1 + np.exp(w0 + np.matmul(wj, np.transpose(features_test))))
estimate_test = np.rint(px1_test)
test_error = np.linalg.norm(r_test-estimate_test)**2

print [(r.size - np.sum(np.abs(r - estimate_train)))/r.size, train_error]
print [(r_test.size - np.sum(np.abs(r_test - estimate_test)))/r_test.size, test_error]
print np.abs(r_test - estimate_test)
print [np.sum(np.abs(r - estimate_train)), r.size]
print [np.sum(np.abs(r_test - estimate_test)), r_test.size]
# features = train_data[:, 1:train_data.]

# train_data_sizes = [100, 200, 500]

# for train_data_size in train_data_sizes:
#
#     train_x = plasma_clucose[0:train_data_size]
#     train_class = true_class[0:train_data_size]
#
#     validation_x = plasma_clucose[train_data_size:(plasma_clucose.size - 1)]
#     validation_class = true_class[train_data_size:(true_class.size - 1)]
#
#     positive_indices = np.where(train_class == 1)
#     negative_indices = np.where(train_class == 0)
#
#     positive_train_x = train_x[positive_indices].astype(float)
#     negative_train_x = train_x[negative_indices].astype(float)
#
#     N1 = float(positive_train_x.size)
#     N2 = float(negative_train_x.size)
#     p1 = N1/(N1 + N2)
#     p2 = N2/(N1 + N2)
#
#     mu1 = float(np.sum(positive_train_x))/N1
#     mu2 = float(np.sum(negative_train_x))/N2
#
#     sigma1 = np.sqrt(np.sum(np.power(positive_train_x - mu1, 2))/N1)
#     sigma2 = np.sqrt(np.sum(np.power(negative_train_x - mu2, 2))/N2)
#
#     posterior_positive = 1/np.sqrt(2 * np.pi)/sigma1*np.exp(-np.power(validation_x - mu1, 2) / 2 / (sigma1**2)) * p1
#     posterior_negative = 1 / np.sqrt(2 * np.pi) / sigma2 * np.exp(-np.power(validation_x - mu2, 2) / 2 / (sigma2 ** 2)) * p2
#
#     positive_classification = np.where((posterior_positive - posterior_negative) >= 0)
#     negative_classification = np.where((posterior_positive - posterior_negative) < 0)
#
#     true_positives = np.where(validation_class[positive_classification] == 1)[0].size
#     true_negatives = np.where(validation_class[negative_classification] == 0)[0].size
#
#     accuracy = (float(true_positives) + float(true_negatives))/float(validation_class.size)
#     print true_positives, true_negatives, validation_class.size, 'accuracy is ' + str(accuracy)
#     print [p1, mu1, sigma1, p2, mu2, sigma2]
#     n, bins, patches = plt.hist(negative_train_x, 25, normed=1, facecolor='green', alpha=0.75)
#     y = mlab.normpdf(bins, mu1, sigma1)
#     plt.plot(bins, y, 'r--', linewidth=1)
#     plt.show()


