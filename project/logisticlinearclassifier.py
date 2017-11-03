import gradientdescent as gd
import numpy as np
import fileoperations as fo

class LogisticLinearClassifier:
    def __init__(self, data, labels):
        self.w = gd.gradient_descent(labels, data, 0.000015)

    def calc_probabilities(self, data):
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        data = np.c_[np.ones([data.shape[0], 1]), data]
        return 1.0 / (1.0 + np.exp(-(np.matmul(self.w, np.transpose(data)))))

    def calc_accuracy(self, data, labels):
        probabilities = self.calc_probabilities(data)
        indices = np.zeros([labels.shape[0], ])
        indices[np.where(probabilities >= 0.5)[0]] = 1
        return 1.0 - np.sum(np.abs(labels - indices))/labels.shape[0]

    @staticmethod
    def run_training(filename_data, filename_labels, n_classes):
        train_data = fo.read_data(filename_data)
        train_labels = fo.read_data(filename_labels)

        test_indices = np.arange(0, train_data.shape[0], 3)
        test_data = train_data[test_indices, :]
        test_labels = train_labels[test_indices]

        train_data = np.delete(train_data, test_indices, axis=0)
        train_labels = np.delete(train_labels, test_indices)

        probabilities = np.zeros([test_data.shape[0], n_classes])
        for class_index_iter in range(0, n_classes):
            class_indices = np.where(train_labels == (class_index_iter + 1))[0]
            non_class_indices = np.where(train_labels != (class_index_iter + 1))[0]
            class_labels = np.zeros([(class_indices.size + non_class_indices.size), ])
            class_labels[class_indices] = 1.0
            class_labels[non_class_indices] = 0.0
            train_class_labels = LogisticLinearClassifier.create_class_indices(train_labels, class_index_iter)
            classifier = LogisticLinearClassifier(train_data, train_class_labels)
            test_class_labels = LogisticLinearClassifier.create_class_indices(test_labels, class_index_iter)
            accuracy = classifier.calc_accuracy(test_data, test_class_labels)
            print('accuracy is ' + str(accuracy))
            probabilities[:, class_index_iter] = classifier.calc_probabilities(test_data)

        evaluated_classes = np.argmax(probabilities, axis=1) + 1
        correct_classes_fraction = np.where((test_labels[:, 0] - evaluated_classes) == 0.0)[0].size/test_labels.shape[0]
        logloss = LogisticLinearClassifier.calc_loglogg(probabilities, test_labels[:, 0])
        print('accuracy is ' + str(correct_classes_fraction))
        print('logloss is' + str(logloss))

    @staticmethod
    def run_test(filename_train_data, filename_train_labels, filename_competition_data,
                 filename_accuracy_result, filename_logloss_result, n_classes):
        train_data = fo.read_data(filename_train_data)
        train_labels = fo.read_data(filename_train_labels)

        competition_data = fo.read_data(filename_competition_data)
        probabilities = np.zeros([competition_data.shape[0], n_classes])

        for class_index_iter in range(0, n_classes):
            train_class_labels = LogisticLinearClassifier.create_class_indices(train_labels, class_index_iter)
            classifier = LogisticLinearClassifier(train_data, train_class_labels)
            probabilities[:, class_index_iter] = classifier.calc_probabilities(competition_data)

        evaluated_classes = np.argmax(probabilities, axis=1) + 1
        indices = np.arange(1, competition_data.shape[0] + 1)

        accuracy_result = np.column_stack((indices, evaluated_classes))
        fo.write_csv_accuracy(filename_accuracy_result, ['Sample_id', 'Sample_label'], accuracy_result)

        logloss_result = np.column_stack((indices, probabilities))
        fo.write_csv_logloss(filename_logloss_result,
                             ['Sample_id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7',
                      'Class_8', 'Class_9', 'Class_10'], logloss_result)

    @staticmethod
    def calc_loglogg(probabilities, labels):

        logloss = 0
        for label_iter in range(0, labels.size):
            logloss = logloss - 1.0/float(labels.size)*np.log(probabilities[label_iter, labels[label_iter] - 1])

        return logloss

    @staticmethod
    def create_class_indices(labels, class_index):
        class_indices = np.where(labels == (class_index + 1))[0]
        non_class_indices = np.where(labels != (class_index + 1))[0]
        class_labels = np.zeros([(class_indices.size + non_class_indices.size), ])
        class_labels[class_indices] = 1.0
        class_labels[non_class_indices] = 0.0
        return class_labels
