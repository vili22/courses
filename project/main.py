import logisticlinearclassifier as logclassifier

filename_train_data = '/home/vvirkkal/Documents/kurssit/machine_leaning_basic_principles_2017/project_work/python/data/train_data.csv'
filename_train_labels = '/home/vvirkkal/Documents/kurssit/machine_leaning_basic_principles_2017/project_work/python/data/train_labels.csv'
competition_data = '/home/vvirkkal/Documents/kurssit/machine_leaning_basic_principles_2017/project_work/python/data/test_data.csv'
accuracy_result = '/home/vvirkkal/Documents/kurssit/machine_leaning_basic_principles_2017/project_work/python/data/accuracy_testtt.csv'
logloss_result = '/home/vvirkkal/Documents/kurssit/machine_leaning_basic_principles_2017/project_work/python/data/logloss_testt.csv'

logclassifier.LogisticLinearClassifier.run_training(filename_train_data, filename_train_labels, 10)

#logclassifier.LogisticLinearClassifier.run_test(filename_train_data, filename_train_labels, competition_data, accuracy_result, logloss_result, 10)
