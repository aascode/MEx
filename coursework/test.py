import mex
from keras.utils import np_utils
import util

path = '/Users/anjanawijekoon/MEx_wtpm/'

# read all data
all_data = mex.read_all(path)
# extract windows from all three sensors
all_features = mex.extract_features(all_data)

# get features by sensor index
acw_features = util.get_features(all_features, 0)
act_features = util.get_features(all_features, 1)
pm_features = util.get_features(all_features, 2)

# get all people ids
all_people = all_features.keys()

# pm
# to make sure all windows have same length
padded_pm_features = util.pad_features(pm_features)
# to reduce the frame rate to mex.frames_per_second rate
reduced_pm_features = util.frame_reduce(padded_pm_features)

for i in range(len(all_people)):
    test_persons = [all_people[i]]
    pm_train_features, pm_test_features = util.train_test_split(reduced_pm_features, test_persons)

    pm_train_features, pm_train_labels = util.flatten(pm_train_features)
    pm_test_features, pm_test_labels = util.flatten(pm_test_features)

    pm_train_labels = np_utils.to_categorical(pm_train_labels, len(mex.activity_list))
    pm_test_labels = np_utils.to_categorical(pm_test_labels, len(mex.activity_list))

    util.run_model_2D(pm_train_features, pm_train_labels, pm_test_features, pm_test_labels)

# acw
for i in range(len(all_people)):
    test_persons = [all_people[i]]
    acw_train_features, acw_test_features = util.train_test_split(acw_features, test_persons)

    acw_train_features, acw_train_labels = util.flatten(acw_train_features)
    acw_test_features, acw_test_labels = util.flatten(acw_test_features)

    util.run_knn_model(acw_train_features, acw_train_labels, acw_test_features, acw_test_labels)

# act
for i in range(len(all_people)):
    test_persons = [all_people[i]]
    act_train_features, act_test_features = util.train_test_split(act_features, test_persons)

    act_train_features, act_train_labels = util.flatten(act_train_features)
    act_test_features, act_test_labels = util.flatten(act_test_features)

    util.run_svm_model(act_train_features, act_train_labels, act_test_features, act_test_labels)

