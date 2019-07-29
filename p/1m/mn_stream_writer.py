import csv
import datetime as dt
import numpy as np
import tensorflow as tf
import os
import keras
import random
from tensorflow import set_random_seed

random.seed(0)
np.random.seed(1)

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

path = '/Volumes/1708903/MEx/Data/pm_scaled/1.0/'
# path = '/home/mex/data/pm_1.0/'

# train size per person, samples per class, classes per set, feature length, window, increment, frames per second
pm_mn_stream_folder = '/Volumes/1708903/MEx/Data/p_pm_mn_stream_500_5_7_720_5_2_1'
# pm_mn_stream_folder = '/home/mex/data/p_pm_mn_stream_500_5_7_720_5_2_1'
pm_mn_stream_y_file = 'target_y.csv'

height = 32
width = 16
frame_size = height * width
samples_per_class = 5
classes_per_set = len(activity_list)
embedding_length = 720

number_of_users = 30
train_size = 500
frames_per_second = 1
window = 5
increment = 2

test_user_fold = [['01', '02', '03', '04', '05'],
                  ['06', '07', '08', '09', '10'],
                  ['11', '12', '13', '14', '15'],
                  ['16', '17', '18', '19', '20'],
                  ['21', '22', '23', '24', '25'],
                  ['26', '27', '28', '29', '30']]

# test_user_fold = [['21'], ['22'], ['23'], ['24'], ['25']]

pm_min_length = 14 * window
pm_max_length = 15 * window
dc_min_length = 10 * window
dc_max_length = 15 * window
ac_min_length = 95 * window
ac_max_length = 100 * window


def _read(_file):
    reader = csv.reader(open(_file, "r"), delimiter=",")
    _data = []
    for row in reader:
        if len(row[0]) == 19 and '.' not in row[0]:
            row[0] = row[0] + '.000000'
        temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
        _temp = [float(f) for f in row[1:]]
        temp.extend(_temp)
        _data.append(temp)
    return _data


def read():
    alldata = {}
    subjects = os.listdir(path)
    for subject in subjects:
        allactivities = {}
        subject_path = os.path.join(path, subject)
        activities = os.listdir(subject_path)
        for activity in activities:
            sensor = activity.split('.')[0].replace('_pm', '')
            activity_id = sensor.split('_')[0]
            _data = _read(os.path.join(subject_path, activity), )
            if activity_id in allactivities:
                allactivities[activity_id][sensor] = _data
            else:
                allactivities[activity_id] = {}
                allactivities[activity_id][sensor] = _data
        alldata[subject] = allactivities
    return alldata


def find_index(_data, _time_stamp):
    return [_index for _index, _item in enumerate(_data) if _item[0] >= _time_stamp][0]


def trim(_data):
    _length = len(_data)
    _inc = _length / (window * frames_per_second)
    _new_data = []
    for i in range(window * frames_per_second):
        _new_data.append(_data[i * _inc])
    return _new_data


def frame_reduce(_data):
    if frames_per_second == 0:
        return _data
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            time_windows = []
            for item in activity_data:
                time_windows.append(trim(item))
            _activities[activity] = time_windows
        _features[subject] = _activities
    return _features


def split_windows(data):
    outputs = []
    start = data[0][0]
    end = data[len(data) - 1][0]
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window)

    frames = [a[1:] for a in data[:]]
    frames = np.array(frames)
    _length = frames.shape[0]
    frames = np.reshape(frames, (_length * frame_size))
    frames = frames / max(frames)
    frames = [float("{0:.5f}".format(f)) for f in frames.tolist()]
    frames = np.reshape(np.array(frames), (_length, frame_size))

    while start + _window < end:
        _end = start + _window
        start_index = find_index(data, start)
        end_index = find_index(data, _end)
        instances = [a[:] for a in frames[start_index:end_index]]
        start = start + _increment
        outputs.append(instances)
    return outputs


# single sensor
def extract_features(_data):
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            time_windows = []
            activity_id = activity_id_dict.get(activity)
            activity_data = activities[activity]
            for sensor in activity_data:
                time_windows.extend(split_windows(activity_data[sensor]))
            _activities[activity_id] = time_windows
        _features[subject] = _activities
    return _features


def split(_data, _labels, test_indices):
    _train_data = []
    _train_labels = []
    _test_data = []
    _test_labels = []
    index = 0
    for _datum, _label in zip(_data, _labels):
        if index in test_indices:
            _test_data.append(_datum)
            _test_labels.append(_label)
        else:
            _train_data.append(_datum)
            _train_labels.append(_label)
        index += 1
    return _train_data, _train_labels, _test_data, _test_labels


def train_test_split(user_data, test_ids):
    train_data = {key: value for key, value in user_data.items() if key not in test_ids}
    test_data = {key: value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data


def flatten(_data):
    flatten_data = []
    flatten_labels = []

    for subject in _data:
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            flatten_data.extend(activity_data)
            flatten_labels.extend([activity for i in range(len(activity_data))])
    return flatten_data, flatten_labels


def pad(data, length):
    pad_length = []
    if length % 2 == 0:
        pad_length = [int(length / 2), int(length / 2)]
    else:
        pad_length = [int(length / 2) + 1, int(length / 2)]
    new_data = []
    for index in range(pad_length[0]):
        new_data.append(data[0])
    new_data.extend(data)
    for index in range(pad_length[1]):
        new_data.append(data[len(data) - 1])
    return new_data


def reduce(data, length):
    red_length = []
    if length % 2 == 0:
        red_length = [int(length / 2), int(length / 2)]
    else:
        red_length = [int(length / 2) + 1, int(length / 2)]
    new_data = data[red_length[0]:len(data) - red_length[1]]
    return new_data


def pad_features(_features):
    new_features = {}
    for subject in _features:
        new_activities = {}
        activities = _features[subject]
        for act in activities:
            items = activities[act]
            new_items = []
            for item in items:
                _len = len(item)
                if _len < pm_min_length:
                    continue
                elif _len > pm_max_length:
                    item = reduce(item, _len - pm_max_length)
                    new_items.append(item)
                elif _len < pm_max_length:
                    item = pad(item, pm_max_length - _len)
                    new_items.append(item)
            new_activities[act] = new_items
        new_features[subject] = new_activities
    return new_features


class Writer():

    def __init__(self):
        self.train_count = 0
        self.test_count = 0

    def write_data(self, file_path, data):
        if os.path.isfile(file_path):
            f = open(file_path, 'a')
            f.write(data + '\n')
        else:
            f = open(file_path, 'w')
            f.write(data + '\n')
        f.close()

    def write_slice_x(self, _slice_x, _file, fold):
        _slice_x = np.reshape(_slice_x, (((samples_per_class * classes_per_set) + 1) * height * width * window * 1))
        # print(_slice_x.shape)
        _file_ = os.path.join(pm_mn_stream_folder, fold)
        if not os.path.exists(_file_):
            os.makedirs(_file_)
        _file_ = os.path.join(_file_, str(_file) + '.csv')
        self.write_data(_file_, ','.join([str(f) for f in _slice_x.tolist()]))

    def write_slice_y(self, _slice_y, _file, fold):
        _slice_y = np.reshape(_slice_y, (samples_per_class * classes_per_set))
        # print(_slice_y.shape)
        _file_ = os.path.join(pm_mn_stream_folder, fold)
        if not os.path.exists(_file_):
            os.makedirs(_file_)
        _file_ = os.path.join(_file_, str(_file) + '.csv')
        self.write_data(_file_, ','.join([str(f) for f in _slice_y.tolist()]))

    def write_target_y(self, _target_y, fold):
        _file_ = os.path.join(pm_mn_stream_folder, fold)
        if not os.path.exists(_file_):
            os.makedirs(_file_)
        _file_ = os.path.join(_file_, str(pm_mn_stream_y_file))
        self.write_data(_file_, str(_target_y))

    def support_set_split(self, _test_features):
        support_set = {}
        everything_else = {}
        for user, labels in _test_features.items():
            _support_set = {}
            _everything_else = {}
            for label, data in labels.items():
                while len(data) < samples_per_class:
                    data.append(data[len(data)-1])
                support_set_indexes = np.random.choice(range(len(data)), samples_per_class, False)
                __support_set = [d for index, d in enumerate(data) if index in support_set_indexes]
                __everything_else = [d for index, d in enumerate(data) if index not in support_set_indexes]
                _support_set[label] = __support_set
                _everything_else[label] = __everything_else
            support_set[user] = _support_set
            everything_else[user] = _everything_else
        return support_set, everything_else

    def packslice(self, data_set, fold):
        n_samples = samples_per_class * classes_per_set
        for itr in range(train_size):
            slice_x = np.zeros((n_samples + 1, height, width * window * frames_per_second, 1))
            slice_y = np.zeros((n_samples,))

            ind = 0
            pinds = np.random.permutation(n_samples)

            hat_classes = [key for key, value in data_set.items() if len(value) > samples_per_class]

            x_hat_class_index = np.random.randint(len(hat_classes))
            x_hat_class = hat_classes[x_hat_class_index]
            _classes = np.random.choice(list(data_set.keys()), classes_per_set, False)

            for j, cur_class in enumerate(_classes):
                data_pack = data_set[cur_class]
                new_data_pack = []
                new_data_pack.extend(data_pack)
                while len(new_data_pack) < samples_per_class:
                    new_data_pack.append(data_pack[len(data_pack)-1])
                data_pack = new_data_pack
                data_pack = np.array(data_pack)
                data_pack = np.reshape(data_pack, (data_pack.shape[0], data_pack.shape[1], height, width))
                data_pack = np.swapaxes(data_pack, 1, 2)
                data_pack = np.swapaxes(data_pack, 2, 3)
                data_pack = np.reshape(data_pack, (data_pack.shape[0], data_pack.shape[1],
                                                   data_pack.shape[2] * data_pack.shape[3]))
                data_pack = np.expand_dims(data_pack, 4)
                example_inds = np.random.choice(len(data_pack), samples_per_class, False)

                for eind in example_inds:
                    slice_x[pinds[ind], :] = data_pack[eind, :]
                    slice_y[pinds[ind]] = cur_class
                    ind += 1

                if cur_class == x_hat_class:
                    target_indx = np.random.choice(len(data_pack))
                    while target_indx in example_inds:
                        target_indx = np.random.choice(len(data_pack))
                    slice_x[n_samples, :] = data_pack[target_indx, :]
                    target_y = cur_class

            self.write_slice_x(slice_x, self.train_count, fold)
            self.write_slice_y(slice_y, self.train_count, fold)
            if self.train_count == 1999:
                print('here')
            self.train_count = self.train_count + 1
            self.write_target_y(target_y, fold)

    def create_train_instances(self, _train_features, fold):
        for user_id, train_feats in _train_features.items():
            self.packslice(train_feats, fold)

    def packslice_test(self, data_set, support_set, fold):
        n_samples = samples_per_class * classes_per_set
        support_cacheX = []
        support_cacheY = []
        target_cacheY = []
        count = 0
        support_X = np.zeros((n_samples, height, width * window * frames_per_second, 1))
        support_y = np.zeros((n_samples,))
        for i, _class in enumerate(support_set.keys()):
            X = support_set[_class]
            X = np.array(X)
            X = np.reshape(X, (X.shape[0], X.shape[1], height, width))
            X = np.swapaxes(X, 1, 2)
            X = np.swapaxes(X, 2, 3)
            X = np.reshape(X, (X.shape[0], X.shape[1],
                               X.shape[2] * X.shape[3]))
            X = np.expand_dims(X, 4)
            for j in range(len(X)):
                support_X[(i * samples_per_class) + j, :] = X[j, :]
                support_y[(i * samples_per_class) + j] = _class

        for _class in data_set:
            X = data_set[_class]
            X = np.array(X)
            X = np.reshape(X, (X.shape[0], X.shape[1], height, width))
            X = np.swapaxes(X, 1, 2)
            X = np.swapaxes(X, 2, 3)
            X = np.reshape(X, (X.shape[0], X.shape[1],
                               X.shape[2] * X.shape[3]))
            X = np.expand_dims(X, 4)
            for inde in range(len(X)):
                slice_x = np.zeros((n_samples + 1, height, width * window * frames_per_second, 1))
                slice_y = np.zeros((n_samples,))

                slice_x[:n_samples, :] = support_X[:]
                slice_x[n_samples, :] = X[inde, :]

                slice_y[:n_samples] = support_y[:]

                target_y = _class

                self.write_slice_x(slice_x, count, fold)
                self.write_slice_y(slice_y, count, fold)
                count = count + 1
                self.write_target_y(target_y, fold)
        return count

    def create_test_instance(self, test_set, support_set, fold):
        for user_id, _test_set in test_set.items():
            support_data = support_set[user_id]
            self.packslice_test(_test_set, support_data, fold)

    def run_model_mn(self, fold, _train_features, _test_features):
        self.create_train_instances(_train_features, str(fold) + '/train')

        test_support_set, _test_features = self.support_set_split(_test_features)
        self.create_test_instance(_test_features, test_support_set, str(fold) + '/test')


all_data = read()
all_features = extract_features(all_data)
all_data = None
all_features = pad_features(all_features)
all_features = frame_reduce(all_features)

for i in range(len(test_user_fold)):
    set_random_seed(2)
    train_features, test_features = train_test_split(all_features, test_user_fold[i])
    writer = Writer()
    writer.run_model_mn(i, train_features, test_features)
