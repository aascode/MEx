import csv
import datetime as dt
import numpy as np
import tensorflow as tf
import os
import keras
import random

random.seed(0)
np.random.seed(1)

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

#path = '/Volumes/1708903/MEx/Data/pm_scaled/1.0/'
path = '/home/mex/data/pm_1.0/'
#results_file = '/home/mex/results/np_pm_1.0.csv'

# train size per person, samples per class, classes per set, feature length, window, increment, frames per second
#pm_mn_stream_folder = '/Volumes/1708903/MEx/Data/pm_mn_stream_500_5_7_720_5_2_1'
pm_mn_stream_folder = '/home/mex/data/np_pm_mn_stream_500_5_7_720_5_2_1'
pm_mn_stream_y_file = 'target_y.csv'

height = 32
width = 16
frame_size = height*width
samples_per_class = 5
classes_per_set = len(activity_list)
embedding_length = 720

number_of_users = 5
train_size = 500 * number_of_users
frames_per_second = 1
window = 5
increment = 2

pm_min_length = 14*window
pm_max_length = 15*window
dc_min_length = 10*window
dc_max_length = 15*window
ac_min_length = 95*window
ac_max_length = 100*window


def write_data(file_path, data):
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


def _read(_file):
    reader = csv.reader(open(_file, "r"), delimiter=",")
    _data = []
    for row in reader:
        if len(row[0]) == 19 and '.' not in row[0]:
            row[0] = row[0]+'.000000'
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
    _inc = _length/(window*frames_per_second)
    _new_data = []
    for i in range(window*frames_per_second):
        _new_data.append(_data[i*_inc])
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
    frames = np.reshape(frames, (_length*frame_size))
    frames = frames/max(frames)
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


def train_test_split(_data, _labels):
    indices = range(len(_data))
    random.shuffle(indices)
    split_length = int(len(_data)/6)
    test_indices_1 = indices[0:split_length]
    test_indices_2 = indices[split_length:split_length*2]
    test_indices_3 = indices[split_length*2:split_length*3]
    test_indices_4 = indices[split_length*3:split_length*4]
    test_indices_5 = indices[split_length*4:split_length*5]
    test_indices_6 = indices[split_length*5:split_length*6]

    _train_data_1, _train_labels_1, _test_data_1, _test_labels_1 = split(_data, _labels, test_indices_1)
    _train_data_2, _train_labels_2, _test_data_2, _test_labels_2 = split(_data, _labels, test_indices_2)
    _train_data_3, _train_labels_3, _test_data_3, _test_labels_3 = split(_data, _labels, test_indices_3)
    _train_data_4, _train_labels_4, _test_data_4, _test_labels_4 = split(_data, _labels, test_indices_4)
    _train_data_5, _train_labels_5, _test_data_5, _test_labels_5 = split(_data, _labels, test_indices_5)
    _train_data_6, _train_labels_6, _test_data_6, _test_labels_6 = split(_data, _labels, test_indices_6)

    return [[_train_data_1, _train_data_2, _train_data_3, _train_data_4, _train_data_5, _train_data_6],
            [_train_labels_1, _train_labels_2,_train_labels_3, _train_labels_4, _train_labels_5, _train_labels_6],
            [_test_data_1, _test_data_2, _test_data_3, _test_data_4, _test_data_5, _test_data_6],
            [_test_labels_1, _test_labels_2, _test_labels_3, _test_labels_4, _test_labels_5, _test_labels_6]]


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


def write_slice_x(_slice_x, _file, fold):
    _slice_x = np.reshape(_slice_x, (((samples_per_class*classes_per_set)+1) * height * width*window * 1))
    #print(_slice_x.shape)
    _file_ = os.path.join(pm_mn_stream_folder, fold)
    if not os.path.exists(_file_):
        os.makedirs(_file_)
    _file_ = os.path.join(_file_, str(_file)+'.csv')
    write_data(_file_, ','.join([str(f) for f in _slice_x.tolist()]))


def write_slice_y(_slice_y, _file, fold):
    _slice_y = np.reshape(_slice_y, (samples_per_class*classes_per_set))
    #print(_slice_y.shape)
    _file_ = os.path.join(pm_mn_stream_folder, fold)
    if not os.path.exists(_file_):
        os.makedirs(_file_)
    _file_ = os.path.join(_file_, str(_file)+'.csv')
    write_data(_file_, ','.join([str(f) for f in _slice_y.tolist()]))


def write_target_y(_target_y, fold):
    _file_ = os.path.join(pm_mn_stream_folder, fold)
    if not os.path.exists(_file_):
        os.makedirs(_file_)
    _file_ = os.path.join(_file_, str(pm_mn_stream_y_file))
    write_data(_file_, str(_target_y))


def support_set_split(test_features, test_labels):
    support_set = {}
    test_set = {}
    _values = {}
    for _val, _label in zip(test_features, test_labels):
        if _label in _values:
            _values[_label].append(_val)
        else:
            _values[_label] = [_val]

    for _class, _val in _values.items():
        support_set_indexes = np.random.choice(range(len(_val)), samples_per_class, False)
        X = [x for index, x in enumerate(_val) if index not in support_set_indexes]
        ss = [x for index, x in enumerate(_val) if index in support_set_indexes]
        support_set[_class] = ss
        test_set[_class] = X
    return support_set, test_set


def packslice(data_set, fold):
    n_samples = samples_per_class * classes_per_set
    support_cacheX = []
    support_cacheY = []
    target_cacheY = []
    count = 0
    for itr in range(train_size):
        slice_x = np.zeros((n_samples + 1, height, width * window * frames_per_second, 1))
        slice_y = np.zeros((n_samples,))

        ind = 0
        pinds = np.random.permutation(n_samples)

        x_hat_class = np.random.randint(classes_per_set)
        _classes = np.random.choice(list(data_set.keys()), classes_per_set, False)

        for j, cur_class in enumerate(_classes):
            data_pack = data_set[cur_class]
            example_inds = np.random.choice(len(data_pack), samples_per_class, False)

            for eind in example_inds:
                slice_x[pinds[ind], :] = data_pack[eind]
                slice_y[pinds[ind]] = cur_class
                ind += 1

            if j == x_hat_class:
                target_indx = np.random.choice(len(data_pack))
                while target_indx in example_inds:
                    target_indx = np.random.choice(len(data_pack))
                slice_x[n_samples, :] = data_pack[target_indx]
                target_y = cur_class

        write_slice_x(slice_x, count, fold)
        write_slice_y(slice_y, count, fold)
        count = count + 1
        write_target_y(target_y, fold)


def create_train_instances(train_features, train_labels, fold):
    _train_feats = {}
    for _feats, _label in zip(train_features, train_labels):
        _label_stuff = []
        if _label in _train_feats:
            _label_stuff = _train_feats[_label]
            _label_stuff.append(_feats)
        else:
            _label_stuff.append(_feats)
        _train_feats[_label] = _label_stuff

    packslice(_train_feats, fold)


def packslice_test(data_set, support_set, fold):
    n_samples = samples_per_class * classes_per_set
    support_cacheX = []
    support_cacheY = []
    target_cacheY = []
    count = 0
    support_X = np.zeros((n_samples, height, width * window * frames_per_second, 1))
    support_y = np.zeros((n_samples,))
    for i, _class in enumerate(support_set.keys()):
        X = support_set[_class]
        for j in range(len(X)):
            support_X[(i * samples_per_class) + j, :] = X[j]
            support_y[(i * samples_per_class) + j] = _class

    for _class in data_set:
        X = data_set[_class]
        for iiii in range(len(X)):
            slice_x = np.zeros((n_samples + 1, height, width * window * frames_per_second, 1))
            slice_y = np.zeros((n_samples,))

            slice_x[:n_samples, :] = support_X[:]
            slice_x[n_samples, :] = X[iiii]

            slice_y[:n_samples] = support_y[:]

            target_y = _class

            write_slice_x(slice_x, count, fold)
            write_slice_y(slice_y, count, fold)
            count = count + 1
            write_target_y(target_y, fold)


def create_test_instance(test_set, support_set, fold):
    packslice_test(test_set, support_set, fold)


def run_model_mn(fold, _train_features, _train_labels, _test_features, _test_labels):
    _train_features = np.array(_train_features)
    _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1], height, width))
    _train_features = np.swapaxes(_train_features, 1, 2)
    _train_features = np.swapaxes(_train_features, 2, 3)
    _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1],
                                                   _train_features.shape[2] * _train_features.shape[3]))
    _train_features = np.expand_dims(_train_features, 4)
    #print(_train_features.shape)

    _test_features = np.array(_test_features)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], _test_features.shape[1], height, width))
    _test_features = np.swapaxes(_test_features, 1, 2)
    _test_features = np.swapaxes(_test_features, 2, 3)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], _test_features.shape[1],
                                                 _test_features.shape[2] * _test_features.shape[3]))
    _test_features = np.expand_dims(_test_features, 4)
    #print(_test_features.shape)

    create_train_instances(_train_features, _train_labels, str(fold)+'/train')

    test_support_set, _test_features = support_set_split(_test_features, _test_labels)
    create_test_instance(_test_features, test_support_set, str(fold)+'/test')


def run():
    all_data = read()
    all_features = extract_features(all_data)
    all_data = None
    all_features = pad_features(all_features)
    all_features = frame_reduce(all_features)

    all_features, all_labels = flatten(all_features)

    all_split = train_test_split(all_features, all_labels)
    train_features, train_labels, test_features, test_labels = all_split[0], all_split[1], all_split[2], all_split[3]

    for i in range(len(train_features)):
        run_model_mn(i, train_features[i], train_labels[i], test_features[i], test_labels[i])


run()