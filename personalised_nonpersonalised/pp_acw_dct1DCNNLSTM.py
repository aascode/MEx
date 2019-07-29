import os
import csv
import datetime as dt
import numpy as np
import sklearn.metrics as metrics
from keras.layers import Input, Dense, BatchNormalization, Conv1D, MaxPooling1D, LSTM, TimeDistributed, Reshape
from keras.models import Model
import pandas as pd
import random
import keras.backend as K
from scipy import fftpack
from keras.utils import np_utils
from tensorflow import set_random_seed
import sys

random.seed(0)
np.random.seed(1)

frame_size = 3*1

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

path = '/Volumes/1708903/MEx/Data/acw/'
results_file = '/Volumes/1708903/MEx/results/p_vs_np/pp_acw.csv'

frames_per_second = 100
window = 5
increment = 2
dct_length = 60
feature_length = dct_length * 3

ac_min_length = 95*window
ac_max_length = 100*window

user_index = int(sys.argv[1])


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
    for index, subject in enumerate(subjects):
        if index == user_index:
            allactivities = {}
            subject_path = os.path.join(path, subject)
            activities = os.listdir(subject_path)
            for activity in activities:
                sensor = activity.split('.')[0].replace('_act', '')
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


def dct(data):
    new_data = []
    data = np.array(data)
    print(data.shape)
    data = np.reshape(data, (data.shape[0], window, frames_per_second, 3))
    print(data.shape)
    for item in data:
        new_item = []
        for i in range(item.shape[0]):
            if dct_length > 0:
                x = [t[0] for t in item[i]]
                y = [t[1] for t in item[i]]
                z = [t[2] for t in item[i]]

                dct_x = np.abs(fftpack.dct(x, norm='ortho'))
                dct_y = np.abs(fftpack.dct(y, norm='ortho'))
                dct_z = np.abs(fftpack.dct(z, norm='ortho'))

                v = np.array([])
                v = np.concatenate((v, dct_x[:dct_length]))
                v = np.concatenate((v, dct_y[:dct_length]))
                v = np.concatenate((v, dct_z[:dct_length]))
                new_item.append(v)
        new_data.append(new_item)
    return new_data


def flatten(_data):
    flatten_data = []
    flatten_labels = []

    for subject in _data:
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            flatten_data.extend(activity_data)
            flatten_labels.extend([activity for i in range(len(activity_data))])
    return dct(flatten_data), flatten_labels


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
                if _len < ac_min_length:
                    continue
                elif _len > ac_max_length:
                    item = reduce(item, _len - ac_max_length)
                    new_items.append(item)
                elif _len < ac_max_length:
                    item = pad(item, ac_max_length - _len)
                    new_items.append(item)
            new_activities[act] = new_items
        new_features[subject] = new_activities
    return new_features


def build_1D_model():
    _input = Input(shape=(window, feature_length, 1))
    x = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(_input)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv1D(64, kernel_size=5, activation='relu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv1D(128, kernel_size=5, activation='relu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Reshape((K.int_shape(x)[1], K.int_shape(x)[2]*K.int_shape(x)[3]))(x)
    x = LSTM(600)(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(len(activity_list), activation='softmax')(x)

    model = Model(inputs=_input, outputs=x)
    model.summary()
    return model


def _run_(_train_features, _train_labels, _test_features, _test_labels):
    _train_features = np.array(_train_features)
    print(_train_features.shape)

    _test_features = np.array(_test_features)
    print(_test_features.shape)

    _train_features = np.expand_dims(_train_features, 3)
    print(_test_features.shape)
    _test_features = np.expand_dims(_test_features, 3)
    print(_test_features.shape)

    model = build_1D_model()
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(_train_features, _train_labels, verbose=0, batch_size=16, epochs=50, shuffle=True)
    _predict_labels = model.predict(_test_features, batch_size=64, verbose=0)
    f_score = metrics.f1_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1), average='macro')
    accuracy = metrics.accuracy_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1))
    results = str(accuracy)+',' + str(f_score)
    print(results)
    write_data(results_file, str(results))

    _test_labels = pd.Series(_test_labels.argmax(axis=1), name='Actual')
    _predict_labels = pd.Series(_predict_labels.argmax(axis=1), name='Predicted')
    df_confusion = pd.crosstab(_test_labels, _predict_labels)
    print(df_confusion)
    write_data(results_file, str(df_confusion))


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
        train_labels[i] = np_utils.to_categorical(train_labels[i], len(activity_list))
        test_labels[i] = np_utils.to_categorical(test_labels[i], len(activity_list))
        set_random_seed(2)
        _run_(train_features[i], train_labels[i], test_features[i], test_labels[i])

run()
