import os
import csv
import datetime as dt
import numpy as np
import sklearn.metrics as metrics
from keras.layers import Input, Dense, BatchNormalization, Conv1D, MaxPooling1D, LSTM, TimeDistributed, Reshape, Conv2D
from keras.layers import MaxPooling2D, Flatten
from keras.models import Model
import pandas as pd
import keras.backend as K
import random
from scipy import fftpack
from keras.utils import np_utils
from tensorflow import set_random_seed
random.seed(0)
np.random.seed(1)

frame_size = 3*1

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

act_path = '/Volumes/1708903/MEx/Data/act/'
acw_path = '/Volumes/1708903/MEx/Data/acw/'
pm_path = '/Volumes/1708903/MEx/Data/pm_scaled/pm_1.0_1.0'

results_file = '/Volumes/1708903/MEx/results/p_act_acw.csv'

frames_per_second = 100
window = 5
increment = 2
dct_length = 60
feature_length = dct_length * 3

test_user_fold = [['01', '02', '03', '04', '05'],
                  ['06', '07', '08', '09', '10'],
                  ['11', '12', '13', '14', '15'],
                  ['16', '17', '18', '19', '20'],
                  ['21', '22', '23', '24', '25'],
                  ['26', '27', '28', '29', '30']]

ac_min_length = 95*window
ac_max_length = 100*window
pm_min_length = 14*window
pm_max_length = 15*window


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


def read(path, sensor):
    alldata = {}
    subjects = os.listdir(path)
    for subject in subjects:
        allactivities = {}
        subject_path = os.path.join(path, subject)
        activities = os.listdir(subject_path)
        for activity in activities:
            sensor = activity.split('.')[0].replace(sensor, '')
            activity_id = sensor.split('_')[0]
            _data = _read(os.path.join(subject_path, activity), )
            if activity_id in allactivities:
                allactivities[activity_id][sensor] = _data
            else:
                allactivities[activity_id] = {}
                allactivities[activity_id][sensor] = _data
        alldata[subject] = allactivities
    return alldata


def build_2D_model():
    _input = Input(shape=(32, 16 * window * frames_per_second, 1))
    x = Conv2D(32, kernel_size=(1,5), activation='relu')(_input)
    x = MaxPooling2D(pool_size=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(1,5), activation='relu')(x)
    x = MaxPooling2D(pool_size=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(1,5), activation='relu')(x)
    x = MaxPooling2D(pool_size=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(len(activity_list), activation='softmax')(x)

    model = Model(inputs=_input, outputs=x)
    model.summary()
    return model


def _run_(_train_features, _train_labels, _test_features, _test_labels):
    _train_features = np.array(_train_features)
    print(_train_features.shape)

    _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1], 32, 16))
    _train_features = np.swapaxes(_train_features, 1, 2)
    _train_features = np.swapaxes(_train_features, 2, 3)
    _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1],
                                                   _train_features.shape[2] * _train_features.shape[3]))
    _train_features = np.expand_dims(_train_features, 4)
    print(_train_features.shape)

    _test_features = np.array(_test_features)
    print(_test_features.shape)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], _test_features.shape[1], 32, 16))
    _test_features = np.swapaxes(_test_features, 1, 2)
    _test_features = np.swapaxes(_test_features, 2, 3)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], _test_features.shape[1],
                                                 _test_features.shape[2] * _test_features.shape[3]))
    _test_features = np.expand_dims(_test_features, 4)
    print(_test_features.shape)

    model = build_2D_model()
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(_train_features, _train_labels, verbose=0, batch_size=32, epochs=100, shuffle=True)
    _predict_labels = model.predict(_test_features, batch_size=64, verbose=0)
    f_score = metrics.f1_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1), average='macro')
    accuracy = metrics.accuracy_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1))
    results = 'pm' + ',' + '2D' + ',' + str(accuracy)+',' + str(f_score)
    print(results)
    write_data(results_file, str(results))

    _test_labels = pd.Series(_test_labels.argmax(axis=1), name='Actual')
    _predict_labels = pd.Series(_predict_labels.argmax(axis=1), name='Predicted')
    df_confusion = pd.crosstab(_test_labels, _predict_labels)
    print(df_confusion)
    write_data(results_file, str(df_confusion))


def find_index(_data, _time_stamp):
    return [_index for _index, _item in enumerate(_data) if _item[0] >= _time_stamp][0]


def trim(_data):
    _length = len(_data)
    _inc = _length/(window*frames_per_second)
    _new_data = []
    for i in range(window*frames_per_second):
        _new_data.append(_data[i*_inc])
    return _new_data


def frame_reduce(_features):
    if frames_per_second == 0:
        return _features
    new_features = {}
    for subject in _features:
        _activities = {}
        activities = _features[subject]
        for activity in activities:
            activity_data = activities[activity]
            time_windows = []
            for item in activity_data:
                new_item = []
                new_item.append(trim(item[0]))
                new_item.append(trim(item[1]))
                new_item.append(trim(item[2]))
                time_windows.append(new_item)
            _activities[activity] = time_windows
        new_features[subject] = _activities
    return new_features


def split_windows(act_data, acw_data, pm_data):
    outputs = []
    start = act_data[0][0]
    end = act_data[len(act_data) - 1][0]
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window)

    act_frames = [a[1:] for a in act_data[:]]
    acw_frames = [a[1:] for a in acw_data[:]]
    pm_frames = [a[1:] for a in pm_data[:]]
    act_frames = np.array(act_frames)
    acw_frames = np.array(acw_frames)
    pm_frames = np.array(pm_frames)

    while start + _window < end:
        _end = start + _window
        act_start_index = find_index(act_frames, start)
        act_end_index = find_index(act_frames, _end)
        acw_start_index = find_index(acw_frames, start)
        acw_end_index = find_index(acw_frames, _end)
        pm_start_index = find_index(pm_frames, start)
        pm_end_index = find_index(pm_frames, _end)
        act_instances = [a[:] for a in act_frames[act_start_index:act_end_index]]
        acw_instances = [a[:] for a in acw_frames[acw_start_index:acw_end_index]]
        pm_instances = [a[:] for a in pm_frames[pm_start_index:pm_end_index]]
        start = start + _increment
        instances = [act_instances, acw_instances, pm_instances]
        outputs.append(instances)
    return outputs


def extract_features(act_data, acw_data, pm_data):
    _features = {}
    for subject in act_data:
        _activities = {}
        act_activities = act_data[subject]
        for act_activity in act_activities:
            time_windows = []
            activity_id = activity_id_dict.get(act_activity)
            act_activity_data = act_activities[act_activity]
            acw_activity_data = acw_data[subject][act_activity]
            pm_activity_data = pm_data[subject][act_activity]
            time_windows.extend(split_windows(act_activity_data, acw_activity_data, pm_activity_data))
            _activities[activity_id] = time_windows
        _features[subject] = _activities
    return _features


def train_test_split(user_data, test_ids):
    train_data = {key: value for key, value in user_data.items() if key not in test_ids}
    test_data = {key: value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data


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
                new_item = []
                act_len = len(item[0])
                acw_len = len(item[1])
                pm_len = len(item[2])
                if act_len < ac_min_length or acw_len < ac_min_length:
                    continue
                if act_len > ac_max_length:
                    new_item.append(reduce(item[0], act_len - ac_max_length))
                elif act_len < ac_max_length:
                    new_item.append(pad(item[0], ac_max_length - act_len))
                else:
                    new_item.append(item[0])

                if acw_len > ac_max_length:
                    new_item.append(reduce(item[1], acw_len - ac_max_length))
                elif acw_len < ac_max_length:
                    new_item.append(item[1], ac_max_length - acw_len)
                else:
                    new_item.append(item[1])

                if pm_len > pm_max_length:
                    new_item.append(reduce(item[2], pm_len - pm_max_length))
                elif pm_len < pm_max_length:
                    new_item.append(item[2], pm_max_length - pm_len)
                else:
                    new_item.append(item[2])

                new_items.append(new_item)
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
    model.fit(_train_features, _train_labels, verbose=0, batch_size=32, epochs=100, shuffle=True)
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
    act_data = read(act_path, '_act')
    acw_data = read(acw_path, '_acw')
    pm_data = read(pm_path, '_pm')

    all_features = extract_features(act_data, acw_data)

    all_features = pad_features(all_features)
    all_features = frame_reduce(all_features)

    for i in range(len(test_user_fold)):
        set_random_seed(2)
        train_features, test_features = train_test_split(all_features, test_user_fold[i])

        train_features, train_labels = flatten(train_features)
        test_features, test_labels = flatten(test_features)

        train_labels = np_utils.to_categorical(train_labels, len(activity_list))
        test_labels = np_utils.to_categorical(test_labels, len(activity_list))

        _run_(train_features, train_labels, test_features, test_labels)

run()


