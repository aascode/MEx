import os
import csv
import datetime as dt
import numpy as np
from scipy import fftpack
from keras.utils import np_utils
import sklearn.metrics as metrics
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Model

path = '/Users/anjanawijekoon/MEx_wtpm/'
file_name = 'acw_act_pm_'

epochs = 15
batch_size = 64
increment = 2
window = 5
frame_size = 3
dct_length = 60

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

min_length = 14*window
max_length = 15*window
frames_per_second = 1

def read_file(_file):
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

# people
#    |_ person A
#              |_ activity 01
#                       |_ [0]
#              |_ activity 02
#                       |_ [0]
#              |_ activity 03
#                       |_ [0]
#              |_ activity 04
#                       |_ [0]
#                       |_ [1]
#              |_ activity 05
#                       |_ [0]
#              |_ activity 06
#                       |_ [0]
#              |_ activity 07
#                       |_ [0]


def read_all():
    all_files = os.listdir(path)
    people = {}
    for _file in [f for f in all_files if not f.startswith('.')]:
        _data = read_file(os.path.join(path, _file))
        _time_stamp = [_d[0] for _d in _data]
        _acw = [_d[1:4] for _d in _data]
        _act = [_d[4:7] for _d in _data]
        _pm = [_d[7:] for _d in _data]
        _person = _file.split('.csv')[0].replace(file_name, '').split('_')[0]
        _activity = _file.split('.csv')[0].replace(file_name, '').split('_')[1]
        _index = _file.split('.csv')[0].replace(file_name, '').split('_')[2]
        if _person in people:
            activities = people[_person]
            if _activity in activities:
                _activity_ = activities[_activity]
                _activity_.append([_time_stamp, _acw, _act, _pm])
                activities[_activity] = _activity_
                people[_person] = activities
            else:
                activities[_activity] = []
                activities[_activity].append([_time_stamp, _acw, _act, _pm])
                people[_person] = activities
        else:
            activities = {}
            activities[_activity] = []
            activities[_activity].append([_time_stamp, _acw, _act, _pm])
            people[_person] = activities
    return people


def find_index(_data, _time_stamp):
    return [_index for _index, _item in enumerate(_data) if _item >= _time_stamp][0]


def split_windows(times, data):
    outputs = []
    start = times[0]
    end = times[len(times) - 1]
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window)

    frames = data
    while start + _window < end:
        _end = start + _window
        start_index = find_index(times, start)
        end_index = find_index(times, _end)
        instances = [a[:] for a in frames[start_index:end_index]]
        start = start + _increment
        outputs.append(instances)
    return outputs


def dct(windows):
    dct_window = []
    for tw in windows:
        x = [t[0] for t in tw]
        y = [t[1] for t in tw]
        z = [t[2] for t in tw]

        dct_x = np.abs(fftpack.dct(x, norm='ortho'))
        dct_y = np.abs(fftpack.dct(y, norm='ortho'))
        dct_z = np.abs(fftpack.dct(z, norm='ortho'))

        v = np.array([])
        v = np.concatenate((v, dct_x[:dct_length]))
        v = np.concatenate((v, dct_y[:dct_length]))
        v = np.concatenate((v, dct_z[:dct_length]))

        dct_window.append(v)
    return dct_window


def extract_features(_data):
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            activity_id = activity_id_dict.get(activity)
            activity_data = activities[activity]
            _acw = []
            _act = []
            _pm = []
            for item in activity_data:
                _acw.extend(dct(split_windows(item[0], item[1])))
                _act.extend(dct(split_windows(item[0], item[2])))
                _pm.extend(split_windows(item[0], item[3]))
            _activities[activity_id] = [_acw, _act, _pm]
        _features[subject] = _activities
    return _features


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


def pad_features(_all_features):
    new_features = {}
    for subject in _all_features:
        new_activities = {}
        activities = _all_features[subject]
        for act in activities:
            items = activities[act]
            new_items = []
            for item in items:
                _len = len(item)
                if _len < min_length:
                    continue
                elif _len > max_length:
                    item = reduce(item, _len - max_length)
                    new_items.append(item)
                elif _len < max_length:
                    item = pad(item, max_length - _len)
                    new_items.append(item)
            new_activities[act] = new_items
        new_features[subject] = new_activities
    return new_features


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


def get_features(_data, index):
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            _activities[activity] = activities[activity][index]
        _features[subject] = _activities
    return _features


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


def build_model_2D():
    _input = Input(shape=(32, 16 * window * frames_per_second, 1))
    x = Conv2D(32, kernel_size=(1,5), activation='relu')(_input)
    x = MaxPooling2D(pool_size=2, strides=1, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(1,5), activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=1, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(720, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(len(activity_list), activation='softmax')(x)

    model = Model(inputs=_input, outputs=x)
    return model


def run_knn_model(_train_features, _train_labels, _test_features, _test_labels):
    _train_features = np.array(_train_features)
    print(_train_features.shape)

    _test_features = np.array(_test_features)
    print(_test_features.shape)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(_train_features, _train_labels)
    _predict_labels = model.predict(_test_features)
    _train_labels = np_utils.to_categorical(_train_labels, len(activity_list))
    _test_labels = np_utils.to_categorical(_test_labels, len(activity_list))
    _predict_labels = np_utils.to_categorical(_predict_labels, len(activity_list))

    f_score = metrics.f1_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1), average='macro')
    accuracy = metrics.accuracy_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1))
    results = 'pm,' + str(accuracy)+',' + str(f_score)
    print(results)

    _test_labels = pd.Series(_test_labels.argmax(axis=1), name='Actual')
    _predict_labels = pd.Series(_predict_labels.argmax(axis=1), name='Predicted')
    df_confusion = pd.crosstab(_test_labels, _predict_labels)
    print(df_confusion)


def run_model_2D(_train_features, _train_labels, _test_features, _test_labels):
    _train_features = np.array(_train_features)
    _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1], 32, 16))
    _train_features = np.swapaxes(_train_features, 1, 2)
    _train_features = np.swapaxes(_train_features, 2, 3)
    _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1],
                                                   _train_features.shape[2] * _train_features.shape[3]))
    _train_features = np.expand_dims(_train_features, 4)
    print(_train_features.shape)

    _test_features = np.array(_test_features)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], _test_features.shape[1], 32, 16))
    _test_features = np.swapaxes(_test_features, 1, 2)
    _test_features = np.swapaxes(_test_features, 2, 3)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], _test_features.shape[1],
                                                 _test_features.shape[2] * _test_features.shape[3]))
    _test_features = np.expand_dims(_test_features, 4)
    print(_test_features.shape)

    pm_model = build_model_2D()
    pm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    pm_model.fit(_train_features, _train_labels, verbose=0, batch_size=batch_size, epochs=epochs, shuffle=True)
    _predict_labels = pm_model.predict(_test_features, batch_size=64, verbose=0)
    f_score = metrics.f1_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1), average='macro')
    accuracy = metrics.accuracy_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1))
    results = 'pm,' + str(accuracy)+',' + str(f_score)
    print(results)

    _test_labels = pd.Series(_test_labels.argmax(axis=1), name='Actual')
    _predict_labels = pd.Series(_predict_labels.argmax(axis=1), name='Predicted')
    df_confusion = pd.crosstab(_test_labels, _predict_labels)
    print(df_confusion)


def run():
    all_data = read_all()
    all_features = extract_features(all_data)

    acw_features = get_features(all_features, 0)
    act_features = get_features(all_features, 1)
    pm_features = get_features(all_features, 2)

    all_people = pm_features.keys()

    padded_pm_features = pad_features(pm_features)
    reduced_pm_features = frame_reduce(padded_pm_features)
    
    for i in range(len(all_people)):
        test_persons = [all_people[i]]
        pm_train_features, pm_test_features = train_test_split(reduced_pm_features, test_persons)

        pm_train_features, pm_train_labels = flatten(pm_train_features)
        pm_test_features, pm_test_labels = flatten(pm_test_features)

        pm_train_labels = np_utils.to_categorical(pm_train_labels, len(activity_list))
        pm_test_labels = np_utils.to_categorical(pm_test_labels, len(activity_list))

        run_model_2D(pm_train_features, pm_train_labels, pm_test_features, pm_test_labels)

    for i in range(len(all_people)):
        test_persons = [all_people[i]]
        acw_train_features, acw_test_features = train_test_split(acw_features, test_persons)

        acw_train_features, acw_train_labels = flatten(acw_train_features)
        acw_test_features, acw_test_labels = flatten(acw_test_features)

        run_knn_model(acw_train_features, acw_train_labels, acw_test_features, acw_test_labels)

    for i in range(len(all_people)):
        test_persons = [all_people[i]]
        act_train_features, act_test_features = train_test_split(act_features, test_persons)

        act_train_features, act_train_labels = flatten(act_train_features)
        act_test_features, act_test_labels = flatten(act_test_features)

        run_knn_model(act_train_features, act_train_labels, act_test_features, act_test_labels)


run()
