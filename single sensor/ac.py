import os
import csv
import datetime as dt
import numpy as np
from keras.utils import np_utils
from keras.layers import Input, Dense, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, LSTM, Reshape, TimeDistributed
from keras.models import Model
from tensorflow import set_random_seed
import cv2 as cv
import keras.backend as K
import matplotlib.pyplot as plt
np.random.seed(1)
set_random_seed(1)

frame_size = 3*1

sensors = ['acw']

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

#path = '/Volumes/1708903/MEx/Data/3-16/'
path = '/Users/anjanawijekoon/Data/MEx/min3/'

#test_user_fold = ['21', '22', '23', '24', '25']
test_user_fold = ['21']

frames_per_second = 1
window = 5
increment = 2

pm_min_length = 70
pm_max_length = 75
dc_min_length = 40
act_min_length = 500
acw_min_length = 450
dc_max_length = 60
act_max_length = 500
acw_max_length = 450


def _read(_file, _length):
    reader = csv.reader(open(_file, "r"), delimiter=",")
    _data = []
    for row in reader:
        if len(row) == _length:
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
            activity_data = {}
            activity_path = os.path.join(subject_path, activity)
            datas = os.listdir(activity_path)
            for data in datas:
                sensor = data.split('.')[0]
                if 'pm' in data and 'pm' in sensors:
                    _data = _read(os.path.join(activity_path, data), 513)
                    activity_data[sensor] = _data
                if 'dc' in data and 'dc' in sensors:
                    _data = _read(os.path.join(activity_path, data), 76801)
                    activity_data[sensor] = _data
                if 'act' in data and 'act' in sensors:
                    _data = _read(os.path.join(activity_path, data), 4)
                    activity_data[sensor] = _data
                if 'acw' in data and 'acw' in sensors:
                    _data = _read(os.path.join(activity_path, data), 4)
                    activity_data[sensor] = _data
            allactivities[activity] = activity_data
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


def train_test_split(user_data, test_ids):
    train_data = {key: value for key, value in user_data.items() if key not in test_ids}
    test_data = {key: value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data


def get_hold_out_users(users):
    indices = np.random.choice(len(users), int(len(users) / 5), False)
    _users = [u for indd, u in enumerate(users) if indd in indices]
    return _users


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


def scale(_features):
    _newfeatures = []
    for _item in _features:
        __newfeatures = []
        for __item in _item:
            # plt.imshow(__item)
            # plt.show()
            # print(__item.shape)
            __item = cv.resize(__item, None, fx=2, fy=1)
            __newfeatures.append(__item)
            # print(__item.shape)
            # plt.imshow(__item)
            # plt.show()
        _newfeatures.append(__newfeatures)
    _newfeatures = np.array(_newfeatures)
    return _newfeatures


def scale_and_threshold(_features):
    _new_features = []
    for _item in _features:
        __newfeatures = []
        for __item in _item:
            __item = cv.resize(__item, None, fx=2, fy=1)
            __newfeatures.append(__item)
        __newfeatures = threshold(__newfeatures)
        _new_features.append(__newfeatures)
    _new_features = np.array(_new_features)
    print(_new_features.shape)
    return _new_features


def threshold(_features):
    _features = np.array(_features)
    # print(_features.shape)
    _n_features = np.reshape(_features, (_features.shape[0], _features.shape[1]*_features.shape[2]))
    # print(_n_features.shape)
    # print(_n_features.mean(axis=0).shape)
    # f, axarr = plt.subplots(3,2)
    # axarr[0,0].imshow(np.reshape(_n_features[0, :], (32, 32)))
    # axarr[0,1].imshow(np.reshape(_n_features[1, :], (32, 32)))
    # axarr[1,0].imshow(np.reshape(_n_features[2, :], (32, 32)))
    # axarr[1,1].imshow(np.reshape(_n_features[3, :], (32, 32)))
    # axarr[2,0].imshow(np.reshape(_n_features[4, :], (32, 32)))
    # plt.show()

    _n_features = _n_features - _n_features.mean(axis=0)
    # print(_n_features.shape)
    # plt.imshow(np.reshape(_n_features[4, :], (32, 32)))
    # plt.show()

    _cov = np.cov(_n_features, rowvar=True)
    # print(_cov.shape)

    u, s, v = np.linalg.svd(_cov)
    # print(u.shape)
    # print(s.shape)
    # print(v.shape)
    # epsilon = 0.1
    # _n_features = u.dot(np.diag(1.0/np.sqrt(s + epsilon))).dot(u.T).dot(_n_features)
    # f, axarr = plt.subplots(3,2)
    # axarr[0,0].imshow(np.reshape(_n_features[0, :], (32, 32)))
    # axarr[0,1].imshow(np.reshape(_n_features[1, :], (32, 32)))
    # axarr[1,0].imshow(np.reshape(_n_features[2, :], (32, 32)))
    # axarr[1,1].imshow(np.reshape(_n_features[3, :], (32, 32)))
    # axarr[2,0].imshow(np.reshape(_n_features[4, :], (32, 32)))
    # plt.show()

    epsilon = 0.00001
    _n_features = u.dot(np.diag(1.0/np.sqrt(s + epsilon))).dot(u.T).dot(_n_features)
    # f, axarr = plt.subplots(3,2)
    # axarr[0,0].imshow(np.reshape(_n_features[0, :], (32, 32)))
    # axarr[0,1].imshow(np.reshape(_n_features[1, :], (32, 32)))
    # axarr[1,0].imshow(np.reshape(_n_features[2, :], (32, 32)))
    # axarr[1,1].imshow(np.reshape(_n_features[3, :], (32, 32)))
    # axarr[2,0].imshow(np.reshape(_n_features[4, :], (32, 32)))
    # plt.show()

    _n_features = (_n_features - _n_features.min()) / (_n_features.max() - _n_features.min())
    # plt.imshow(np.reshape(_n_features[4, :], (32, 32)))
    # plt.show()

    _n_features = np.reshape(_n_features, (_features.shape[0], _features.shape[1], _features.shape[2]))
    # print(_n_features.shape)

    return _n_features


def build_model_LSTM():
    _input = Input(shape=(32, 16, window*frames_per_second))
    x = Conv2D(100, kernel_size=2, activation='relu')(_input)
    x = MaxPooling2D(pool_size=2, strides=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(150, kernel_size=2, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(200, kernel_size=2, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Reshape((K.int_shape(x)[1]*K.int_shape(x)[2], K.int_shape(x)[3]))(x)
    x = LSTM(600)(x)
    x = Dense(300, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(len(activity_list), activation='softmax')(x)

    model = Model(inputs=_input, outputs=x)
    print(model.summary())
    return model


def run_model_LSTM(_train_features, _train_labels, _val_features, _val_labels, _test_features, _test_labels):
    # (None, 32, 16, timestamps)
    _train_features = np.array(_train_features)
    _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1], 32, 16))
    _train_features = np.swapaxes(_train_features, 1, 2)
    _train_features = np.swapaxes(_train_features, 2, 3)
    print(_train_features.shape)

    _val_features = np.array(_val_features)
    _val_features = np.reshape(_val_features, (_val_features.shape[0], _val_features.shape[1], 32, 16))
    _val_features = np.swapaxes(_val_features, 1, 2)
    _val_features = np.swapaxes(_val_features, 2, 3)
    print(_val_features.shape)

    _test_features = np.array(_test_features)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], _test_features.shape[1], 32, 16))
    _test_features = np.swapaxes(_test_features, 1, 2)
    _test_features = np.swapaxes(_test_features, 2, 3)
    print(_test_features.shape)

    pm_model = build_model_LSTM()
    pm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    pm_model.fit(_train_features, _train_labels, verbose=1, batch_size=32, epochs=15, shuffle=True,
                 validation_data=(_val_features, _val_labels))
    score = pm_model.evaluate(_test_features, _test_labels, batch_size=32, verbose=0)
    results = 'pm,' + str(score[0]) + ',' + str(score[1])
    print(results)


def build_model_2D():
    _input = Input(shape=(32, 16, window*frames_per_second))
    x = Conv2D(100, kernel_size=2, activation='relu')(_input)
    x = MaxPooling2D(pool_size=2, strides=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(150, kernel_size=2, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(200, kernel_size=2, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(300, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(len(activity_list), activation='softmax')(x)

    model = Model(inputs=_input, outputs=x)
    print(model.summary())
    return model


def run_model_2D(_train_features, _train_labels, _val_features, _val_labels, _test_features, _test_labels):
    # (None, 32, 16, timestamps)
    _train_features = np.array(_train_features)
    _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1], 32, 16))
    _train_features = np.swapaxes(_train_features, 1, 2)
    _train_features = np.swapaxes(_train_features, 2, 3)
    print(_train_features.shape)

    _val_features = np.array(_val_features)
    _val_features = np.reshape(_val_features, (_val_features.shape[0], _val_features.shape[1], 32, 16))
    _val_features = np.swapaxes(_val_features, 1, 2)
    _val_features = np.swapaxes(_val_features, 2, 3)
    print(_val_features.shape)

    _test_features = np.array(_test_features)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], _test_features.shape[1], 32, 16))
    _test_features = np.swapaxes(_test_features, 1, 2)
    _test_features = np.swapaxes(_test_features, 2, 3)
    print(_test_features.shape)

    pm_model = build_model_2D()
    pm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    pm_model.fit(_train_features, _train_labels, verbose=1, batch_size=32, epochs=15, shuffle=True,
                 validation_data=(_val_features, _val_labels))
    score = pm_model.evaluate(_test_features, _test_labels, batch_size=32, verbose=0)
    results = 'pm,' + str(score[0]) + ',' + str(score[1])
    print(results)


def build_model_TDConvLSTM():
    _input = Input(shape=(window*frames_per_second, 32, 16, 1))
    x = TimeDistributed(Conv2D(32, kernel_size=3, activation='relu'))(_input)
    x = TimeDistributed(MaxPooling2D(pool_size=2, strides=2, data_format='channels_last'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(64, kernel_size=3, activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=2, strides=2, data_format='channels_last'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Reshape((K.int_shape(x)[1], K.int_shape(x)[2]*K.int_shape(x)[3]*K.int_shape(x)[4]))(x)
    x = LSTM(600)(x)
    x = Dense(300, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(len(activity_list), activation='softmax')(x)

    model = Model(inputs=_input, outputs=x)
    print(model.summary())
    return model


def run_model_TDConvLSTM(_train_features, _train_labels, _val_features, _val_labels, _test_features, _test_labels):
    # (None, timestamps, 32, 16)
    _train_features = np.array(_train_features)
    _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1], 32, 16))
    _train_features = np.expand_dims(_train_features, 4)
    print(_train_features.shape)

    _val_features = np.array(_val_features)
    _val_features = np.reshape(_val_features, (_val_features.shape[0], _val_features.shape[1], 32, 16))
    _val_features = np.expand_dims(_val_features, 4)
    print(_val_features.shape)

    _test_features = np.array(_test_features)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], _test_features.shape[1], 32, 16))
    _test_features = np.expand_dims(_test_features, 4)
    print(_test_features.shape)

    pm_model = build_model_TDConvLSTM()
    pm_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    pm_model.fit(_train_features, _train_labels, verbose=1, batch_size=32, epochs=20, shuffle=True,
                 validation_data=(_val_features, _val_labels))
    score = pm_model.evaluate(_test_features, _test_labels, batch_size=32, verbose=0)
    results = 'pm,' + str(score[0]) + ',' + str(score[1])
    print(results)


def build_model_scaledTDConvLSTM():
    _input = Input(shape=(window*frames_per_second, 32, 32, 1))
    x = TimeDistributed(Conv2D(32, kernel_size=3, activation='relu'))(_input)
    x = TimeDistributed(MaxPooling2D(pool_size=2, strides=2, data_format='channels_last'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(64, kernel_size=3, activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=2, strides=2, data_format='channels_last'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(128, kernel_size=3, activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=2, strides=2, data_format='channels_last'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Reshape((K.int_shape(x)[1], K.int_shape(x)[2]*K.int_shape(x)[3]*K.int_shape(x)[4]))(x)
    x = LSTM(600)(x)
    x = Dense(300, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(len(activity_list), activation='softmax')(x)

    model = Model(inputs=_input, outputs=x)
    print(model.summary())
    return model


def run_model_scaledTDConvLSTM(_train_features, _train_labels, _val_features, _val_labels, _test_features, _test_labels):
    # (None, timestamps, 32, 32)
    _train_features = np.array(_train_features)
    _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1], 32, 16))
    _train_features = scale_and_threshold(_train_features)
    # _train_features = scale(_train_features)
    _train_features = np.expand_dims(_train_features, 4)
    print(_train_features.shape)

    _val_features = np.array(_val_features)
    _val_features = np.reshape(_val_features, (_val_features.shape[0], _val_features.shape[1], 32, 16))
    _val_features = scale_and_threshold(_val_features)
    # _val_features = scale(_val_features)
    _val_features = np.expand_dims(_val_features, 4)
    print(_val_features.shape)

    _test_features = np.array(_test_features)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], _test_features.shape[1], 32, 16))
    #_test_features = scale(_test_features)
    _test_features = scale_and_threshold(_test_features)
    _test_features = np.expand_dims(_test_features, 4)
    print(_test_features.shape)

    pm_model = build_model_scaledTDConvLSTM()
    pm_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    pm_model.fit(_train_features, _train_labels, verbose=1, batch_size=32, epochs=20, shuffle=True,
                 validation_data=(_val_features, _val_labels))
    score = pm_model.evaluate(_test_features, _test_labels, batch_size=32, verbose=0)
    results = 'pm,' + str(score[0]) + ',' + str(score[1])
    print(results)


all_data = read()
all_features = extract_features(all_data)
all_data = None
all_features = pad_features(all_features)
all_features = frame_reduce(all_features)

train_features, test_features = train_test_split(all_features, test_user_fold)
val_user_fold = ['22']#get_hold_out_users(list(train_features.keys()))
train_features, val_features = train_test_split(train_features, val_user_fold)

train_features, train_labels = flatten(train_features)
val_features, val_labels = flatten(val_features)
test_features, test_labels = flatten(test_features)

train_labels = np_utils.to_categorical(train_labels, len(activity_list))
val_labels = np_utils.to_categorical(val_labels, len(activity_list))
test_labels = np_utils.to_categorical(test_labels, len(activity_list))

# 1-0.7511
# 2-0.6123
# 3-0.6968
# 5-0.6410
# 10-0.7375
# 15-0.6817
# 20-0.5565
# 25-0.4389
# 50-0.5384
# 75-0.6033,0.4675,0.5279,0.5037
# run_model_2D(train_features, train_labels, val_features, val_labels, test_features, test_labels)

# 1-0.6501
# 10-0.6606
# 20-0.4962
# 30-0.4449
# 75-0.6349
# run_model_LSTM(train_features, train_labels, val_features, val_labels, test_features, test_labels)


run_model_scaledTDConvLSTM(train_features, train_labels, val_features, val_labels, test_features, test_labels)
