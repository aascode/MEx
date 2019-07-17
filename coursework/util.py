import mex
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Model
import numpy as np
import sklearn.metrics as metrics
from keras.utils import np_utils
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# get features by sensor index
# index 0: wrist accelerometer
# index 1: thigh accelerometer
# index 2: pressure mat
def get_features(_data, index):
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            _activities[activity] = activities[activity][index]
        _features[subject] = _activities
    return _features


# pad data to increase length
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


# remove data to reduce length
def reduce(data, length):
    red_length = []
    if length % 2 == 0:
        red_length = [int(length / 2), int(length / 2)]
    else:
        red_length = [int(length / 2) + 1, int(length / 2)]
    new_data = data[red_length[0]:len(data) - red_length[1]]
    return new_data


# check length of each train/test instance, pad or reduce to make sure they all are the same size
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
                if _len < mex.min_length:
                    continue
                elif _len > mex.max_length:
                    item = reduce(item, _len - mex.max_length)
                    new_items.append(item)
                elif _len < mex.max_length:
                    item = pad(item, mex.max_length - _len)
                    new_items.append(item)
            new_activities[act] = new_items
        new_features[subject] = new_activities
    return new_features


# reduce frame size to mex.frames_per_second
def trim(_data):
    _length = len(_data)
    _inc = _length / (mex.window * mex.frames_per_second)
    _new_data = []
    for i in range(mex.window * mex.frames_per_second):
        _new_data.append(_data[i * _inc])
    return _new_data


# reduce frame size to mex.frames_per_second
def frame_reduce(_data):
    if mex.frames_per_second == 0:
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


# train/test split by test ids
def train_test_split(user_data, test_ids):
    train_data = {key: value for key, value in user_data.items() if key not in test_ids}
    test_data = {key: value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data


# flatten the dictionary structure to train and test instance and labels
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


# build the convolutional model for 2D data
def build_model_2D():
    _input = Input(shape=(32, 16 * mex.window * mex.frames_per_second, 1))
    x = Conv2D(32, kernel_size=(1, 5), activation='relu')(_input)
    x = MaxPooling2D(pool_size=2, strides=1, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(1, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=1, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(720, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(len(mex.activity_list), activation='softmax')(x)

    model = Model(inputs=_input, outputs=x)
    return model


# train and test a convolutional model on 2D data
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

    _model = build_model_2D()
    _model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    _model.fit(_train_features, _train_labels, verbose=0, batch_size=mex.batch_size, epochs=mex.epochs, shuffle=True)
    _predict_labels = _model.predict(_test_features, batch_size=64, verbose=0)
    f_score = metrics.f1_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1), average='macro')
    accuracy = metrics.accuracy_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1))
    results = str(accuracy) + ',' + str(f_score)
    print(results)

    _test_labels = pd.Series(_test_labels.argmax(axis=1), name='Actual')
    _predict_labels = pd.Series(_predict_labels.argmax(axis=1), name='Predicted')
    df_confusion = pd.crosstab(_test_labels, _predict_labels)
    print(df_confusion)


# train and test a knn algorithm on 1D data
def run_knn_model(_train_features, _train_labels, _test_features, _test_labels):
    _train_features = np.array(_train_features)
    print(_train_features.shape)

    _test_features = np.array(_test_features)
    print(_test_features.shape)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(_train_features, _train_labels)
    _predict_labels = model.predict(_test_features)
    _train_labels = np_utils.to_categorical(_train_labels, len(mex.activity_list))
    _test_labels = np_utils.to_categorical(_test_labels, len(mex.activity_list))
    _predict_labels = np_utils.to_categorical(_predict_labels, len(mex.activity_list))

    f_score = metrics.f1_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1), average='macro')
    accuracy = metrics.accuracy_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1))
    results = str(accuracy) + ',' + str(f_score)
    print(results)

    _test_labels = pd.Series(_test_labels.argmax(axis=1), name='Actual')
    _predict_labels = pd.Series(_predict_labels.argmax(axis=1), name='Predicted')
    df_confusion = pd.crosstab(_test_labels, _predict_labels)
    print(df_confusion)


# train and test a svm algorithm on 1D data
def run_svm_model(_train_features, _train_labels, _test_features, _test_labels):
    _train_features = np.array(_train_features)
    print(_train_features.shape)

    _test_features = np.array(_test_features)
    print(_test_features.shape)

    model = SVC()
    model.fit(_train_features, _train_labels)
    _predict_labels = model.predict(_test_features)
    _train_labels = np_utils.to_categorical(_train_labels, len(mex.activity_list))
    _test_labels = np_utils.to_categorical(_test_labels, len(mex.activity_list))
    _predict_labels = np_utils.to_categorical(_predict_labels, len(mex.activity_list))

    f_score = metrics.f1_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1), average='macro')
    accuracy = metrics.accuracy_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1))
    results = str(accuracy) + ',' + str(f_score)
    print(results)

    _test_labels = pd.Series(_test_labels.argmax(axis=1), name='Actual')
    _predict_labels = pd.Series(_predict_labels.argmax(axis=1), name='Predicted')
    df_confusion = pd.crosstab(_test_labels, _predict_labels)
    print(df_confusion)
