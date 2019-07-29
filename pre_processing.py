import csv
import datetime as dt
import os
import numpy as np
from scipy import fftpack

np.random.seed(100000)

# sample rates
# pressure mat-15Hz,(32 rows 16 columns)
# depth camera-15Hz, (40 rows 30 columns)
# acclerometers-100Hz

# feature indices: 0-pm, 1-dc, 2-act, 3-acw

activityList = ["01", "02", "03", "04", "05", "06", "07"]
idList = range(len(activityList))
activityIdDict = dict(zip(activityList, idList))

min_pad_lengths = [70, 40, 500, 450]
max_pad_lengths = [75, 60, 500, 500]


def read_dc(path):
    subjects_dict = {}
    subjects = os.listdir(path)
    for subject in subjects:
        activity_dict = {}
        subject_files = os.path.join(path, subject)
        activities = os.listdir(subject_files)
        for activity in activities:
            data_file = os.path.join(subject_files, activity)
            activity_id = activity.split('_')[1].replace('.csv', '')
            temp_data = []
            reader = csv.reader(open(data_file, "r"), delimiter=",")
            for row in reader:
                if len(row) == 76801:
                    tt = dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
                    temp = [tt]
                    _temp = [float(f) for f in row[1:]]
                    _temp = np.array(_temp)
                    _temp = np.reshape(_temp, (240, 320))
                    _temp = ski.block_reduce(_temp, (2, 2), func=np.mean)
                    _temp = ski.block_reduce(_temp, (2, 2), func=np.mean)
                    # _temp = ski.block_reduce(_temp, (2, 2), func=np.mean)
                    _temp = np.ndarray.tolist(np.reshape(_temp, (1, 4800)))[0]
                    _temp = [float(x) for x in _temp]
                    temp.extend(_temp)
                    temp_data.append(temp)
            activity_dict[activity_id] = temp_data
        subjects_dict[subject] = activity_dict
    return subjects_dict


def print_lengths(data):
    for subject in data:
        activities = data[subject]
        for activity in activities:
            print(str(subject) + ',' + str(activity) + ',' + str(len(activities[activity])))


def find_index(_data, _time_stamp):
    return [_index for _index, _item in enumerate(_data) if _item[0] >= _time_stamp][0]


def split_windows(data, increment, window):
    inputs = []
    start = data[0][0]
    end = data[len(data) - 1][0]
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window)
    start_index = 0
    end_index = 0
    while start + _window < end:
        _end = start + _window
        start_index = find_index(data, start)
        end_index = find_index(data, _end)
        ins = [a[1:] for a in data[start_index:end_index]]
        start = start + _increment
        inputs.append(ins)
    return inputs


def extract_features(subjects_data, increment, window):
    features = []
    labels = []
    for subject in subjects_data:
        subject_data = subjects_data[subject]
        for activity in subject_data:
            activity_data = subject_data[activity]
            act = activityIdDict.get(activity)
            inputs = split_windows(activity_data, increment, window)
            features.extend(inputs)
            labels.extend([act for i in range(len(inputs))])
    return features, labels


def _extract_features(sensor_data_array, increment, window):
    all_features = {}
    for subject in sensor_data_array[0]:
        _activities = {}
        for activity in activityList:
            act = activityIdDict.get(activity)
            inputs = []
            input_lengths = []
            features = []
            for sensor_data in sensor_data_array:
                activity_data = sensor_data[subject][activity]
                _inputs = split_windows(activity_data, increment, window)
                inputs.append(_inputs)
                input_lengths.append(len(_inputs))
            min_length = min(input_lengths)
            _inputs_ = [ins[:min_length] for ins in inputs]
            for item1, item2, item3, item4 in zip(_inputs_[0], _inputs_[1], _inputs_[2], _inputs_[3]):
                features.append([item1, item2, item3, item4])
            _activities[act] = features
        all_features[subject] = _activities
    return all_features


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
        _acts = {}
        activities = _features[subject]
        for act in activities:
            items = activities[act]
            _items = []
            for item in items:
                item_lengths = []
                for _min, _item in zip(min_pad_lengths, item):
                    _len = len(_item)
                    if _len < _min:
                        break
                    item_lengths.append(_len)
                if len(item_lengths) == 4:
                    new_items = []
                    for _max, _item in zip(max_pad_lengths, item):
                        _len = len(_item)
                        if _len > _max:
                            _item = reduce(_item, _len - _max)
                        elif _len < _max:
                            _item = pad(_item, _max - _len)
                        new_items.append(_item)
                    _items.append(new_items)
            _acts[act] = _items
        new_features[subject] = _acts
    return new_features


def dct(data, dct_length):
    new_data = []
    for item in data:
        new_item = []
        new_item.append(np.array(item[0]))
        new_item.append(np.array(item[1]))
        if dct_length > 0:
            for ac_item in [item[2], item[3]]:
                x = [t[0] for t in ac_item]
                y = [t[1] for t in ac_item]
                z = [t[2] for t in ac_item]

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


def flatten(data, dct_length):
    flatten_data = []
    flatten_labels = []

    for subject in data:
        activities = data[subject]
        for activity in activities:
            activity_data = activities[activity]
            flatten_data.extend(activity_data)
            flatten_labels.extend([activity for i in range(len(activity_data))])
    return dct(flatten_data, dct_length), flatten_labels


def get_features(data_array, increment, window):
    _features = _extract_features(data_array, increment, window)
    _features = pad_features(_features)
    return _features


def extract_sensor(features, sensor_indices):
    new_features = []
    for item in features:
        new_features.append([ite for index, ite in enumerate(item) if index in [sensor_indices]])
    return new_features


def _read(_file, _length):
    reader = csv.reader(open(_file, "r"), delimiter=",")
    _data = []
    for row in reader:
        if len(row) == _length:
            if len(row[0]) == 19 and '.' not in row[0]:
                row[0] = row[0]+'.000000'
            temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
            temp.extend([float(f) for f in row[1:]])
            _data.append(temp)
    return _data


def read():
    sensors = ['pm']
    alldata = {}
    path = '/Volumes/1708903/MEx/Data/3/'
    subjects = os.listdir(path)
    for subject in subjects:
        allactivities = {}
        subject_path = os.path.join(path, subject)
        activities = os.listdir(subject_path)
        activity_data = {}
        for activity in activities:
            activity_path = os.path.join(subject_path, activity)
            datas = os.listdir(activity_path)
            for data in datas:
                sensor = data.split('.')[0]
                _data = None
                if 'pm' in data and 'pm' in sensors:
                    _data = _read(os.path.join(activity_path, data), 513)
                if 'dc' in data and 'dc' in sensors:
                    _data = _read(os.path.join(activity_path, data), 76801)
                if 'act' in data and 'act' in sensors:
                    _data = _read(os.path.join(activity_path, data), 4)
                if 'acw' in data and 'acw' in sensors:
                    _data = _read(os.path.join(activity_path, data), 4)
                activity_data[sensor] = _data
        allactivities[activity] = activity_data
    alldata[subject] = allactivities


read()

