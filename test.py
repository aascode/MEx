import os
import csv
import datetime as dt
import numpy as np

sensors = ['pm']

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

path = '/Volumes/1708903/MEx/Data/min3/'

test_user_fold = ['21']

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


def split_windows(data, _increment, _window):
    outputs = []
    start = data[0][0]
    end = data[len(data) - 1][0]
    increment = dt.timedelta(seconds=_increment)
    window = dt.timedelta(seconds=_window)

    frames = [a[1:] for a in data[:]]
    frames = np.array(frames)
    _length = frames.shape[0]
    frames = np.reshape(frames, (_length*512))
    frames = frames/max(frames)
    frames = [float("{0:.5f}".format(f)) for f in frames.tolist()]
    frames = np.reshape(np.array(frames), (_length, 512))

    while start + window < end:
        _end = start + window
        start_index = find_index(data, start)
        end_index = find_index(data, _end)
        instances = [a[:] for a in frames[start_index:end_index]]
        start = start + increment
        outputs.append(instances)
    return outputs


# single sensor
def extract_features(_data, _increment, _window):
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            time_windows = []
            activity_id = activity_id_dict.get(activity)
            activity_data = activities[activity]
            for sensor in activity_data:
                time_windows.extend(split_windows(activity_data[sensor], _increment, _window))
            time_windows = np.array(time_windows)
            _activities[activity_id] = time_windows
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


all_data = read()
all_features = extract_features(all_data, 2, 5)
all_data = None
all_features = pad_features(all_features)

train_features, test_features = train_test_split(all_features, test_user_fold)

train_features, train_labels = flatten(train_features)
test_features, test_labels = flatten(train_features)
