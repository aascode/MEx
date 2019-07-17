import os
import csv
import datetime as dt
import numpy as np
from scipy import fftpack

file_name = 'acw_act_pm_'

epochs = 15
batch_size = 64
increment = 3
window = 10
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


def read_all(path):
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
