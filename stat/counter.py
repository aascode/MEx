import csv
import datetime as dt
import os
import numpy as np

path = '/Volumes/1708903/MEx/Data/pre_5/'
subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
            '19', '20', '21', '23', '24', '25', '26', '27', '28', '29', '30']


activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

window = 5
increment = 5


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


def read(subject):
    alldata = {}
    allactivities = {}
    subject_path = os.path.join(path, subject)
    activities = os.listdir(subject_path)
    for activity in activities:
        allsensors = {}
        activity_path = os.path.join(subject_path, activity)
        sensors = os.listdir(activity_path)
        for sensor in sensors:
            _sensor = sensor.split('_')[0]
            _sensor_file = os.path.join(activity_path, sensor)
            _sensor_data = _read(_sensor_file)
            if _sensor in allsensors:
                _current_sensor_data = allsensors[_sensor]
                _current_sensor_data.append(_sensor_data)
                allsensors[_sensor] = _current_sensor_data
            else:
                allsensors[_sensor] = [_sensor_data]
        allactivities[activity] = allsensors
    alldata[subject] = allactivities
    return alldata


def find_index(_data, _time_stamp):
    return [_index for _index, _item in enumerate(_data) if _item[0] >= _time_stamp][0]


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
def extract_features(_data, sensor):
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            time_windows = []
            activity_id = activity_id_dict.get(activity)
            activity_data = activities[activity]
            sensor_data = activity_data[sensor]
            for item in sensor_data:
                time_windows.extend(split_windows(item))
            _activities[activity_id] = time_windows
        _features[subject] = _activities
    return _features


def flatten(_data):
    flatten_data = []
    for subject in _data:
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            print(str(activity)+','+str(len(activity_data)))
            flatten_data.extend(activity_data)
    return flatten_data


for subject in subjects:
    all_data = read(subject)
    print(subject)
    pm_features = extract_features(all_data, 'pm')

    all_f = flatten(pm_features)
    all_f = np.array(all_f)
    print(all_f.shape)

    act_features = extract_features(all_data, 'act')

    all_f = flatten(act_features)
    all_f = np.array(all_f)
    print(all_f.shape)

    acw_features = extract_features(all_data, 'acw')
    all_f = flatten(acw_features)
    all_f = np.array(all_f)
    print(all_f.shape)

    dc_features = extract_features(all_data, 'dc')

    all_f = flatten(dc_features)
    all_f = np.array(all_f)
    print(all_f.shape)
