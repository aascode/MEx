import os
import csv
import datetime as dt
import numpy as np
import os
import random

random.seed(0)
np.random.seed(1)

frame_size = 3
dct_length = 60

sensors = ['acw', 'act', 'pm']

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

path = '/Volumes/1708903/MEx/Data/pre_4/'
dump_file = '/Volumes/1708903/MEx/coursework/acw_act_pm'

frames_per_second = 15


def write(_map, subject, activity):
    for i in range(len(_map['acw'])):
        for x,y,z in zip(_map['acw'][i], _map['act'][i], _map['pm'][i]):
            x.extend(y[1:])
            x.extend(z[1:])
            write_data(dump_file+'_'+str(subject)+'_'+str(activity)+'_'+str(i)+'.csv', ','.join([str(xx) for xx in x]))


def write_data(file_path, data):
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


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
            activity_path = os.path.join(subject_path, activity)
            sensors = os.listdir(activity_path)
            activity_map = {}
            if len(sensors) == 4:
                _data = _read(os.path.join(activity_path, 'pm_1.csv'), 513)
                _times = [d[0] for d in _data]

                activity_map['pm'] = []
                activity_map['pm'].append(_data)

                _acw_data = _read(os.path.join(activity_path, 'acw_1.csv'), 4)
                _act_data = _read(os.path.join(activity_path, 'act_1.csv'), 4)

                _acw_data_ = []
                _act_data_ = []

                for _time in _times:
                    for _element in _acw_data:
                        if _element[0] > _time:
                            _acw_data_.append(_element)
                            break

                    for _element in _act_data:
                        if _element[0] > _time:
                            _act_data_.append(_element)
                            break

                activity_map['acw'] = []
                activity_map['acw'].append(_acw_data_)
                activity_map['act'] = []
                activity_map['act'].append(_act_data_)

                write(activity_map, subject, activity)
            elif len(sensors) == 6:
                _data = _read(os.path.join(activity_path, 'pm_1.csv'), 513)
                _times = [d[0] for d in _data]

                activity_map['pm'] = []
                activity_map['pm'].append(_data)

                _acw_data = _read(os.path.join(activity_path, 'acw_1.csv'), 4)
                _act_data = _read(os.path.join(activity_path, 'act_1.csv'), 4)

                _acw_data_ = []
                _act_data_ = []

                for _time in _times:
                    for _element in _acw_data:
                        if _element[0] > _time:
                            _acw_data_.append(_element)
                            break

                    for _element in _act_data:
                        if _element[0] > _time:
                            _act_data_.append(_element)
                            break

                activity_map['acw'] = []
                activity_map['acw'].append(_acw_data_)
                activity_map['act'] = []
                activity_map['act'].append(_act_data_)

                _data = _read(os.path.join(activity_path, 'pm_2.csv'), 513)
                _times = [d[0] for d in _data]

                activity_map['pm'].append(_data)

                _acw_data = _read(os.path.join(activity_path, 'acw_2.csv'), 4)
                _act_data = _read(os.path.join(activity_path, 'act_2.csv'), 4)

                _acw_data_ = []
                _act_data_ = []

                for _time in _times:
                    for _element in _acw_data:
                        if _element[0] > _time:
                            _acw_data_.append(_element)
                            break

                    for _element in _act_data:
                        if _element[0] > _time:
                            _act_data_.append(_element)
                            break

                activity_map['acw'].append(_acw_data_)
                activity_map['act'].append(_act_data_)
                write(activity_map,subject, activity)
            allactivities[activity] = activity_map
        alldata[subject] = allactivities
    return alldata


all_data = read()
