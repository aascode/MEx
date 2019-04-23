import datetime as dt
import numpy as np
import os
import cv2 as cv
import csv

scalex = 0.1
scaley = 0.1
path = '/Volumes/1708903/MEx/Data/pre_3/'
new_path = '/Volumes/1708903/MEx/Data/dc_scaled/'+str(scaley)+'/'

def write_data(file_path, data):
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


def _read(_file, _new_file):
    if '.DS_Store' not in _file:
        reader = csv.reader(open(_file, "r"), delimiter=",")
        _data = []
        for row in reader:
            if len(row) == 76801:
                if len(row[0]) == 19 and '.' not in row[0]:
                    row[0] = row[0]+'.000000'
                temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
                _temp = [float(f) for f in row[1:]]
                _temp = np.array(_temp)
                _temp = np.reshape(_temp, (240, 320))
                __temp = cv.resize(_temp, None, fx=scalex, fy=scaley)
                temp.extend(np.reshape(__temp, (240*scalex*320*scaley)).tolist())
                write_data(_new_file, ','.join([str(i) for i in temp]))
        return _data


def read():
    alldata = {}
    subjects = os.listdir(path)
    for subject in subjects:
        allactivities = {}
        subject_path = os.path.join(path, subject)
        activities = os.listdir(subject_path)
        new_subject_path = os.path.join(new_path, subject)
        if not os.path.exists(new_subject_path):
            os.makedirs(new_subject_path)
        for activity in activities:
            activity_data = {}
            activity_path = os.path.join(subject_path, activity)
            datas = os.listdir(activity_path)
            for data in datas:
                sensor = data.split('.')[0]
                if 'dc' in sensor:
                    _data = _read(os.path.join(activity_path, data),
                                  os.path.join(new_subject_path, activity+'_'+sensor+'.csv'))

read()


