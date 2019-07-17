import mex
from keras.utils import np_utils
import util
import numpy as np
import os

path = '/Users/anjanawijekoon/MEx_wtpm/'


def average(p_data):
    new_p_data = []
    for i in range(len(p_data[0])):
        new_p_data.append(sum([x[i] for x in p_data])/len(p_data))
    return new_p_data


def write_data(file_path, data):
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


def flatten(data):
    for subject in data:
        activities = data[subject]
        for activity in activities:
            activity_data = activities[activity]
            for w, t, p in zip(activity_data[0], activity_data[1], activity_data[2]):
                temp = [subject]
                temp.extend(w)
                temp.extend(t)
                temp.extend(average(p))
                temp.append(activity)
                write_data('mex_'+str(mex.window)+'_'+str(mex.increment)+'.csv', ','.join([str(t) for t in temp]))


all_features = mex.extract_features(mex.read_all(path))
flatten(all_features)

