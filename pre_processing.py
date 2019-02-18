import csv
import datetime as dt
import os
import numpy as np
import skimage.measure as ski
from scipy import fftpack
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

np.random.seed(100000)
#sample rates
#pressure mat-15Hz,(32 rows 16 columns)
#depth camera-15Hz, (40 rows 30 columns)
#acclerometers-100Hz

#feature indices: 0-pm, 1-dc, 2-act, 3-acw

class MexPreprocessing:
    activityList = ["01", "02", "03", "04", "05", "06", "07"]
    idList = range(len(activityList))
    activityIdDict = dict(zip(activityList, idList))
    subjects_15_second_gap = ["03", "04", "05", "06", "07"]
    min_pad_lengths = [70, 40, 500, 450]
    max_pad_lengths = [75, 60, 500, 500]

    def read_pm(self, path):
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
                    if(len(row) == 513):
                        temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
                        temp.extend([float(f) for f in row[1:]])
                        temp_data.append(temp)
                activity_dict[activity_id] = temp_data
            subjects_dict[subject] = activity_dict
        return subjects_dict

    def extract_times(self, pm_data):
        times_dict = {}
        for subject in pm_data:
            subject_dict = pm_data[subject]
            subject_times ={}
            for activity in subject_dict:
                activity_dict = subject_dict[activity]
                subject_times[activity] = [activity_dict[0][0], activity_dict[len(activity_dict)-1][0]]
            times_dict[subject] = subject_times
        return times_dict

    def read_dc(self, path):
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
                    if(len(row) == 76801):
                        tt = dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
                        if subject in self.subjects_15_second_gap:
                            tt = tt + dt.timedelta(seconds=-15)
                        temp = [tt]
                        _temp = [float(f) for f in row[1:]]
                        _temp = np.array(_temp)
                        _temp = np.reshape(_temp, (240,320))
                        _temp = ski.block_reduce(_temp, (2, 2), func=np.mean)
                        _temp = ski.block_reduce(_temp, (2, 2), func=np.mean)
                        #_temp = ski.block_reduce(_temp, (2, 2), func=np.mean)
                        _temp = np.ndarray.tolist(np.reshape(_temp, (1,4800)))[0]
                        _temp = [float("{0:.4f}".format(x)) for x in _temp]
                        temp.extend(_temp)
                        temp_data.append(temp)
                activity_dict[activity_id] = temp_data
            subjects_dict[subject] = activity_dict
        return subjects_dict

    def read_ac(self, path):
        subjects_dict = {}
        subjects = os.listdir(path)
        for subject in [s for s in subjects if s.endswith('.csv')]:
            sub = subject.split('_')[0]
            data_file = os.path.join(path, subject)
            temp_data = []
            reader = csv.reader(open(data_file, "r"), delimiter=",")
            for row in reader:
                if(len(row) == 4):
                    temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
                    _temp = [float(f) for f in row[1:]]
                    temp.extend(_temp)
                    temp_data.append(temp)
            subjects_dict[sub] = temp_data
        return subjects_dict

    def strip_dc(self, dc_data, times_dict):
        new_dc_data = {}
        for subject in times_dict:
            activities = times_dict[subject]
            new_activities = {}
            for activity in activities:
                times = activities[activity]
                start, end = times[0], times[1]
                dc_data_item = dc_data[subject][activity]
                new_dc_data_item = []
                for item in dc_data_item:
                    time_stamp = item[0]
                    if time_stamp < start or time_stamp > end:
                        continue
                    elif time_stamp >= start and time_stamp <= end:
                        new_dc_data_item.append(item)
                new_activities[activity] = new_dc_data_item
            new_dc_data[subject] = new_activities
        return new_dc_data

    def strip_ac(self, ac_data, times_dict):
        new_ac_data = {}
        for subject in times_dict:
            activities = times_dict[subject]
            new_activities = {}
            for activity in activities:
                times = activities[activity]
                start, end = times[0], times[1]
                ac_data_item = ac_data[subject]
                new_ac_data_item = []
                for item in ac_data_item:
                    time_stamp = item[0]
                    if time_stamp < start:
                        continue
                    elif time_stamp >= start and time_stamp <= end:
                        new_ac_data_item.append(item)
                    elif time_stamp > end:
                        break
                new_activities[activity] = new_ac_data_item
            new_ac_data[subject] = new_activities
        return new_ac_data

    def print_lengths(self, data):
        for subject in data:
            activities = data[subject]
            for activity in activities:
                print(str(subject)+','+str(activity)+','+str(len(activities[activity])))

    def find_index(self, _data, _time_stamp):
        return [_index for _index,_item in enumerate(_data) if _item[0] >= _time_stamp][0]

    def split_windows(self, data, increment, window):
        inputs = []
        start = data[0][0]
        end = data[len(data)-1][0]
        _increment = dt.timedelta(seconds=increment)
        _window = dt.timedelta(seconds=window)
        start_index = 0
        end_index = 0
        while start + _window < end:
            _end = start + _window
            start_index = self.find_index(data, start)
            end_index = self.find_index(data, _end)
            ins = [a[1:] for a in data[start_index:end_index]]
            start = start + _increment
            inputs.append(ins)
        return inputs

    ###################### 1 sensor #############################
    def extract_features(self, subjects_data, increment, window):
        features = []
        labels = []
        for subject in subjects_data:
            subject_data = subjects_data[subject]
            for activity in subject_data:
                activity_data = subject_data[activity]
                act = self.activityIdDict.get(activity)
                inputs = self.split_windows(activity_data, increment, window)
                features.extend(inputs)
                labels.extend([act for i in range(len(inputs))])
        return features, labels

    ###################### all sensor #############################
    def _extract_features(self, sensor_data_array, increment, window):
        all_features = {}
        for subject in sensor_data_array[0]:
            _activities = {}
            for activity in self.activityList:
                act = self.activityIdDict.get(activity)
                inputs = []
                input_lengths = []
                features = []
                for sensor_data in sensor_data_array:
                    activity_data = sensor_data[subject][activity]
                    _inputs = self.split_windows(activity_data, increment, window)
                    inputs.append(_inputs)
                    input_lengths.append(len(_inputs))
                min_length = min(input_lengths)
                _inputs_ = [ins[:min_length] for ins in inputs]
                for item1,item2,item3,item4 in zip(_inputs_[0],_inputs_[1],_inputs_[2],_inputs_[3]):
                    features.append([item1,item2,item3,item4])
                _activities[act] = features
            all_features[subject] = _activities
        return all_features

    def pad(self, data, length):
        pad_length = []
        if length%2 == 0:
            pad_length = [int(length / 2), int(length/2)]
        else:
            pad_length = [int(length/2)+1, int(length/2)]
        new_data = []
        for index in range(pad_length[0]):
            new_data.append(data[0])
        new_data.extend(data)
        for index in range(pad_length[1]):
            new_data.append(data[len(data)-1])
        return new_data


    def reduce(self, data, length):
        red_length = []
        if length%2 == 0:
            red_length = [int(length / 2), int(length/2)]
        else:
            red_length = [int(length/2)+1, int(length/2)]
        new_data = data[red_length[0]:len(data)-red_length[1]]
        return new_data

    def pad_features(self, _features):
        new_features = {}
        for subject in _features:
            _acts = {}
            activities = _features[subject]
            for act in activities:
                items = activities[act]
                _items = []
                for item in items:
                    item_lengths = []
                    for _min, _item in zip(self.min_pad_lengths, item):
                        _len = len(_item)
                        if _len < _min:
                           break
                        item_lengths.append(_len)
                    if len(item_lengths) == 4:
                        new_items = []
                        for _max, _item in zip(self.max_pad_lengths, item):
                            _len = len(_item)
                            if _len > _max:
                                _item = self.reduce(_item, _len - _max)
                            elif _len < _max:
                                _item = self.pad(_item, _max - _len)
                            new_items.append(_item)
                        _items.append(new_items)
                _acts[act] = _items
            new_features[subject] = _acts
        return new_features

    def dct(self, data, dct_length):
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

    def get_data(self):
        pm_data = self.read_pm("E:\\Mex\\Data\\1\\pm\\")
        dc_data = self.read_dc("E:\\Mex\\Data\\1\\dc\\")
        acw_data = self.read_ac("E:\\Mex\\Data\\1\\acw\\")
        act_data = self.read_ac("E:\\Mex\\Data\\1\\act\\")
        #pm_data = self.read_pm('C:\\DCBackup\\smalltest\\pm\\')
        #dc_data = self.read_dc('C:\\DCBackup\\smalltest\\dc\\')
        #act_data = self.read_ac('C:\\DCBackup\\smalltest\\act\\')
        #acw_data = self.read_ac('C:\\DCBackup\\smalltest\\acw\\')
        times_dict = self.extract_times(pm_data)

        dc_data = self.strip_dc(dc_data, times_dict)
        act_data = self.strip_ac(act_data, times_dict)
        acw_data = self.strip_ac(acw_data, times_dict)

        return [pm_data, dc_data, act_data, acw_data]

    def get_features(self, data_array, increment, window):
        _features = self._extract_features(data_array, increment, window)
        _features = self.pad_features(_features)
        return _features

    def flatten(self, data, dct_length):
        flatten_data = []
        flatten_labels = []

        for subject in data:
            activities = data[subject]
            for activity in activities:
                activity_data = activities[activity]
                flatten_data.extend(activity_data)
                flatten_labels.extend([activity for i in range(len(activity_data))])
        return self.dct(flatten_data, dct_length), flatten_labels

    def extract_sensor(self, features, sensor_indices):
        new_features = []
        for item in features:
            new_features.append([ite for index, ite in enumerate(item) if index in [sensor_indices]])
        return new_features




