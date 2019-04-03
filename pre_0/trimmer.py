import os
import csv
import datetime as dt


def write_data(file_path, data):
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


def write_sensor(file_path, data, sensor):
    for subject in data:
        subject_data = data[subject]
        subject_folder = os.path.join(file_path, subject)
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)
        for activity in subject_data:
            activity_folder = os.path.join(subject_folder, activity)
            if not os.path.exists(activity_folder):
                os.makedirs(activity_folder)
            activity_data = subject_data[activity]
            file_path = os.path.join(activity_folder, sensor+'.csv')
            for item in activity_data:
                write_data(file_path, ','.join([str(i) for i in item]))


def read_pm(path):
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
                if len(row) == 513:
                    if len(row[0]) == 19 and '.' not in row[0]:
                        row[0] = row[0]+'.000000'
                    temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
                    temp.extend([float(f) for f in row[1:]])
                    temp_data.append(temp)

            activity_dict[activity_id] = temp_data
        subjects_dict[subject] = activity_dict
    return subjects_dict


def extract_times(pm_data):
    times_dict = {}
    for subject in pm_data:
        subject_dict = pm_data[subject]
        subject_times = {}
        for activity in subject_dict:
            activity_dict = subject_dict[activity]
            subject_times[activity] = [activity_dict[0][0], activity_dict[len(activity_dict) - 1][0]]
        times_dict[subject] = subject_times
    return times_dict


def read_dc(path):
    subjects_dict = {}
    subjects = os.listdir(path)
    for subject in subjects:
        activity_dict = {}
        subject_files = os.path.join(path, subject)
        activities = os.listdir(subject_files)
        for activity in activities:
            data_file = os.path.join(subject_files, activity)
            activity_id = activity.split('_')[1]
            if not activity_id == '04':
                temp_data = []
                reader = csv.reader(open(data_file, "r"), delimiter=",")
                for row in reader:
                    if len(row) == 76801:
                        if len(row[0]) == 19 and '.' not in row[0]:
                            row[0] = row[0]+'.000000'
                        temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
                        _temp = [float(f) for f in row[1:]]
                        temp.extend(_temp)
                        temp_data.append(temp)
                activity_dict[activity_id] = temp_data
        subjects_dict[subject] = activity_dict
    return subjects_dict


def read_ac(path):
    subjects_dict = {}
    subjects = os.listdir(path)
    for subject in [s for s in subjects if s.endswith('.csv')]:
        sub = subject.split('_')[0]
        data_file = os.path.join(path, subject)
        temp_data = []
        reader = csv.reader(open(data_file, "r"), delimiter=",")
        for row in reader:
            if len(row) == 4:
                if len(row[0]) == 19 and '.' not in row[0]:
                    row[0] = row[0]+'.000000'
                temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
                _temp = [float(f) for f in row[1:]]
                temp.extend(_temp)
                temp_data.append(temp)
        subjects_dict[sub] = temp_data
    return subjects_dict


def strip_dc(dc_data, times_dict):
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


def strip_ac(ac_data, times_dict):
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


def get_data():
    pm_data = read_pm("/Volumes/1708903/MEx/Data/2/pm/")
    dc_data = read_dc("/Volumes/1708903/MEx/Data/2/dc/")
    #acw_data = read_ac("E:\\Mex\\Data\\2\\acw\\")
    #act_data = read_ac("E:\\Mex\\Data\\2\\act\\")
    times_dict = extract_times(pm_data)

    dc_data = strip_dc(dc_data, times_dict)
    #act_data = strip_ac(act_data, times_dict)
    #acw_data = strip_ac(acw_data, times_dict)

    write_sensor("E:\\Mex\\Data\\3\\", pm_data, 'pm')

get_data()