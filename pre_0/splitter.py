import os
import csv
import datetime as dt

subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '23', '24', '25', '26', '27', '28', '29', '30']
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


def split_pm(path, new_path, time_dict):
    for subject in time_dict:
        subject_files = os.path.join(path, subject)

        data_file = os.path.join(subject_files, '04')
        data_file = os.path.join(data_file, 'pm_1.csv')

        temp_data = []
        reader = csv.reader(open(data_file, "r"), delimiter=",")
        for row in reader:
            if len(row) == 513:
                if len(row[0]) == 19 and '.' not in row[0]:
                    row[0] = row[0]+'.000000'
                temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
                temp.extend([float(f) for f in row[1:]])
                temp_data.append(temp)
        times = time_dict[subject]
        start, end = times[0], times[1]
        file_path_1 = os.path.join(new_path, subject)
        file_path_1 = os.path.join(file_path_1, '04')
        if not os.path.exists(file_path_1):
            os.makedirs(file_path_1)
        file_path_1 = os.path.join(file_path_1, 'pm_1.csv')

        file_path_2 = os.path.join(new_path, subject)
        file_path_2 = os.path.join(file_path_2, '04')
        if not os.path.exists(file_path_2):
            os.makedirs(file_path_2)
        file_path_2 = os.path.join(file_path_2, 'pm_2.csv')

        for item in temp_data:
            time_stamp = item[0]
            if time_stamp <= start:
                write_data(file_path_1, ','.join([str(i) for i in item]))
            elif time_stamp >= end:
                write_data(file_path_2, ','.join([str(i) for i in item]))


def read_dc(path):
    times = {}
    for subject in subjects:

        subject_files = os.path.join(path, subject)
        data_files = os.path.join(subject_files, '04')
        dc1_file = os.path.join(data_files, 'dc_1.csv')
        dc2_file = os.path.join(data_files, 'dc_2.csv')
        reader = csv.reader(open(dc1_file, "r"), delimiter=",")
        last_time = None
        for row in reader:
            if len(row) == 76801:
                if len(row[0]) == 19 and '.' not in row[0]:
                    row[0] = row[0]+'.000000'
                last_time = dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
        _times = [last_time]
        reader = csv.reader(open(dc2_file, "r"), delimiter=",")
        first_time = None
        for row in reader:
            if first_time == None:
                if len(row) == 76801:
                    if len(row[0]) == 19 and '.' not in row[0]:
                        row[0] = row[0]+'.000000'
                    first_time = dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
        _times.append(first_time)
        times[subject] = _times
    return times


def split_ac(path, new_path, sensor, time_dict):
    for subject in time_dict:
        subject_files = os.path.join(path, subject)

        data_file = os.path.join(subject_files, '04')
        data_file = os.path.join(data_file, sensor+'_1.csv')

        temp_data = []
        reader = csv.reader(open(data_file, "r"), delimiter=",")
        for row in reader:
            if len(row) == 4:
                if len(row[0]) == 19 and '.' not in row[0]:
                    row[0] = row[0]+'.000000'
                temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
                temp.extend([float(f) for f in row[1:]])
                temp_data.append(temp)
        times = time_dict[subject]
        start, end = times[0], times[1]
        file_path_1 = os.path.join(new_path, subject)
        file_path_1 = os.path.join(file_path_1, '04')
        file_path_1 = os.path.join(file_path_1, sensor+'_1.csv')

        file_path_2 = os.path.join(new_path, subject)
        file_path_2 = os.path.join(file_path_2, '04')
        file_path_2 = os.path.join(file_path_2, sensor+'_2.csv')

        for item in temp_data:
            time_stamp = item[0]
            if time_stamp <= start:
                write_data(file_path_1, ','.join([str(i) for i in item]))
            elif time_stamp >= end:
                write_data(file_path_2, ','.join([str(i) for i in item]))


def get_data():
    times_dict = read_dc("E:/MEx/Data/pre_3/")

    split_pm("E:/MEx/Data/pre_3/", "E:/MEx/Data/pre_4/", times_dict)
    split_ac("E:/MEx/Data/pre_3/", "E:/MEx/Data/pre_4/", 'acw', times_dict)
    split_ac("E:/MEx/Data/pre_3/", "E:/MEx/Data/pre_4/", 'act', times_dict)

get_data()