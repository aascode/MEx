import os
import csv
import datetime as dt

subjects = []


def write_data(file_path, data):
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


def fix():
    subjects_path = 'E:/MEx/Data/3/'
    for subject in subjects:
        activities_path = os.path.join(subjects_path, subject)
        activities = os.listdir(activities_path)
        for activity in activities:
            activity_path = os.path.join(activities_path, activity)
            if activity == '04':
                dc_file_1 = os.path.join(activity_path, 'dc_1.csv')
                dc_file_2 = os.path.join(activity_path, 'dc_2.csv')
                fix_file(dc_file_1, os.path.join(activity_path, 'dc_1_.csv'))
                fix_file(dc_file_2, os.path.join(activity_path, 'dc_2_.csv'))
            else:
                dc_file = os.path.join(activity_path, 'dc.csv')
                fix_file(dc_file, os.path.join(activity_path, 'dc_.csv'))


def fix_file(dc_file, new_file):
    reader = csv.reader(open(dc_file, "r"), delimiter=",")
    data = []
    for row in reader:
        data.append(row)

    for index in range(len(data)-1):
        next_index = index+1
        if len(data[index][0]) == 19 and '.' not in data[index][0]:
            data[index][0] = data[index][0] + '.000000'
        if len(data[next_index][0]) == 19 and '.' not in data[next_index][0]:
            data[next_index][0] = data[next_index][0] + '.000000'
        time = dt.datetime.strptime(data[index][0], '%Y-%m-%d %H:%M:%S.%f')
        next_time = dt.datetime.strptime(data[next_index][0], '%Y-%m-%d %H:%M:%S.%f')
        if next_time < time:
            stime = str(time)
            stimes = stime.split('.')
            milli = stimes[1].split('0')[0]
            if len(milli) == 1:
                milli = '00'+ milli
            elif len(milli) == 2:
                milli = '0' + milli
            stime = stimes[0]+'.'+milli + '000'

            data[index][0] = stime
        write_data(new_file, ','.join([str(i) for i in data[index]]))
    write_data(new_file, ','.join([str(i) for i in data[len(data)-1]]))


fix()