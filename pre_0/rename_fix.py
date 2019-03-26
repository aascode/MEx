import os

path = '/Volumes/1708903/MEx/Data/3/'
subjects = os.listdir(path)
for subject in subjects:
    subject_path = os.path.join(path, subject)
    activities = os.listdir(subject_path)
    activity_path = os.path.join(subject_path, '04')
    sensors = os.listdir(activity_path)
    for sensor in sensors:
        sensor_path = os.path.join(activity_path, sensor)
        # if '_1' not in sensor and '_2' not in sensor:
        #    sensor_ = sensor.split('.')[0].replace('_', '')
        #    sensor_ = sensor_+'_1.csv'
        #    os.rename(sensor_path, os.path.join(activity_path, sensor_))
        sensor_ = sensor.split('.')[0]
        if sensor_.endswith('_'):
            sensor_ = sensor_[:-1]
        sensor_ = sensor_+'.csv'
        os.rename(sensor_path, os.path.join(activity_path, sensor_))

