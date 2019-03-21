import os
import numpy as np
import csv
import datetime as dt
import matplotlib.pyplot as plt

#subject = '03'
subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
            '19', '20', '21', '22', '23', '24', '25']
activities = ['01', '02', '03', '04', '05', '06', '07']

folder = '/Volumes/1708903/MEx/Data/fullbackup/pm/'
target_folder = '/Volumes/1708903/MEx/Data/1/pm_/'

data_file = '_1.csv'


def write_data(file_path, data):
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


for subject in subjects:
    for activity in activities:
        #print(activity)
        data = []
        times = []
        reader = csv.reader(open(folder+subject+'_'+activity+data_file, "r"), delimiter=",")
        for row in reader:
            if len(row) == 513:
                times.append(dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f'))
                _temp = [float(f) for f in row[1:]]
                data.append(_temp)
        data = np.array(data)

        #data = np.genfromtxt(folder+subject+'_'+activity+data_file, delimiter=',')
        #print(data.shape)
        #data = data[:,1:]
        #print(data.shape)
        length = len(data)
        #data = np.reshape(data, (length*512))
        #print(data.shape)
        #data.sort()
        #print(data[0])
        #print(str(data[len(data)-1])+','+str(data[len(data)-2])+','+str(data[len(data)-3]))

        # minmax normalisation
        data = np.reshape(data, (length*512))
        data = data / max(data)
        data = np.reshape(data, (length,512))

        target_file = target_folder+subject
        if not os.path.exists(target_file):
            os.makedirs(target_file)
        target_file = target_file+'/'+subject+'_'+activity+'.csv'
        for itime, idata in zip(times, data):
            write_data(target_file, str(itime)+','+','.join([str(float("{0:.4f}".format(f))) for f in idata]))


        # visualise
        #for i in range(len(_data)):
        #    if i % 2000:
        #        plt.imshow(_data[i])
        #        plt.show()


