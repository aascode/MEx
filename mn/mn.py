import os
import csv
import datetime as dt
import numpy as np
from keras.utils import np_utils
from keras.layers import Input, Dense, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, LSTM, \
    Reshape, Dropout, TimeDistributed, Lambda
from keras.models import Model
import tensorflow as tf
import sklearn.metrics as metrics
import pandas as pd
from keras.layers.merge import _Merge
from keras.utils import Sequence
import os
import keras
import sys
import random

random.seed(0)
np.random.seed(1)

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

#results_file = '/Volumes/1708903/MEx/Data/np_pm_1.0.csv'
pm_mn_stream_folder = '/Volumes/1708903/MEx/Data/pm_mn_stream_500_5_7_720_5_2_1'
#pm_mn_stream_folder = '/home/mex/data/np_pm_mn_stream_500_5_7_720_5_2_1/'
results_file = '/home/mex/results/np_pm_mn.csv'

height = 32
width = 16
frame_size = height*width
samples_per_class = 5
classes_per_set = len(activity_list)
embedding_length = 100

number_of_users = 30
train_size = 500 * number_of_users
frames_per_second = 1
window = 5
increment = 2

pm_min_length = 14*window
pm_max_length = 15*window
dc_min_length = 10*window
dc_max_length = 15*window
ac_min_length = 95*window
ac_max_length = 100*window


class MNGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, is_test):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.is_test = is_test

    def __len__(self):
        return np.ceil(len(self.image_filenames) / float(self.batch_size))

    def read_mn(self, _file):
        reader = csv.reader(open(_file, "r"), delimiter=",")
        _slice_x = []
        _slice_y = []
        for row in reader:
            if len(row) == ((samples_per_class*classes_per_set)+1) * height * width*window * 1:
                _slice_x = [float(f) for f in row]
            if len(row) == samples_per_class*classes_per_set:
                _slice_y = [float(f) for f in row]
        _slice_x = np.array(_slice_x)
        _slice_y = np.array(_slice_y)
        _slice_y = keras.utils.to_categorical(_slice_y, classes_per_set)
        _slice_x = np.reshape(_slice_x, (((samples_per_class*classes_per_set)+1),  height, width*window, 1))
        return [_slice_x, _slice_y]

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = keras.utils.to_categorical(batch_y, classes_per_set)

        slices = [self.read_mn(file_name) for file_name in batch_x]
        slice_x = np.array([xx[0] for xx in slices])
        slice_y = np.array([xx[1] for xx in slices])
        if self.is_test:
            return [slice_x, slice_y]
        return [slice_x, slice_y], batch_y


def get_batch_data(_fold, mode):
    folder = os.path.join(pm_mn_stream_folder, _fold+'/'+mode+'/')
    files = os.listdir(folder)
    train_file_names = [os.path.join(folder, f) for f in files if f != 'target_y.csv']

    reader = csv.reader(open(os.path.join(folder, 'target_y.csv'), "r"), delimiter=",")
    labels = []
    for row in reader:
        labels.append(float(row[0]))
    return train_file_names, labels


class CosineSimilarity(_Merge):
    def __init__(self, nway=5, n_samp=1, **kwargs):
        super(CosineSimilarity, self).__init__(**kwargs)
        self.eps = 1e-10
        self.nway = nway
        self.n_samp = n_samp

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != self.nway * self.n_samp + 2:
            raise ValueError(
                'A ModelCosine layer should be called on a list of inputs of length %d' % (self.nway * self.n_samp + 2))

    def call(self, inputs):
        self.nway = (len(inputs) - 2) / self.n_samp
        similarities = []

        targetembedding = inputs[-2]
        numsupportset = len(inputs) - 2
        for ii in range(numsupportset):
            supportembedding = inputs[ii]

            sum_support = tf.reduce_sum(tf.square(supportembedding), 1, keep_dims=True)
            supportmagnitude = tf.rsqrt(tf.clip_by_value(sum_support, self.eps, float("inf")))

            sum_query = tf.reduce_sum(tf.square(targetembedding), 1, keep_dims=True)
            querymagnitude = tf.rsqrt(tf.clip_by_value(sum_query, self.eps, float("inf")))

            dot_product = tf.matmul(tf.expand_dims(targetembedding, 1), tf.expand_dims(supportembedding, 2))
            dot_product = tf.squeeze(dot_product, [1])

            cosine_similarity = dot_product * supportmagnitude * querymagnitude
            similarities.append(cosine_similarity)

        similarities = tf.concat(axis=1, values=similarities)
        softmax_similarities = tf.nn.softmax(similarities)
        preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities, 1), inputs[-1]))

        preds.set_shape((inputs[0].shape[0], self.nway))
        return preds

    def compute_output_shape(self, input_shape):
        input_shapes = input_shape
        return input_shapes[0][0], self.nway


def write_data(file_path, data):
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


def embedding_2D(x):
    x = Conv2D(32, kernel_size=(1,5), activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=1, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(1,5), activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=1, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(embedding_length, activation='relu')(x)
    x = BatchNormalization()(x)
    return x


def run_model_mn(fold):
    print(fold)
    write_data(results_file, 'fold:'+str(fold))
    numsupportset = samples_per_class * classes_per_set
    model_input = Input((numsupportset + 1, height, width * window * frames_per_second, 1))
    model_inputs = []
    for lidx in range(numsupportset):
        model_inputs.append(embedding_2D(Lambda(lambda x: x[:, lidx, :, :, :])(model_input)))
    targetembedding = embedding_2D(Lambda(lambda x: x[:, -1, :, :, :])(model_input))
    model_inputs.append(targetembedding)
    support_labels = Input((numsupportset, classes_per_set))
    model_inputs.append(support_labels)

    knn_similarity = CosineSimilarity(nway=classes_per_set, n_samp=samples_per_class)(model_inputs)

    model = Model(inputs=[model_input, support_labels], outputs=knn_similarity)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    train_file_names, train_labels = get_batch_data(str(fold), 'train')
    train_gen = MNGenerator(train_file_names, train_labels, 16, False)
    model.fit_generator(train_gen, epochs=10, verbose=0)

    test_file_names, test_labels = get_batch_data(str(fold), 'test')
    test_gen = MNGenerator(test_file_names, test_labels, len(test_labels)/5, True)
    _predict_labels = model.predict_generator(test_gen, steps=5)
    #score = model.evaluate_generator(test_gen)
    #print(score)
    #write_data(results_file, ','.join(score))
    #write_data(results_file, 'label lengths:'+str(len(test_labels))+','+str(len(_predict_labels)))
    _test_labels = keras.utils.to_categorical(test_labels, classes_per_set)
    f_score = metrics.f1_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1), average='macro')
    accuracy = metrics.accuracy_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1))
    results = 'pm,' + str(accuracy)+',' + str(f_score)
    print(results)
    write_data(results_file, 'results:'+results)
    _test_labels = pd.Series(_test_labels.argmax(axis=1), name='Actual')
    _predict_labels = pd.Series(_predict_labels.argmax(axis=1), name='Predicted')
    df_confusion = pd.crosstab(_test_labels, _predict_labels)
    print(df_confusion)
    write_data(results_file, 'confusion matrix:'+str(df_confusion))


def run():
    tf.set_random_seed(2)
    run_model_mn(sys.argv[1])


run()
