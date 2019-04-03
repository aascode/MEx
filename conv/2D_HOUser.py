import pre_processing
from keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Model
from keras.utils import np_utils
import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(1)

write_path = 'E:\\Mex\\results\\1.txt'


def write_data(file_path, data):
    if (os.path.isfile(file_path)):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


def build_pm_model(class_length):
    pm_input = Input(shape=(32, 16, 75))
    pm_conv = Conv2D(150, kernel_size=3, activation='relu')(pm_input)
    pm_conv = MaxPooling2D(pool_size=2, strides=2)(pm_conv)
    pm_conv = BatchNormalization()(pm_conv)
    pm_conv = Conv2D(150, kernel_size=3, activation='relu')(pm_conv)
    pm_conv = MaxPooling2D(pool_size=2, strides=2)(pm_conv)
    pm_conv = BatchNormalization()(pm_conv)
    flatten = Flatten()(pm_conv)
    dense_layer = Dense(600, activation='tanh')(flatten)
    dense_layer = BatchNormalization()(dense_layer)
    dense_layer = Dense(100, activation='tanh')(dense_layer)
    dense_layer = Dense(class_length, activation='softmax')(dense_layer)

    model = Model(inputs=pm_input, outputs=dense_layer)
    print(model.summary())
    return model


def run_pm(pm_train_features, pm_test_features, train_labels, test_labels, class_length):
    pm_train_features = np.array(pm_train_features)
    print(pm_train_features.shape)
    pm_train_features = np.reshape(pm_train_features, (pm_train_features.shape[0], pm_train_features.shape[2], 32, 16))
    print(pm_train_features.shape)
    pm_train_features = np.swapaxes(pm_train_features, 1, 3)
    print(pm_train_features.shape)
    pm_train_features = np.swapaxes(pm_train_features, 1, 2)
    print(pm_train_features.shape)
    pm_test_features = np.array(pm_test_features)
    print(pm_test_features.shape)
    pm_test_features = np.reshape(pm_test_features, (pm_test_features.shape[0], pm_test_features.shape[2], 32, 16))
    print(pm_test_features.shape)
    pm_test_features = np.swapaxes(pm_test_features, 1, 3)
    print(pm_test_features.shape)
    pm_test_features = np.swapaxes(pm_test_features, 1, 2)
    print(pm_test_features.shape)

    pm_model = build_pm_model(class_length=class_length)
    pm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    pm_model.fit(pm_train_features, train_labels, verbose=1, batch_size=50, epochs=10, shuffle=True)
    pm_score = pm_model.evaluate(pm_test_features, test_labels, batch_size=50, verbose=1)
    pm_result = 'pm,' + str(pm_score[0]) + ',' + str(pm_score[1])
    print(pm_result)
    write_data(write_path, pm_result)


def build_dc_model(class_length):
    dc_input = Input(shape=(80, 60, 60))
    dc_conv = Conv2D(150, kernel_size=3, activation='relu')(dc_input)
    dc_conv = MaxPooling2D(pool_size=2, strides=2)(dc_conv)
    dc_conv = BatchNormalization()(dc_conv)
    dc_conv = Conv2D(150, kernel_size=3, activation='relu')(dc_conv)
    dc_conv = MaxPooling2D(pool_size=2, strides=2)(dc_conv)
    dc_conv = BatchNormalization()(dc_conv)
    flatten = Flatten()(dc_conv)
    dense_layer = Dense(600, activation='tanh')(flatten)
    dense_layer = BatchNormalization()(dense_layer)
    dense_layer = Dense(100, activation='tanh')(dense_layer)
    dense_layer = Dense(class_length, activation='softmax')(dense_layer)

    model = Model(inputs=dc_input, outputs=dense_layer)
    print(model.summary())
    return model


def run_dc(dc_train_features, dc_test_features, train_labels, test_labels, class_length):
    dc_train_features = np.array(dc_train_features)
    print(dc_train_features.shape)
    dc_train_features = np.reshape(dc_train_features, (dc_train_features.shape[0], dc_train_features.shape[2], 80, 60))
    print(dc_train_features.shape)
    dc_train_features = np.swapaxes(dc_train_features, 1, 3)
    print(dc_train_features.shape)
    dc_train_features = np.swapaxes(dc_train_features, 1, 2)
    print(dc_train_features.shape)
    dc_test_features = np.array(dc_test_features)
    print(dc_test_features.shape)
    dc_test_features = np.reshape(dc_test_features, (dc_test_features.shape[0], dc_test_features.shape[2], 80, 60))
    print(dc_test_features.shape)
    dc_test_features = np.swapaxes(dc_test_features, 1, 3)
    print(dc_test_features.shape)
    dc_test_features = np.swapaxes(dc_test_features, 1, 2)
    print(dc_test_features.shape)

    dc_model = build_dc_model(class_length=class_length)
    dc_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dc_model.fit(dc_train_features, train_labels, verbose=1, batch_size=50, epochs=10, shuffle=True)
    dc_score = dc_model.evaluate(dc_test_features, test_labels, batch_size=50, verbose=1)
    dc_result = 'dc,' + str(dc_score[0]) + ',' + str(dc_score[1])
    print(dc_result)
    write_data(write_path, dc_result)


def build_ac_model(class_length):
    ac_input = Input(shape=(180, 1))
    ac_conv = Conv1D(150, kernel_size=3, activation='relu')(ac_input)
    ac_conv = MaxPooling1D(pool_size=2, strides=2)(ac_conv)
    ac_conv = BatchNormalization()(ac_conv)
    ac_conv = Conv1D(150, kernel_size=3, activation='relu')(ac_conv)
    ac_conv = MaxPooling1D(pool_size=2, strides=2)(ac_conv)
    ac_conv = BatchNormalization()(ac_conv)
    flatten = Flatten()(ac_conv)
    dense_layer = Dense(600, activation='tanh')(flatten)
    dense_layer = BatchNormalization()(dense_layer)
    dense_layer = Dense(100, activation='tanh')(dense_layer)
    dense_layer = Dense(class_length, activation='softmax')(dense_layer)

    model = Model(inputs=ac_input, outputs=dense_layer)
    print(model.summary())
    return model


def run_ac(ac_train_features, ac_test_features, train_labels, test_labels, class_length):
    ac_train_features = np.array(ac_train_features)
    print(ac_train_features.shape)
    ac_train_features = np.reshape(ac_train_features,
                                   (ac_train_features.shape[0], ac_train_features.shape[1], ac_train_features.shape[2]))
    print(ac_train_features.shape)
    ac_train_features = np.swapaxes(ac_train_features, 1, 2)
    print(ac_train_features.shape)
    ac_test_features = np.array(ac_test_features)
    print(ac_test_features.shape)
    ac_test_features = np.reshape(ac_test_features,
                                  (ac_test_features.shape[0], ac_test_features.shape[1], ac_test_features.shape[2]))
    print(ac_test_features.shape)
    ac_test_features = np.swapaxes(ac_test_features, 1, 2)
    print(ac_test_features.shape)

    ac_model = build_ac_model(class_length=class_length)
    ac_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ac_model.fit(ac_train_features, train_labels, verbose=1, batch_size=50, epochs=10, shuffle=True)
    return ac_model.evaluate(ac_test_features, test_labels, batch_size=50, verbose=1)


def run_acw(acw_train_features, acw_test_features, train_labels, test_labels, class_length):
    acw_score = run_ac(acw_train_features, acw_test_features, train_labels, test_labels, class_length)
    acw_result = 'act,' + str(acw_score[0]) + ',' + str(acw_score[1])
    print(acw_result)
    write_data(write_path, acw_result)


def run_act(act_train_features, act_test_features, train_labels, test_labels, class_length):
    act_score = run_ac(act_train_features, act_test_features, train_labels, test_labels, class_length)
    act_result = 'act,' + str(act_score[0]) + ',' + str(act_score[1])
    print(act_result)
    write_data(write_path, act_result)


def get_hold_out_users(users):
    indices = np.random.choice(len(users), int(len(users) / 3), False)
    test_users = [u for indd, u in enumerate(users) if indd in indices]
    return test_users


def run():
    mex = pre_processing.MexPreprocessing()
    class_length = len(mex.activityList)

    all_data = mex.get_data()
    all_data = mex.get_features(all_data, increment=2, window=5)

    subject_ids = list(all_data.keys())
    test_subjects = get_hold_out_users(subject_ids)

    train_data = {key: value for key, value in all_data.items() if key not in test_subjects}
    test_data = {key: value for key, value in all_data.items() if key in test_subjects}
    all_data = None

    train_features, train_labels = mex.flatten(train_data, dct_length=60)
    test_features, test_labels = mex.flatten(test_data, dct_length=60)
    train_data = None
    test_data = None

    pm_train_features = mex.extract_sensor(train_features, 0)
    dc_train_features = mex.extract_sensor(train_features, 1)
    act_train_features = mex.extract_sensor(train_features, 2)
    acw_train_features = mex.extract_sensor(train_features, 3)
    pm_test_features = mex.extract_sensor(test_features, 0)
    dc_test_features = mex.extract_sensor(test_features, 1)
    act_test_features = mex.extract_sensor(test_features, 2)
    acw_test_features = mex.extract_sensor(test_features, 3)

    train_features = None
    test_features = None

    train_labels = np_utils.to_categorical(train_labels, class_length)
    test_labels = np_utils.to_categorical(test_labels, class_length)
    print(train_labels.shape)
    print(test_labels.shape)

    run_pm(pm_train_features, pm_test_features, train_labels, test_labels, class_length)
    # run_dc(dc_train_features, dc_test_features, train_labels, test_labels, class_length)
    # run_act(act_train_features, act_test_features, train_labels, test_labels, class_length)
    # run_acw(acw_train_features, acw_test_features, train_labels, test_labels, class_length)


run()
