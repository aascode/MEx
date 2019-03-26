import pre_processing
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, Flatten, BatchNormalization
from keras.models import Model
from keras.utils import np_utils
import numpy as np
import os

np.random.seed(1234)


def write_data(file_path, data):
    if (os.path.isfile(file_path)):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


def build_pm_model(class_length):
    _input = Input(shape=(75, 32, 16, 1))

    _conv = Conv3D(150, kernel_size=3, activation='relu')(_input)
    _conv = MaxPooling3D(pool_size=2, strides=2)(_conv)
    _conv = BatchNormalization()(_conv)
    _conv = Conv3D(150, kernel_size=3, activation='relu')(_conv)
    _conv = MaxPooling3D(pool_size=2, strides=2)(_conv)

    flatten = Flatten()(_conv)

    dense_layer = Dense(600, activation='tanh')(flatten)
    dense_layer = Dense(100, activation='tanh')(dense_layer)
    dense_layer = Dense(class_length, activation='softmax')(dense_layer)

    model = Model(inputs=_input, outputs=dense_layer)
    print(model.summary())
    return model


def get_hold_out_users(users):
    indices = np.random.choice(len(users), int(len(users) / 2), False)
    test_users = [u for indd, u in enumerate(users) if indd in indices]
    return test_users


def run():
    write_path = 'E:\\Mex\\results\\3.txt'
    mex = pre_processing.MexPreprocessing()

    all_data = mex.get_data()
    subject_ids = list(all_data[0].keys())
    test_subjects = get_hold_out_users(subject_ids)

    all_data = mex.get_features(all_data, increment=2, window=5)

    train_data = {key: value for key, value in all_data.items() if key not in test_subjects}
    test_data = {key: value for key, value in all_data.items() if key in test_subjects}

    class_length = len(mex.activityList)

    train_features, train_labels = mex.flatten(train_data, dct_length=60)
    test_features, test_labels = mex.flatten(test_data, dct_length=60)

    pm_train_features = mex.extract_sensor(train_features, 0)
    dc_train_features = mex.extract_sensor(train_features, 1)
    act_train_features = mex.extract_sensor(train_features, 2)
    acw_train_features = mex.extract_sensor(train_features, 3)
    pm_test_features = mex.extract_sensor(test_features, 0)
    dc_test_features = mex.extract_sensor(test_features, 1)
    act_test_features = mex.extract_sensor(test_features, 2)
    acw_test_features = mex.extract_sensor(test_features, 3)

    pm_train_features = np.array(pm_train_features)
    print(pm_train_features.shape)
    pm_train_features = np.reshape(pm_train_features, (
    pm_train_features.shape[0], pm_train_features.shape[1], pm_train_features.shape[2], 32, 16))
    print(pm_train_features.shape)
    pm_train_features = np.swapaxes(pm_train_features, 1, 4)
    print(pm_train_features.shape)
    pm_train_features = np.swapaxes(pm_train_features, 1, 2)
    print(pm_train_features.shape)
    pm_train_features = np.swapaxes(pm_train_features, 2, 3)
    print(pm_train_features.shape)
    pm_test_features = np.array(pm_test_features)
    print(pm_test_features.shape)
    pm_test_features = np.reshape(pm_test_features, (
    pm_test_features.shape[0], pm_test_features.shape[1], pm_test_features.shape[2], 32, 16))
    print(pm_test_features.shape)
    pm_test_features = np.swapaxes(pm_test_features, 1, 4)
    print(pm_test_features.shape)
    pm_test_features = np.swapaxes(pm_test_features, 1, 2)
    print(pm_test_features.shape)
    pm_test_features = np.swapaxes(pm_test_features, 2, 3)
    print(pm_test_features.shape)

    train_labels = np_utils.to_categorical(train_labels, class_length)
    test_labels = np_utils.to_categorical(test_labels, class_length)
    print(train_labels.shape)
    print(test_labels.shape)

    pm_model = build_pm_model(class_length=class_length)
    pm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    pm_model.fit(pm_train_features, train_labels, verbose=1, batch_size=50, epochs=10, shuffle=True)
    pm_score = pm_model.evaluate(pm_test_features, test_labels, batch_size=50, verbose=1)
    pm_result = str(pm_score[0]) + ',' + str(pm_score[1])
    print(pm_result)
    write_data(write_path, pm_result)


run()
