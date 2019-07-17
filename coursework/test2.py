import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
from keras.utils import np_utils
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Model


def prepare_for_2d_cnn(_train_features, _test_features, _train_labels, _test_labels):
    _train_features = np.array(_train_features)
    _train_features = np.reshape(_train_features, (_train_features.shape[0], 32, 16))
    _train_features = np.expand_dims(_train_features, 4)
    print(_train_features.shape)

    _test_features = np.array(_test_features)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], 32, 16))
    _test_features = np.expand_dims(_test_features, 4)
    print(_test_features.shape)

    _train_labels = np_utils.to_categorical(_train_labels, 7)
    _test_labels = np_utils.to_categorical(_test_labels, 7)

    return _train_features, _test_features, _train_labels, _test_labels


def build_2d_cnn_model():
    _input = Input(shape=(32, 16, 1))
    x = Conv2D(32, kernel_size=(1, 5), activation='relu')(_input)
    x = MaxPooling2D(pool_size=2, strides=1, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(1, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=1, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(720, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(7, activation='softmax')(x)

    _model = Model(inputs=_input, outputs=x)
    return _model


seed = 1

# read all data
data = pd.read_csv('mex.csv')
print(data.shape)

# extract labels from data
y = data.iloc[:, 873]
print(y.shape)

# extract wrist accelerometer data
acw_x = data.iloc[:, 1:181]
print(acw_x.shape)

# extract thigh accelerometer data
act_x = data.iloc[:, 181:361]
print(act_x.shape)

# extract pressure mat data
pm_x = data.iloc[:, 361:873]
print(pm_x.shape)

# extract person ids
person = data.iloc[:, :1]
print(person.shape)

# extract wrist and thigh accelerometer data
wt_x = data.iloc[:, 1:361]
print(wt_x.shape)

# extract wrist, thigh and pressure mat data
wtp_x = data.iloc[:, 1:873]
print(wtp_x.shape)
'''
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance')))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='rbf', gamma=0.7, C=1.0)))
models.append(('ANN', MLPClassifier()))

# acw
print('Running algorithms for wrist accelerometer data...')
# train test split
X_train, X_test, y_train, y_test = train_test_split(acw_x, y, test_size=0.20, stratify=y, random_state=seed)

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Model: '+name)
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='macro'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred,  average='macro'))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred,  average='macro'))
    print('Test Accuracy: %.3f' % model.score(X_test, y_test))


# act
print('Running algorithms for thigh accelerometer data...')
# train test split
X_train, X_test, y_train, y_test = train_test_split(act_x, y, test_size=0.20, stratify=y, random_state=seed)

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Model: '+name)
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='macro'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred,  average='macro'))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred,  average='macro'))
    print('Test Accuracy: %.3f' % model.score(X_test, y_test))


# pm
print('Running algorithms for pressure mat data...')
# train test split
X_train, X_test, y_train, y_test = train_test_split(pm_x, y, test_size=0.20, stratify=y, random_state=seed)

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Model: '+name)
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='macro'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred,  average='macro'))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred,  average='macro'))
    print('Test Accuracy: %.3f' % model.score(X_test, y_test))


print('Running algorithms for wrist and thigh accelerometer data...')
# train test split
X_train, X_test, y_train, y_test = train_test_split(wt_x, y, test_size=0.20, stratify=y, random_state=seed)

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Model: '+name)
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='macro'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred,  average='macro'))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred,  average='macro'))
    print('Test Accuracy: %.3f' % model.score(X_test, y_test))

'''
print('Running 2D CNN model for pressure mat data...')
X_train, X_test, y_train, y_test = train_test_split(pm_x, y, test_size=0.20, stratify=y, random_state=seed)
X_train, X_test, y_train, y_test = prepare_for_2d_cnn(X_train, X_test, y_train, y_test)
_model = build_2d_cnn_model()

_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
_model.fit(X_train, y_train, verbose=1, batch_size=64, epochs=5, shuffle=True)
_predict_labels = _model.predict(X_test, batch_size=64, verbose=0)
print('Model: 2d cnn')
print('Precision: %.3f' % precision_score(y_true=y_test.argmax(axis=1), y_pred=_predict_labels.argmax(axis=1), average='macro'))
print('Recall: %.3f' % recall_score(y_true=y_test.argmax(axis=1), y_pred=_predict_labels.argmax(axis=1),  average='macro'))
print('F1: %.3f' % f1_score(y_true=y_test.argmax(axis=1), y_pred=_predict_labels.argmax(axis=1),  average='macro'))
print('Test Accuracy: %.3f' % _model.evaluate(X_test, y_test))
