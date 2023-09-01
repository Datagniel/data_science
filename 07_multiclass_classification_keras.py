#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:57:38 2022

@author: d

Source: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
"""

from keras.backend import gather
from keras.layers import Dense
from keras.models import Sequential
# from keras.wrappers.scikit_learn import KerasClassifier (original)
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
from scikeras.wrappers import KerasClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

#%% load dataset

dataframe = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
y = dataset[:,4]

print(X[:5,:])
print(y[:5])

#%% encode class values as integers

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

for group in zip(*[iter(encoded_y)] * 10):
    print(*group)

#%% convert integers to dummy variables (i.e. one hot encoded)

dummy_y = to_categorical(encoded_y)
print(dummy_y[:5,:])
print(dummy_y[50:55,:])
print(dummy_y[100:105,:])

#%% define baseline model

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# invoke instance
estimator = KerasClassifier(model=baseline_model, epochs=200, batch_size=5, verbose=0)

# define cross validation method
kfold = KFold(n_splits=10, shuffle=True)

# get results (original)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)

# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#%% record history

# K-fold Cross Validation model evaluation
histories = {'accuracy':[], 'val_accuracy':[], 'loss':[], 'val_loss':[]}

for train, test in kfold.split(X, dummy_y):
    print("- "*20)
    history = estimator.fit(
        gather(X[:,0:4], train),
        gather(dummy_y, train),
        validation_data=(gather(X[:,0:4], test),
                         gather(dummy_y, test)),
        batch_size=5,
        epochs=200
    )
    histories['accuracy'].append(history.history_['accuracy'])
    histories['val_accuracy'].append(history.history_['val_accuracy'])
    histories['loss'].append(history.history_['loss'])
    histories['val_loss'].append(history.history_['val_loss'])

#%% plot history

for key in histories.keys():
    for i in histories[key]:
        # summarize history for accuracy
        plt.plot(i)
        # plt.plot(history.history['val_accuracy'])
    plt.title(f'model {key}')
    plt.ylabel(f'{key}')
    plt.xlabel('epoch')
    plt.show()

#%% plot all acc/val_acc + loss/val_loss

for i in range(len(histories['accuracy'])):
    for key in histories.keys():
        plt.plot(histories[f'{key}'][i])
        plt.ylim(0,1.25)
        plt.title(f'Fold {i+1} - accuracy/loss')
        plt.ylabel('accuracy/loss')
        plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'], loc='best')
    plt.show()
