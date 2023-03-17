#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:01:10 2022

@author: d

Source: https://medium.com/@sachith.prasanna90/simple-neural-network-for-classification-problem-with-pytorch-9ace5f98cdc9
"""

#%% Load libraries

import matplotlib.pyplot as plt
# import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

#%% Load data

df = sns.load_dataset("iris")

#%% Preprocessing

le =  LabelEncoder()
df['species_category'] = le.fit_transform(df['species'])

#%% First plot

colormap = {'setosa':"red", 'versicolor':"blue", 'virginica':"green"}
markers = [plt.Line2D([0,0],[0,0],color=color,marker='o',linestyle='') for color in colormap.values()]

plt.figure(figsize=(7, 5))
plt.scatter(df['sepal_length'], 
            df['sepal_width'],
            c=df['species'].map(colormap))
plt.title("Iris Visualization")
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.legend(markers, colormap.keys(), numpoints=1)
plt.tight_layout()
plt.show()

#%% Original plot

labels = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

plt.figure(figsize=(7, 5))
plt.scatter(df['sepal_length'].values, 
            df['sepal_width'].values,
            c=df['species_category'].values)
plt.colorbar(ticks=[0, 1, 2], 
              format=plt.FuncFormatter(lambda i, *args: labels[i]))
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.tight_layout()
plt.show()

#%% X-y-split

X = df.iloc[:, 0:4].values  # Input values.
y = df.iloc[:, 5].values    # Output values (species categories)

#%% Train-test-split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,  test_size=0.2, random_state=42
    )

#%% Conversion to tensors

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#%% Definition of NN-class

class NeuralNet(torch.nn.Module):
    
    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=in_features,
                                   out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120,
                                   out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84,
                                   out_features=out_features)
        
    def forward(self, X):
        X = torch.nn.functional.relu(self.fc1(X))
        X = torch.nn.functional.relu(self.fc2(X))
        return self.fc3(X)

#%% Creation of class instance

model = NeuralNet()

#%% Hyperparameters

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#%% Training

epochs = 50
losses = []

for i in range(epochs):
    i += 1
    
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    losses.append(loss)
    
    if i % 10 == 0:
        print(f'epoch: {i} -> loss: {loss}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#%% Plot learning curve

plt.figure(figsize=(7, 5))
plt.plot([loss.detach().numpy() for loss in losses])
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#%% Validation

with torch.no_grad():
    
    y_pred = model(X_test)
    preds = torch.max(y_pred, dim=1)[1]
    correct = (preds == y_test).sum()
    
print(f'{correct} out of {y_test.shape[0]} are correct : {correct.item() / y_test.shape[0] * 100}%')

#%% Validation Version II

with torch.no_grad():
    
    correct = 0
    
    for i, X in enumerate(X_test):
        
        y_pred = model(X)
        
        if y_pred.argmax().item() == y_test[i]:
            correct += 1
            
print(f'{correct} out of {y_test.shape[0]} are correct : {correct / y_test.shape[0] * 100}%')

#%% Prediction

@torch.no_grad()
def predict_unknown(X_unknown):
    
    y_pred = model(X_unknown)
    print(labels[y_pred.argmax()])
    
unknown_iris = torch.tensor([5.6, 3.7, 2.2, 0.5])

predict_unknown(unknown_iris)

#%% Plot unknown

plt.figure(figsize=(7, 5))
plt.scatter(df['sepal_length'].values, 
            df['sepal_width'].values,
            c=df['species_category'].values)
plt.colorbar(ticks=[0, 1, 2], 
             format=plt.FuncFormatter(lambda i, *args: labels[i]))
plt.scatter(unknown_iris[0], unknown_iris[1],
            c="red")
plt.arrow(x=unknown_iris[0]+1, y=unknown_iris[1]+0.5, dx=-0.75, dy=-0.4, width=.04, facecolor='red', edgecolor='none')
plt.annotate("Unknown Iris", xy = (unknown_iris[0]+1.06, unknown_iris[1]+0.5), c="red", size=13, fontweight='bold')
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.tight_layout()
plt.show()
