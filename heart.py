#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split # To split the actual data into training and testing sets
from sklearn.preprocessing import StandardScaler # Scaling the training and testing sets as a part of data pre-processing
from sklearn.metrics import confusion_matrix,classification_report # This is to show the accuracy and the precision score of the models that has been built
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score # accuracy metric for the models
import matplotlib.pyplot as plt # To plot the visualizations
from pandas.plotting import scatter_matrix
import seaborn as sns


# In[31]:


dataset = pd.read_csv("/Users/rakeshchoudhary/Downloads/heart.csv")
dataset


# In[32]:


print( 'Shape of DataFrame: {}'.format(dataset.shape))
print (dataset.loc[1])


# In[33]:


# To plot the histogram of the feature and Target variables
dataset.hist(figsize = (20,20))
plt.show()


# In[34]:


# To visualize the Target column which is the dependant variable
sns.countplot(x="target", data=dataset)
plt.show()


# In[35]:


#No. of Patients having Heart disease
No_Disease = len(dataset[dataset.target == 0])
Heart_Disease = len(dataset[dataset.target == 1])
print("Percentage of Patients Doesn't have Heart Disease: {:.2f}%".format((No_Disease / (len(dataset.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((Heart_Disease / (len(dataset.target))*100)))


# In[36]:


#Another Visualization to show the count of Male and Female patients
sns.countplot(x='sex', data=dataset, palette="bwr")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()


# In[37]:


#Percentage of male and female patients
F = len(dataset[dataset.sex == 0])
M = len(dataset[dataset.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((F / (len(dataset.sex))*100))) # Rounding off to 2 decimals
print("Percentage of Male Patients: {:.2f}%".format((M / (len(dataset.sex))*100)))  # Rounding off to 2 decimals


# In[38]:


pip install plotly


# In[39]:


# Another visualiation of the patiens having the heart disease or not
import plotly.graph_objs as go

column = "target"
grouped = dataset[column].value_counts().reset_index()
grouped = grouped.rename(columns = {column : "count", "index" : column})

## plot
pieplot = go.Pie(labels=grouped[column], values=grouped['count'], pull=[0.05, 0])
layout = {'title': 'Target(0 = No, 1 = Yes)'}
Target_Plot = go.Figure(data = [pieplot], layout = layout)
Target_Plot


# In[40]:


# Heart disease frequency by age
plt.figure(figsize=(15, 15))
sns.countplot(x='age', hue='target', data=dataset, palette=['blue', 'red'])
plt.legend(["No Disease", "Have Disease"])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[41]:


# To Plot the Correlation matrix visualization of the variables in the dataset
plt.figure(figsize=(10,10))
sns.heatmap(dataset.corr(),annot=True,fmt='.1f')
plt.show()


# In[42]:


# PERFORMING NORMALIZATION AND CALCULATING ACCURACY
Y = dataset.target.values
X_data = dataset.drop(['target'], 1)

# Min-Max Normalization
X = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data)).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)

#transpose matrices
X_train = X_train.T
Y_train = Y_train.T
X_test = X_test.T
Y_test = Y_test.T

#Model 1 - Logistic Regression

from sklearn.linear_model import LogisticRegression
accuracies = {}
lr = LogisticRegression()
lr.fit(X_train.T,Y_train.T)
accuracy = lr.score(X_test.T,Y_test.T)*100

accuracies['Logistic Regression'] = accuracy
print(" Accuracy of our Regression Model is  {:.2f}%".format(accuracy))


# In[43]:


Y = dataset.target.values
X_dat = dataset.drop(['target'], 1)

# Min-Max Normalization
X = (X_dat - np.min(X_dat)) / (np.max(X_dat) - np.min(X_dat)).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)

#transpose matrices
X_train = X_train.T
Y_train = Y_train.T
X_test = X_test.T
Y_test = Y_test.T


# In[44]:


# Machine Learning Algorihm to print the metrics of the scores and accuracies for both Test and training sets 

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[45]:


# Splitting of Test and Train sets

from sklearn.model_selection import train_test_split

X = dataset.drop('target', axis=1)
y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[46]:


#Decision tree classifier is a tree in which internal nodes are labeled by features
#The classifier categorizes an object by recursively testing for the weights that the features labeling the internal nodes have in vector, until a leaf node is reached

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, Y_train)

print_score(tree, X_train, Y_train, X_test, Y_test, train=True)
print_score(tree, X_train, Y_train, X_test, Y_test, train=False)


# In[47]:


pip install keras


# In[48]:


pip install tensorflow


# In[49]:


pip install tensorflow==2.0


# In[50]:


from __future__ import print_function
from keras.models import Sequential
from keras import layers
from keras.layers import Dense,Flatten


# In[51]:


# Check dimensions of both sets.
print("Train Features Size:", X_train.shape)
print("Test Features Size:", X_test.shape)
print("Train Labels Size:", Y_train.shape)
print("Test Labels Size:", Y_test.shape)


# In[52]:


#Model - 3
# Create a Neural network model
model = Sequential()


# In[53]:


model.add(Dense(256, activation='relu', input_dim = 13))
model.add(Dense(256, activation='relu', input_dim = 13))
model.add(Dense(128, activation='relu', input_dim = 13))
model.add(Dense(64, activation='relu', input_dim = 13))
model.add(Dense(32, activation='relu', input_dim = 13))
model.add(Dense(1, activation='sigmoid'))


# In[54]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[55]:


NN_model = model.fit(X_train,Y_train,validation_data=(X_test, Y_test),epochs=50)


# In[56]:


## checking the model score 
result = model.evaluate(X_test,Y_test)


# In[57]:


rmse = np.sqrt(result)
print(rmse)


# In[58]:


from matplotlib import pyplot as plt


# Plot the model accuracy vs. number of Epochs
plt.plot(NN_model.history['accuracy'])
plt.plot(NN_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'])
plt.show()

# Plot the Loss function vs. number of Epochs
plt.plot(NN_model.history['loss'])
plt.plot(NN_model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'])
plt.show()


# In[ ]:




