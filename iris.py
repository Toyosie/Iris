#Import the libraries
import numpy as np
import pandas as pd
import pickle

#import the dataset
iris_dataset = pd.read_csv('IRIS.csv', index_col = 0)
print(iris_dataset.head())
print()
print()
print(iris_dataset.info())
print()
print()

#divide the dataset into independent and dependent variables
x = iris_dataset[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
# print(x)
y = iris_dataset[['Species']]
# print(y)

#check for missing values
print(iris_dataset.isnull().sum())

#split the data into train and test sets
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x,y,test_size=0.30, random_state=0 
) 

#encode categorical data
from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
y = encode.fit_transform(y)
print(y)

#import the decision tree classifier 
from sklearn.tree import DecisionTreeClassifier as DTC
classifier = DTC() #creates an object of the DTC class
classifier.fit(x_train,y_train)

# make predictions with the x_test dataset
y_pred = classifier.predict(x_test)

#get an accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

pickle.dump(classifier, open('iris.pkl', 'wb'))