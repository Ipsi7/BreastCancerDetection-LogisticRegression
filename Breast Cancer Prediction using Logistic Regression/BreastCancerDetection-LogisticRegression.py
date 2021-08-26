import numpy as np
import sklearn.datasets

#getting the datasets

breast_cancer = sklearn.datasets.load_breast_cancer()
print(breast_cancer)

X = breast_cancer.data
Y = breast_cancer.target
#print(X)
#print(Y)
print(X.shape, Y.shape)

#import data to Pandas data frame

import pandas as pd
data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
print(data)
data['class'] = breast_cancer.target
data.head()
data.describe()
print(data['class'].value_counts()) #checking the class - how many benign and malignant
print(breast_cancer.target_names)  #will show the target names malignant and benign
print(data.groupby('class').mean()) #0 for malignant


#Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1) #stratify used for equal distribution
print(Y.shape,Y_train.shape, Y_test.shape)
print(Y.mean(),Y_train.mean(), Y_test.mean())
print(X.mean(),X_train.mean(), X_test.mean())


#Logistic Regression

from sklearn.linear_model import LogisticRegression

#loading the Logistic Regression model to the variable

classifier = LogisticRegression()  #loading the logistic regression model to the variable
classifier.fit(X_train, Y_train)  #training the model on training data

#evaluation of the model
#import accuracy score

from sklearn.metrics import accuracy_score

prediction_on_training_data = classifier.predict(X_train)
accuracy_on_train_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on Training Data', accuracy_on_train_data)

#accuracy on test data

prediction_on_test_data = classifier.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test Data', accuracy_on_test_data)


#Detecting whether the patient has breast cancer in benign or malignant stage

input_data = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
#change the input data to numpy array to make prediction
input_data_as_numpy_array = np.array(input_data)
print(input_data)

#reshape the array as we are predicting the output for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # 1 by 1 array that's why using(1,-1)

#prediction

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
    print("The breast cancer is Malignant")
else:
    print("The breast cancer is Benign")
