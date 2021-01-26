import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import os

folder = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))
dataset = pd.read_csv(folder+'/Winequality.csv')
#Print number of rows and columns
print(dataset.shape)
#To print statistical details about the data
print(dataset.describe())
dataset.isnull().any()
dataset = dataset.fillna(method='ffill')
#Divide the data between X and Y
X = pd.DataFrame(dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates','alcohol']].values, columns=['fixed acidity', 'volatile '
                                                                             'acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates','alcohol'])
y = dataset['quality'].values
#Check the quality
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['quality'])

#Split the data
#Next, we split 80% of the data to the training set while 20% of
# the data to test set
# using below code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Now lets train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df
#Lets see the predicted values for the test
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
#Now lets plt the difference
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#The final step is to evaluate the performance of the algorithm.
# We’ll do this by finding the values for MAE, MSE, and RMSE.
# Execute the following script:
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))