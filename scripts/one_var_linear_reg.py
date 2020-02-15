import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('~/workspaces/data/Weather.csv')
#Explore the data
print("dataset.shape:", dataset.shape)
print("Statistical characteristics in the data:", dataset.describe())

dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['MaxTemp'])
plt.show()

#Our next step is to divide the data into “attributes” and “labels”.
#Attributes are the independent variables while labels are dependent variables whose
# values are to be predicted. In our dataset, we only have two columns. We want to
# predict the MaxTemp depending upon the MinTemp recorded.
# Therefore our attribute set will consist of the “MinTemp” column which is stored in
# the X variable, and the label will be the “MaxTemp” column which is stored in y variable.
kk=X = dataset['MinTemp'].values
X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

#Next, we split 80% of the data to the training set while 20% of the data to test set
# using below code.
#The test_size variable is where we actually specify the proportion of the test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#As we have discussed that the linear regression model basically finds the best value
# for the intercept and slope, which results in a line that best fits the data.
# To see the value of the intercept and slope calculated by the linear regression
# algorithm for our dataset, execute the following code.

regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
#To make predictions on the test data, execute the following script:
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df
plt.figure()
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#Error metrics of the estimation
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

