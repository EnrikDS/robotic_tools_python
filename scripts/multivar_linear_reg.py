import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('C:/workspaces/robotics_tools_python/data/winequality.csv')
# Let’s explore the data a little bit by checking the number of rows and columns in it.
print("shape:", dataset.shape)
'''
It will give (1599, 12) as output which means our dataset has 1599 rows and 12 columns.
To see the statistical details of the dataset, we can use describe()
'''
print(dataset.describe())
'''
Let us clean our data little bit, So first check which are the columns the contains NaN values in it
'''
dataset.isnull().any()
'''
Once the above code is executed, all the columns should give False, In case for any column you find True result,
 then remove all the null values from that column using below code.
'''
dataset = dataset.fillna(method='ffill')
'''
Our next step is to divide the data into “attributes” and “labels”.
 X variable contains all the attributes/features and y variable contains labels.
'''
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates', 'alcohol']].values

y = dataset['quality'].values
plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.distplot(dataset['quality'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
'''
As said earlier, in the case of multivariable linear regression, the regression model has to find the most optimal coefficients for all the attributes. 
To see what coefficients our regression model has chosen, execute the following script
'''
coeff_df = pd.DataFrame(regressor.coef_, dataset.columns[0:-1],
                        columns=['Coefficient'])
print(coeff_df)
'''
This means that for a unit increase in “density”, there is a decrease of 31.51 
units in the quality of the wine. Similarly, a unit decrease in “Chlorides“ results
 in an increase of 1.87 units in the quality of the wine. We can see that the rest 
 of the features have a very little effect on the quality of the wine.
'''
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1 = df.head(25)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
'''
As we can observe here that our model has returned pretty good prediction results.

The final step is to evaluate the performance of the algorithm. We’ll do this by
 finding the values for MAE, MSE, and RMSE. Execute the following script:'''
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

'''
You can see that the value of root mean squared error is 0.62, which is slightly 
greater than 10% of the mean value of the quality, which is 5.63
This means that our algorithm was not very accurate but can still make reasonably 
good predictions.

There are many factors that may have contributed to this inaccuracy, for example :

Need more data: We need to have a huge amount of data to get the best possible prediction.
Bad assumptions: We made the assumption that this data has a linear relationship, 
but that might not be the case. Visualizing the data may help you determine that.
Poor features: The features we used may not have had a high enough correlation 
to the values we were trying to predict.'''