#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:35:35 2020

@author: skay
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from intrextr import cd
import warnings
warnings.filterwarnings('always')
with warnings.catch_warnings():
            warnings.simplefilter('ignore')
dataset = pd.read_csv("/home/skay/Desktop/gwvaluesonly.csv")
dataset.head()
df = dataset.replace(np.nan,0)

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

#dataset.dropna()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.metrics import mean_squared_error


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
cc=regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred[:100]
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cutoff = 8.19                            # deciding on a cutoff limit
y_pred_classes = np.zeros_like(y_pred)    # initialise a matrix full with zeros
y_pred_classes[y_pred > cutoff] = 1       
y_test_classes = np.zeros_like(y_pred)
y_test_classes[y_test > cutoff] = 1
y_pred_classes[:100]
y_test_classes[:100]
confusion_matrix(y_test_classes, y_pred_classes)
print('Confusion matrix : ')
print(confusion_matrix(y_test_classes,y_pred_classes))
print('Classificattion Report : ')
print(classification_report(y_test_classes,y_pred_classes))
print('Accuracy of model = ', accuracy_score(y_test_classes, y_pred_classes))
mse1 = mean_squared_error(y_test, y_pred)
print("Mean squared error before replacement : ",mse1)
gg = dict(enumerate(y_pred.flatten(), 1))

gg
#import sklearn.datasets as datasets

#iris=datasets.load_iris()
##df=pd.DataFrame(iris.data, columns=iris.feature_names)
#yy=iris.target
#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#from sklearn.tree import export_graphviz
#import pydotplus
#dot_data =StringIO()
#from sklearn.datasets import make_regression
#
#X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
#regr = RandomForestRegressor(max_depth=2, random_state=0)
#regr.fit(X, y)
#regr.predict([[0, 0, 0, 0]])
#v=cc.fit(iris.data,iris.target)
#export_graphviz(v, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True)
#graph = pydotplus.graphviz.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())
# importing scikit learn with make_blobs 
#from sklearn.datasets.samples_generator import make_blobs 
#
## creating datasets X containing n_samples 
## Y containing two classes 
#X, Y = make_blobs(n_samples=500, centers=2, 
#				random_state=0, cluster_std=0.40) 
'''this whole thing works
# plotting scatters 
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring'); 
plt.show() 
# creating line space between -1 to 3.5 
xfit = np.linspace(-1, 3.5) 

# plotting scatter 
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring') 

# plot a line between the different sets of data 
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]: 
	yfit = m * xfit + b 
	plt.plot(xfit, yfit, '-k') 
	plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', 
	color='#AAAAAA', alpha=0.4) 

plt.xlim(-1, 3.5); 
plt.show() '''
a = np.array(df) 
y  = a[:,3]
#df
x = np.column_stack((df.YEAR_OBS,df.PREMON)) 
df.shape # 11728 samples and 5 features 
  
print (x),(y)
#plt.style.use('ggplot')
#df.YEAR_OBS.plot(kind='hist',color='purple',edgecolor='black',figsize=(5,5))
#plt.title('Groundwater levels on a yearly basis', size=24)
#plt.xlabel('Year', size=18)
#plt.ylabel('Pre-monsoon gw levels', size=18)
ax1= df.plot(kind='scatter', x='YEAR_OBS',y='PREMON', color='blue',alpha=0.5, figsize=(5,5))

# import support vector classifier 
dataset.describe()
dataset.head()
dataset['MONSOON'].replace(0, np.nan, inplace= True)
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
#from sklearn.linear_model import LogisticRegression
print((df[['MONSOON','POMRB','POMKH','PREMON']] == 0).sum())
print(dataset.isnull().sum())
dataset[['MONSOON','POMRB','POMKH','PREMON']] = dataset[['MONSOON','POMRB','POMKH','PREMON']].replace(0, np.NaN)
print(dataset.head(20))

dataset.info()
from pandas import isnull
y_test[:30]
b= dataset.fillna(value=dataset.mean(),inplace=True)
y_pred[:100]

print(dataset.isnull().sum())
print(b)
values = dataset.values
imputer = Imputer()
transformed_values = imputer.fit_transform(values)
print(np.isnan(transformed_values).sum())
ax2= dataset.plot(kind='scatter', x='YEAR_OBS',y='MONSOON', color='blue',alpha=0.5, figsize=(5,5))
#from sklearn.datasets import make_circles
#from numpy import where
# generate 2d classification dataset
xa = b.iloc[:, 0:4].values
xb = b.iloc[:, 4].values
Xtrain, Xtest, ytrain, ytest = train_test_split(xa, xb, test_size=0.3, random_state=0)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)
xc=regressor.fit(Xtrain, ytrain)
ypred = regressor.predict(Xtest)
y_predclasses = np.zeros_like(ypred)    # initialise a matrix full with zeros
y_predclasses[ypred > cutoff] = 1       
y_testclasses = np.zeros_like(ypred)
y_testclasses[ytest > cutoff] = 1
confusion_matrix(y_testclasses, y_predclasses)
print('Confusion matrix : ')
print(confusion_matrix(y_testclasses,y_predclasses))
print('Classification Report : ')
print(classification_report(y_testclasses,y_predclasses))
print('Accuracy of model = ', accuracy_score(y_testclasses, y_predclasses))
mse2 = mean_squared_error(ytest, ypred)
print("Mean squared error after replacement : ",mse2)

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(verbose=True)
print("Gradient boosting model before replacing with predicted values : ")

gbr1=model.fit(X_train, y_train)
print(gbr1)
print()

print("Gradient boosting model Score before replacement: ",model.score(X_test, y_test))
ypre = model.predict(X_test)
#from sklearn.metrics import f1_score
ypre = np.exp(ypre) - 1 # undo the log we took earlier

ypre[5:]
#feature_importances(model, df.columns, n=15)
print()
print()
print("Gradient boosting model after replacing with predicted values : ")
print()

gbr2=model.fit(Xtrain,ytrain)
print(gbr2)
print()
print()
print("Gradient boosting model Score after` replacement: ",model.score(Xtest,ytest))
ypr = model.predict(Xtest)
ypr[:]
print()
print('Model score')
model.score(Xtrain,ytrain)
ypredclasses = np.zeros_like(ypr)    # initialise a matrix full with zeros
ypredclasses[ypr > cutoff] = 1       
ytestclasses = np.zeros_like(ypr)
ytestclasses[ytest > cutoff] = 1

accuracy_score(ypredclasses,ytestclasses)


  
# reshape for reshaping the data into a len(X_grid)*1 array,  
# i.e. to make a column out of the X_grid value                   

  
# Scatter plot for original data 


#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import Lasso
#from yellowbrick.datasets import load_concrete
#from yellowbrick.regressor import PredictionError
#
## Load a regression dataset
#Xu, yu = load_concrete()
#
## Create the train and test data
## Instantiate the linear model and visualizer
#model2 = Lasso()
#visualizer = PredictionError(model2)
#
#visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
#visualizer.score(X_test, y_test)  # Evaluate the model on the test data
#visualizer.show()                 # Finalize and render the figure

# Fit regression model
from sklearn import ensemble
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
print()
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE of Gradient boosting before replacing values : %.4f" % mse)

# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, ypre in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, ypre)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')


clf.fit(Xtrain, ytrain)
mse = mean_squared_error(ytest, clf.predict(Xtest))
print("MSE of Gradient boosting after replacing values : %.4f" % mse)

# #############################################################################
# Plot training deviance

# compute test set deviance
#test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
#
#for i, ypr in enumerate(clf.staged_predict(Xtest)):
#    test_score[i] = clf.loss_(ytest, ypr)
#
#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.title('Deviance')
#plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
#         label='Training Set Deviance')
#plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
#         label='Test Set Deviance')
#plt.legend(loc='upper right')
#plt.xlabel('Boosting Iterations')
#plt.ylabel('Deviance')
#print() 
#print()
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, dataset.columns)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
############################################################################
#SVR
print()
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
xa = xa.reshape(-1, 1)
xb = xb.reshape(-1, 1)

xa = sc_X.fit_transform(xa)
xb = sc_y.fit_transform(xb)

svr = SVR( kernel = 'rbf')
svr.fit(Xtrain, ytrain)


yp = svr.predict(Xtest)
print(yp)
sc_y.inverse_transform(yp)

# Visualising the SVR results
'''plt.scatter(ytest,yp, color = 'red')
plt.plot(ytest, yp, color = 'blue')
plt.title('SVR Regression Model')
plt.xlabel('years')
plt.ylabel('others')
plt.show'''








