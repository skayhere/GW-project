import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('/home/aishwaryak/Desktop/gwvaluesonly.csv')
dataset.head()
df = dataset.replace(np.nan,0)
X = df.iloc[:, [0, 4]].values
X
y = df.iloc[:, 4].values
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=1, random_state=0)
cc=regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error
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
#------------------------------------
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
ax1= df.plot(kind='scatter', x='PREMON',y='MONSOON', color='blue',alpha=0.5, figsize=(5,5))
df.describe()
df.head()
print((df[['MONSOON','POMRB','POMKH','PREMON']] == 0).sum())
print(df.isnull().sum())
print(df.head(20))
df.info()

from pandas import isnull
y_test[:30]
b = df.fillna(value=df.mean(), inplace=True)
y_pred[:100]
print(df.isnull().sum())
df.fillna(dataset.median())
print(df)
print(b)
values = df.values
print(values)
ax2= dataset.plot(kind='scatter', x='YEAR_OBS',y='MONSOON', color='blue',alpha=0.5, figsize=(5,5))

print(df['MONSOON'])

##print(aa)

#xa = df1.iloc[:, [0, 4]].values
#xb = df1.iloc[:, 4].values
#Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.3, random_state=0)
#Xtrain = sc.fit_transform(Xtrain)
#Xtest = sc.transform(Xtest)
#xc=regressor.fit(Xtrain, ytrain)
#ypred = regressor.predict(Xtest)
#y_predclasses = np.zeros_like(ypred)    # initialise a matrix full with zeros
#y_predclasses[ypred > cutoff] = 1       
#y_testclasses = np.zeros_like(ypred)
#y_testclasses[ytest > cutoff] = 1
#confusion_matrix(y_testclasses, y_predclasses)
#print('Confusion matrix : ')
#print(confusion_matrix(y_testclasses,y_predclasses))
#print('Classification Report : ')
#print(classification_report(y_testclasses,y_predclasses))
#print('Accuracy of model = ', accuracy_score(y_testclasses, y_predclasses))
#mse2 = mean_squared_error(ytest, ypred)
#print("Mean squared error after replacement : ",mse2)
dff = dataset.replace(np.nan,dataset.median())
Xx = dff.iloc[:, [0, 4]].values
Xx
yy = df.iloc[:, 4].values
yy
Xtrain, Xtest, ytrain, ytest = train_test_split(Xx, yy, test_size = 0.25, random_state = 0)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)
regressor1 = RandomForestRegressor(n_estimators=1, random_state=0)
cc=regressor1.fit(Xtrain, ytrain)
ypred = regressor.predict(Xtest)
cutoff = 8.19                            # deciding on a cutoff limit
ypred_classes = np.zeros_like(ypred)    # initialise a matrix full with zeros
ypred_classes[ypred > cutoff] = 1       
ytest_classes = np.zeros_like(ypred)
ytest_classes[ytest > cutoff] = 1
ypred_classes[:100]
ytest_classes[:100]
confusion_matrix(ytest_classes, ypred_classes)
print('Confusion matrix : ')
print(confusion_matrix(ytest_classes,ypred_classes))
print('Classificattion Report : ')
print(classification_report(ytest_classes,ypred_classes))
print('Accuracy of model = ', accuracy_score(ytest_classes, ypred_classes))
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
clf1 = ensemble.GradientBoostingRegressor(**params)
clf1.fit(Xtrain, ytrain)

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


clf1.fit(Xtrain, ytrain)
mse = mean_squared_error(ytest, clf1.predict(Xtest))
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



















