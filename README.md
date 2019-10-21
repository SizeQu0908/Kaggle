# Kaggle-titanic
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score

path = 'C:/Users/sizeq/Downloads/train (3).csv'
path1 = 'C:/Users/sizeq/Downloads/test (3).csv'

train_data = pd.read_csv(path, header = 0)
test_data = pd.read_csv(path1, header = 0)

# pre-processing

train_data = train_data.drop(columns = ['Name'])
train_data.loc[train_data['Sex'] == 'male', ['Sex']] = 0
train_data.loc[train_data['Sex'] == 'female', ['Sex']] = 1
train_data = train_data.drop(columns = ['Ticket'])
values = {'Age' : 30}
train_data = train_data.fillna(value = values)
train_data = train_data.drop(columns = 'Cabin')

path3 = 'C:/Users/sizeq/Downloads/train_label.csv'
train_label = pd.read_csv(path3, header = 0)

# SQC survival, C-Biggest， Q-median， S-smallest ---S : 34%   Q:39.7%  C: 55.6%
train_data.loc[train_data['Embarked'] == 'S', ['Embarked']] = 0
train_data.loc[train_data['Embarked'] == 'C', ['Embarked']] = 2
train_data.loc[train_data['Embarked'] == 'Q', ['Embarked']] = 1

train_data.loc[(train_data['Fare'] == 0) & (train_data['Pclass'] == 1), ['Fare']] = 86.14887441
train_data.loc[(train_data['Fare'] == 0) & (train_data['Pclass'] == 2), ['Fare']] = 21.35866124
train_data.loc[(train_data['Fare'] == 0) & (train_data['Pclass'] == 3), ['Fare']] = 13.78787495

train_data.sort_values(by = 'Embarked')

# fit_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_data,train_label, test_size = 0.30, random_state = 100)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
train_data = train_data.drop(columns = 'Survived')

# Logistic Regression
from sklearn.linear_model import LogisticRegression
LRmodel = LogisticRegression(solver = 'lbfgs', max_iter = 400)
LRmodel.fit(x_train, y_train)
predicted = LRmodel.predict(x_test)
print(accuracy_score(y_test, predicted))

# Naive Byes
from sklearn.naive_bayes import GaussianNB
NBmodel = GaussianNB()
NBmodel.fit(x_train,y_train)
predicted= NBmodel.predict(x_test)
print('Naive Bayes',accuracy_score(y_test, predicted))

# KNN
from sklearn.neighbors import KNeighborsClassifier

Kmodel = KNeighborsClassifier()
Kmodel.fit(x_train, y_train)
predicted = Kmodel.predict(x_test)
print('KNN', accuracy_score(y_test, predicted))

# Random Forest 
from sklearn.ensemble import RandomForestClassifier

RFmodel = RandomForestClassifier(n_estimators = 100, random_state = 100)
RFmodel.fit(x_train, y_train)
predicted = RFmodel.predict(x_test)
print('Random Forest', accuracy_score(y_test, predicted))

# SVM
from sklearn.svm import SVC

model_svm = SVC(gamma = 'auto')
model_svm.fit(x_train, y_train)
predicted = model_svm.predict(x_test)
print('SVM', accuracy_score(y_test, predicted))

# decision tree
from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier()
model_tree.fit(x_train, y_train)
predicted = model_tree.predict(x_test)
print('decision tree', accuracy_score(y_test, predicted))

# XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train,y_train)
xgb_predicted = xgb.predict(x_test)
print('XGBoost',accuracy_score(y_test, xgb_predicted))

# 处理test data
test_data = test_data.drop(columns = 'Name')
test_data = test_data.drop(columns = 'Ticket')
test_data = test_data.drop(columns = 'Cabin')

test_data.loc[test_data['Sex'] == 'male', ['Sex']] = 0
test_data.loc[test_data['Sex'] == 'female', ['Sex']] = 1
test_data.sort_values(by = 'Fare')

test_data.loc[test_data['Embarked'] == 'S', ['Embarked']] = 0
test_data.loc[test_data['Embarked'] == 'C', ['Embarked']] = 2
test_data.loc[test_data['Embarked'] == 'Q', ['Embarked']] = 1

values = {'Age' : 30}
test_data = test_data.fillna(value = values)
test_data.sort_values(by = 'Fare')

values = {'Fare' : 12.45967788}
test_data = test_data.fillna(value = values)

test_data.loc[(test_data['Fare'] == 0) & (test_data['Pclass'] == 1), ['Fare']] = 94.2802972
test_data.loc[(test_data['Fare'] == 0) & (test_data['Pclass'] == 2), ['Fare']] = 22.2021043
test_data.loc[(test_data['Fare'] == 0) & (test_data['Pclass'] == 3), ['Fare']] = 12.45967788

test_data.sort_values(by = 'Fare')

# Applying Grid Search to improve the accuracy 
from sklearn.model_selection import GridSearchCV, cross_val_score

# Random Forest Classifier
rfc = RandomForestClassifier(random_state = 100)
# USING GRID SEARCH(gmm)
n_estimators = [50, 80, 100, 120, 200]
max_depth = [3,10, 30, 60, 100]
param_grid = dict(n_estimators = n_estimators, max_depth = max_depth)

grid_search_rfc = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)
grid_search_rfc.fit(train_data,np.ravel(train_label))
print(grid_search_rfc.best_score_)
rfc_best = grid_search_rfc.best_estimator_

grid_search_rfc.best_params_
rfc_best.fit(train_data, np.ravel(train_label))

# SVM with gmm
svc = SVC()

parameters = [{'kernel' : ['linear'], 'C' : [1, 10, 100]},
              {'kernel' : ['rbf'],'C':[1,10,100],'gamma':[0.05,0.0001,0.01,0.001]}]
grid_search_svc = GridSearchCV(estimator = svc, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search_svc.fit(train_data, np.ravel(train_label))
print(grid_search_svc.best_score_)
grid_search_svc.best_params_

# KNN
knn = KNeighborsClassifier()

n_neighbors = [3, 5, 7, 9, 11, 13, 15]
param_grid = dict(n_neighbors = n_neighbors)

grid_search_knn = GridSearchCV(estimator = knn, param_grid = param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)

grid_search_knn.fit(train_data, np.ravel(train_label))
print(grid_search_knn.best_score_)
grid_search_knn.best_params_


# Logistic Regression

lr = LogisticRegression()

tuned_parameters=[{'penalty':['l1','l2'],
                   'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
                    'solver':['liblinear'],
                    'multi_class':['ovr']},
                {'penalty':['l2'],
                 'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
                'solver':['lbfgs'],
                 'max_iter':[200, 300, 400, 500],
                'multi_class':['ovr','multinomial']}]

grid_search_lr = GridSearchCV(estimator = lr, param_grid = tuned_parameters, cv = 10, scoring = 'accuracy', n_jobs = -1)

grid_search_lr.fit(train_data, np.ravel(train_label))

print(grid_search_lr.best_score_)
print(grid_search_lr.best_params_)

# XGBoost n_estimator
xgb = XGBClassifier()

n_estimators = np.linspace(50, 150, 11, dtype=int)
param_grid = dict(n_estimators = n_estimators)
grid_search_XG = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)

grid_search_XG.fit(train_data, np.ravel(train_label))
print(grid_search_XG.best_score_)

grid_search_XG.best_params_
grid_search_XG.best_estimator_

# XGBoost max_depth
xgb = XGBClassifier(n_estimator = 50)

max_depth = np.linspace(1, 10, 20, dtype=int)
param_grid = dict(max_depth = max_depth)
grid_search_XG = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)

grid_search_XG.fit(train_data, np.ravel(train_label))
print(grid_search_XG.best_score_)
grid_search_XG.best_params_

# XGBoost min_child_weight
xgb = XGBClassifier(n_estimator = 50, max_depth = 2)

min_child_weight = np.linspace(1, 10, 10, dtype=int)
param_grid = dict(min_child_weight = min_child_weight)
grid_search_XG = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)

grid_search_XG.fit(train_data, np.ravel(train_label))
print(grid_search_XG.best_score_)
grid_search_XG.best_params_

# XGBoost gamma
xgb = XGBClassifier(n_estimator = 50, max_depth = 2, min_child_weight = 8)

gamma = np.linspace(0, 0.1, 11)
param_grid = dict(gamma = gamma)
grid_search_XG = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)

grid_search_XG.fit(train_data, np.ravel(train_label))
print(grid_search_XG.best_score_)
grid_search_XG.best_params_

# XGBoost subsample
xgb = XGBClassifier(n_estimator = 50, max_depth = 2, min_child_weight = 8, gamma = 0.0)

subsample =  np.linspace(0.9, 1, 11)
param_grid = dict(subsample = subsample)
grid_search_XG = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)

grid_search_XG.fit(train_data, np.ravel(train_label))
print(grid_search_XG.best_score_)
grid_search_XG.best_params_

# XGBoost colsample_bytree
xgb = XGBClassifier(n_estimator = 50, max_depth = 2, min_child_weight = 8, gamma = 0.0, subsample = 1)

colsample_bytree =  np.linspace(0, 1, 11)[1:]
param_grid = dict(colsample_bytree = colsample_bytree)
grid_search_XG = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)

grid_search_XG.fit(train_data, np.ravel(train_label))
print(grid_search_XG.best_score_)
grid_search_XG.best_params_


# XGBoost reg_lambda
xgb = XGBClassifier(n_estimator = 50, max_depth = 2, min_child_weight = 8, gamma = 0.0, subsample = 1, colsample_bytree = 0.7)

reg_lambda =  np.linspace(30, 70, 11)
param_grid = dict(reg_lambda = reg_lambda)
grid_search_XG = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)

grid_search_XG.fit(train_data, np.ravel(train_label))
print(grid_search_XG.best_score_)
grid_search_XG.best_params_


# XGBoost reg_alpha
xgb = XGBClassifier(n_estimator = 50, max_depth = 2, min_child_weight = 8, gamma = 0.0, subsample = 1, colsample_bytree = 0.7, reg_lambda = 50)

reg_alpha =  np.linspace(0, 10, 11)
param_grid = dict(reg_alpha = reg_alpha)
grid_search_XG = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)

grid_search_XG.fit(train_data, np.ravel(train_label))
print(grid_search_XG.best_score_)
grid_search_XG.best_params_

# XGBoost eta
xgb = XGBClassifier(n_estimator = 50, max_depth = 2, min_child_weight = 8, gamma = 0.0, subsample = 1, colsample_bytree = 0.7, reg_lambda = 50, reg_alpha = 0)

eta =  np.logspace(-2, 0, 10)
param_grid = dict(eta = eta)
grid_search_XG = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)

grid_search_XG.fit(train_data, np.ravel(train_label))
print(grid_search_XG.best_score_)
grid_search_XG.best_params_
xgb_best = grid_search_XG.best_estimator_
xgb_best.fit(train_data, np.ravel(train_label))

# Decision Tree
model_DD = DecisionTreeClassifier()
 
max_depth = range(1,10,1)
min_samples_leaf = range(1,10,2)
tuned_parameters = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

from sklearn.model_selection import GridSearchCV
DD = GridSearchCV(model_DD, tuned_parameters,cv=10)
DD.fit(train_data, np.ravel(train_label))
 
print("Best: %f using %s" % (DD.best_score_, DD.best_params_))

test_result = pd.DataFrame(rfc_best.predict(test_data))
test_result.to_csv('C:/Users/sizeq/Downloads/test_result.csv')











