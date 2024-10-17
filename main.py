# In the Name of God
# Author: Behnaz Hadi
#**********************************************************************************************************#
# Classification on breast cancer data
#**********************************************************************************************************#


############################################################################################################
# Logistic Classification method
############################################################################################################
#step 0
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
#----------------------------------------------------------------------------------------------------------#
# step 1

x = data.data
y = data.target
#----------------------------------------------------------------------------------------------------------#
# step 2
from sklearn.model_selection import KFold
kf= KFold(n_splits=4,shuffle=True,random_state=42)  # 25% data for test data in each kfold
#----------------------------------------------------------------------------------------------------------#
# step 3
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
my_params= {'C':[1,2,3], 'max_iter':[10000,20000],'solver':['lbfgs', 'newton-cg','newton-cholesky'],
            'random_state': [42]}
#----------------------------------------------------------------------------------------------------------#
#step4
from sklearn.preprocessing import MinMaxScaler
'''
scaler = MinMaxScaler() 
X_scaled = scaler.fit_transform(x)
'''
from sklearn.model_selection import GridSearchCV

gs=GridSearchCV(model,my_params ,cv=kf,scoring='accuracy')

# gs.fit(X_scaled,y)
gs.fit(x,y)
#----------------------------------------------------------------------------------------------------------#
#step5
gs.best_score_ # 0.9736161725598346   % accuracy  0.9525263468925441 
gs.best_params_
print(gs.best_score_,gs.best_params_)

x_new = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001,
         0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
         0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
y_pred = gs.predict(np.array(x_new).reshape(1,-1)) 
print(y_pred)
#----------------------------------------------------------------------------------------------------------#


############################################################################################################
# KNN method
############################################################################################################
# Step 0
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()

data.DESCR
data.data
data.feature_names
target_shape = data.target.shape
data_shape = data.data.shape
print('data_shape', data_shape,'target_shape is :' , target_shape)
#----------------------------------------------------------------------------------------------------------#
# Step 1
x= data.data
y= data.target
#----------------------------------------------------------------------------------------------------------#
# Step 2
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

kf= KFold(n_splits=4,shuffle=True,random_state=42)  # 25% data for test data in each kfold
#----------------------------------------------------------------------------------------------------------#
#step 3  train_test data
model=KNeighborsClassifier() 
my_params= { 'n_neighbors':[3,5,7,10,20],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'] }
#----------------------------------------------------------------------------------------------------------#
#step4---> #khdoe mdoel ro fit konm
from sklearn.model_selection import GridSearchCV

gs=GridSearchCV(model,my_params,cv=kf,scoring='accuracy')

gs.fit(x,y)
#----------------------------------------------------------------------------------------------------------#
#step5
gs.best_score_ # 0.9437112183591057  % accuracy
gs.best_params_
print(gs.best_score_,gs.best_params_)
x_new = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001,
         0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
         0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
y_pred = gs.predict(np.array(x_new).reshape(1,-1)) 
print(y_pred)


############################################################################################################
# Decision Tree (DT)
############################################################################################################
#step 0
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
#----------------------------------------------------------------------------------------------------------#
# step 1
x = data.data
y = data.target
#----------------------------------------------------------------------------------------------------------#
# step 2
from sklearn.model_selection import KFold
kf= KFold(n_splits=4,shuffle=True,random_state=42)  # 25% data for test data in each kfold
#----------------------------------------------------------------------------------------------------------#
# step 3
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
my_params= {'max_depth':[1,2,4,6,8],'criterion': ['gini', 'entropy','log_loss'],
            'min_samples_split':[2,3], 'min_samples_leaf':[1,2],'random_state':[42]}
#----------------------------------------------------------------------------------------------------------#
# step 4
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params ,cv=kf,scoring='accuracy')
gs.fit(x,y)
#----------------------------------------------------------------------------------------------------------#
# step 5
gs.best_score_ # 0.9367058997340687   % accuracy
gs.best_params_
print(gs.best_score_,gs.best_params_)

x_new = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001,
         0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
         0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
y_pred = gs.predict(np.array(x_new).reshape(1,-1)) 
print(y_pred)
#----------------------------------------------------------------------------------------------------------#

############################################################################################################
# Random Forest (RF)
############################################################################################################
#step 0
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
#----------------------------------------------------------------------------------------------------------#
# step 1
x = data.data
y = data.target
#----------------------------------------------------------------------------------------------------------#
# step 2
from sklearn.model_selection import KFold
kf= KFold(n_splits=4,shuffle=True,random_state=42)  # 25% data for test data in each kfold
#----------------------------------------------------------------------------------------------------------#
# step 3

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
my_params= {'n_estimators':[10,20],'criterion': ['gini', 'entropy','log_loss'],
            'min_samples_split':[2,3,4], 'min_samples_leaf':[1,2],'max_features': ['sqrt','log2'],'random_state':[42]}
        
#----------------------------------------------------------------------------------------------------------#
# step 4
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params ,cv=kf,scoring='accuracy')
gs.fit(x,y)
#----------------------------------------------------------------------------------------------------------#
# step 5
gs.best_score_ # 0.9648502905545159   % accuracy
gs.best_params_
print(gs.best_score_,gs.best_params_)

x_new = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001,
         0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
         0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
y_pred = gs.predict(np.array(x_new).reshape(1,-1)) 
print(y_pred)

#----------------------------------------------------------------------------------------------------------#

############################################################################################################
# SVM 
############################################################################################################
#step 0
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
#----------------------------------------------------------------------------------------------------------#
# step 1
x = data.data
y = data.target
#----------------------------------------------------------------------------------------------------------#
# step 2
from sklearn.model_selection import KFold
kf= KFold(n_splits=4,shuffle=True,random_state=42)  # 25% data for test data in each kfold
#----------------------------------------------------------------------------------------------------------#
# step 3

from sklearn.svm import SVC
model=SVC()
'''
my_params= {'kernel':['poly','rbf','sigmoid','precomputed'],'C': [0.1,1,10],
            'gamma':['scale','auto'], 'degree':[2,3,4],'random_state':[42]}
'''
my_params= {'kernel':['poly','rbf'],'C': [0.001,0.01,1,100,1000],
            'gamma':['scale'], 'degree':[2,3,4],'random_state':[42]}

#----------------------------------------------------------------------------------------------------------#
# step 4
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params ,cv=kf,scoring='accuracy')
gs.fit(x,y)
#----------------------------------------------------------------------------------------------------------#
# step 5
gs.best_score_ # 0.9490052201319807    % accuracy
gs.best_params_
print(gs.best_score_,gs.best_params_)

x_new = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001,
         0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
         0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
y_pred = gs.predict(np.array(x_new).reshape(1,-1)) 
print(y_pred)
#----------------------------------------------------------------------------------------------------------#


####################################################################################################################################
#                                                   Conclusion
####################################################################################################################################
# The table below shows the optimal accuracy of each approach and the optimal model parameters:

#----------------------------------------------------------------------------------------------------------------------------------#  
# Method                 Best score (Accuracy)                           Best model parameters
#----------------------------------------------------------------------------------------------------------------------------------#  
# LR                     0.9525263468925441                  {'C': 3, 'max_iter': 10000, 'random_state': 42, 'solver': 'lbfgs'}
# KNN                    0.9437112183591057                  {'metric': 'manhattan', 'n_neighbors': 10}
# DT                     0.9367058997340687                  {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'random_state': 42}
# RF                     0.9648502905545159                  {'criterion': 'entropy', 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 20, 'random_state': 42}
# SVM                    0.9490052201319807                  {'C': 1000, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf', 'random_state': 42}

# According on the available data, the Random Forest algorithm delivers the best results.


