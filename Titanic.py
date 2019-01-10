
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('train.csv')
test1 = pd.read_csv('test.csv')
test = pd.read_csv('test.csv')


train = train.dropna(subset=['Embarked'],how='any')
train_x = train.iloc[:,[2,4,5,6,7,9,11]].values
train_y = train.iloc[:,1].values
sns.set()


sns.countplot(x='Embarked',hue = 'Survived',data = train)
plt.ylabel('Survived Count')
plt.show()
#finding the nuber of missing values
train.isnull().sum()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train_x[:,1] = labelencoder.fit_transform(train_x[:,1])

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(train_x[:,2:3])
train_x[:,2:3] = imputer.transform(train_x[:,2:3])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
train_x[:,6] = labelencoder.fit_transform(train_x[:,6])
onehotencoder = OneHotEncoder(categorical_features=[6])
train_x = onehotencoder.fit_transform(train_x).toarray()

train_x = train_x[:,1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(train_x, train_y)

'''
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(train_x, train_y)
'''

from sklearn.model_selection import GridSearchCV
parameter=[{'n_neighbors':[5,8,10,12,15,18,20,25,30,35]}] 
grid_search=GridSearchCV(estimator=classifier,param_grid=parameter,scoring='accuracy',cv=10)
grid_search=grid_search.fit(train_x,train_y)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_


''' test set '''


test = test.iloc[:,[1,3,4,5,6,8,10]].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
test[:,1] = labelencoder.fit_transform(test[:,1])

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(test[:,2:3])
test[:,2:3] = imputer.transform(test[:,2:3])

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(test[:,5:6])
test[:,5:6] = imputer.transform(test[:,5:6])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
test[:,6] = labelencoder.fit_transform(test[:,6])
onehotencoder = OneHotEncoder(categorical_features=[6])
test = onehotencoder.fit_transform(test).toarray()

test = test[:,1:]

test = sc.transform(test)

y_pred = classifier.predict(test)


submission = pd.DataFrame({"PassengerId":test1["PassengerId"],
                           "Survived":y_pred})
submission.to_csv('titanic.csv',index = False)