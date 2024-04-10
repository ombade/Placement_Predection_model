#importing the libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
tk = GaussianNB()
#importing the dataset
dataset=pd.read_csv('campus_recruitment.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,10].values

#Encoding the categorical features
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

le_x=LabelEncoder()
le_y=LabelEncoder()

X[:,0]=le_x.fit_transform(X[:,0])
X[:,6]=le_x.fit_transform(X[:,6])
X[:,8]=le_x.fit_transform(X[:,8])

le_y.fit(Y)
Y=le_y.transform(Y)

ct=ColumnTransformer([('ohe',OneHotEncoder(categories='auto'),[3,5])],remainder='passthrough')
X=ct.fit_transform(X)

#Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=0)

#Fitting the classifier
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,random_state=0)
clf.fit(X_train,Y_train)

#Predicting the results
Y_pred=clf.predict(X_test)

#Predicting the probability estimates
Y_pred_proba=clf.predict_proba(X_test)

#Analyzing the performance of the model
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_test,Y_pred)
print(cm)
print("Accuracy Score : {:.2f} %".format(accuracy_score(Y_test,Y_pred)*100))

#Applying k-Fold cross validation
from sklearn.model_selection import cross_val_score
acc=cross_val_score(estimator=clf, X=X_train, y=Y_train,cv=10)
print("Mean Accuracy : {:.2f} %".format(acc.mean()*100))
print("Standard Deviation : {:.2f} %".format(acc.std()*100))

