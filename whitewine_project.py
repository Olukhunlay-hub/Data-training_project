# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv("winequality-white.csv",sep=';')
dataset.rename(columns=lambda x: x.replace(" ", "_"),inplace=True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11:].values


# Find Missing or Null Data points
dataset.isnull().sum()
dataset.isna().sum()

# Plots - pair plots
#Pair plots for sample five variables
eda_colnms = [ 'volatile_acidity', 'chlorides', 'sulphates',
                                                 'alcohol','quality']
sns.set(style='whitegrid',context = 'notebook')
sns.pairplot(dataset[eda_colnms], height = 2.5, x_vars = eda_colnms,
                                              y_vars = eda_colnms)
plt.show()

""" Correlation coefficients are calculated to show the level of 
correlation in numeric terminology; these charts are used to drop
variables in the initial stage, if there are many of them to start with:"""
correlation_matrix = dataset.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

# Create Classification version of target variable
y = ["high" if a >= 7 else "low" for a in y]

y_quality = ["low" if a <= 5 else "medium" if a == 6 else "high" for a in y]

# proportion for the good and bad winequality
from collections import Counter
print(Counter(y_quality))

#Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

pca = PCA(n_components=3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# Fitting Naive Bayes classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB() 
classifier1.fit(X_train, y_train.ravel())

# Predicting the Test set results
y_pred1 = classifier1.predict(X_test)

# Estimating the performance metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))
accuracy = accuracy_score(y_test, y_pred1)*100
print('Accuracy of the model is equal to ' + str(round(accuracy, 2)) + ' %.')

# Fitting the Decision Tree Classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier2.fit(X_train, y_train.ravel())

# Predicting the Test set results
y_pred2 = classifier2.predict(X_test)

# Estimating the performance metrics
print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
accuracy = accuracy_score(y_test, y_pred2)*100
print('Accuracy of the model is equal to ' + str(round(accuracy, 2)) + ' %.')

# Fitting the KNN classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier3 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier3.fit(X_train, y_train.ravel())  

# Predicting the Test set results
y_pred3 = classifier3.predict(X_test)

# Estimating the performance metrics
print(confusion_matrix(y_test, y_pred3))
print(classification_report(y_test, y_pred3))
accuracy = accuracy_score(y_test, y_pred3)*100
print('Accuracy of the model is equal to ' + str(round(accuracy, 2)) + ' %.')

# Fitting the Logistic Regression Classifier to the dataset
from sklearn.linear_model import LogisticRegression
classifier4 = LogisticRegression(max_iter=10000 ) 
classifier4.fit(X_train, y_train.ravel())

#Predicting the Test set Results
y_pred4 = classifier4.predict(X_test)

# Estimating the performance metrics
print(confusion_matrix(y_test, y_pred4))
print(classification_report(y_test, y_pred4))
accuracy = accuracy_score(y_test, y_pred4)*100
print('Accuracy of the model is equal to ' + str(round(accuracy, 2)) + ' %.')

# Fitting the SVM classifier to the Training set
from sklearn import svm
classifier5 = svm.SVC(kernel='rbf')
classifier5.fit(X_train, y_train.ravel())

#Predicting the Test set Results
y_pred5 = classifier5.predict(X_test)

# Estimating the performance metrics
print(confusion_matrix(y_test, y_pred5))
print(classification_report(y_test, y_pred5))
accuracy = accuracy_score(y_test, y_pred5)*100
print('Accuracy of the model is equal to ' + str(round(accuracy, 2)) + ' %.')
