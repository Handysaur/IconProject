from Dataset import data_Handling
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import numpy as np

####### Data processing
dataset = data_Handling.getData(0)
symp_weight = data_Handling.getData(3)

dataset = data_Handling.flatten_words(dataset)
dataset = data_Handling.datasetProcessing(dataset)

dataset = data_Handling.symptomInDS(dataset, symp_weight)
dataset.fillna(0, inplace = True)

data = dataset.iloc[:, 1:].values
labels = dataset['Disease'].values

#Divisione del dataset in training set e test set.
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size = 0.65, random_state = 35)

#X_train, X_test = data_Handling.standardizeData(X_train, X_test)

best_prms = []

####### Funzioni per il calcolo degli iperparametri migliori per ciascun classificatore
def hyP_SVC():
   parameters = {'C': [0.1, 0.2, 0.4, 0.6, 1], 'gamma': [0.01, 0.1, 0.25, 0.5, 1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
   classifier = SVC()

   #Adotto una k-fold cross validation con k=10
   grid = GridSearchCV(classifier, parameters, refit=True, verbose=2, scoring='accuracy', n_jobs=-1, cv=10)
   grid.fit(X_train,Y_train)

   best_parameters = grid.best_params_

   classifier = SVC(**best_parameters)
   classifier.fit(X_train, Y_train)
   predictions = classifier.predict(X_test)

   print(classification_report(Y_test, predictions))
   print("Accuracy:", classifier.score(X_test, Y_test))
   print(f"Best parameter: {best_parameters}")
   best_prms.append(best_parameters)

#hyP_SVC()

"""
For the examined diseases dataset with Support Vector Classification:
### Best hyperparameter
The output of the hyP_SVC function is {'C': 0.1, 'gamma': 0.25, 'kernel': 'poly'}
The classification report is
                                         precision    recall  f1-score   support

                               accuracy                           0.99      1722
                              macro avg       0.99      0.99      0.99      1722
                           weighted avg       0.99      0.99      0.99      1722

Accuracy: 0.9936120789779327

### Not good hyperparameter  
The output of the hyP_SVC function is {C=1, gamma=0.5, kernel='sigmoid'}
The classification report is
                                         precision    recall  f1-score   support
                               accuracy                           0.22      1230
                              macro avg       0.25      0.24      0.21      1230
                           weighted avg       0.24      0.22      0.19      1230

Accuracy: 0.22195121951219512                    

"""

def hyP_KNN():
   """
   Visualizzazione dell'andamento del classificatore in base al numero di vicini in fase di training e testing.
   Train_Scores = []
   Test_Scores = []
   neighbors = range(1, 30)

   for k in neighbors:
      classifier = KNeighborsClassifier(n_neighbors = k)
      classifier.fit(X_train, Y_train)
      Train_Scores.append(accuracy_score(Y_train, classifier.predict(X_train)))
      Test_Scores.append(accuracy_score(Y_test, classifier.predict(X_test)))

   plt.figure(figsize = (10, 7))

   plt.plot(neighbors, Train_Scores, label="Train score")
   plt.plot(neighbors, Test_Scores, label="Test score")
   plt.xticks(np.arange(1, 31, 1))
   plt.xlabel("Number of neighbors")
   plt.ylabel("Model score")
   plt.legend()
   plt.show()

   print(f"Maximum KNN score on the train data: {max(Train_Scores)*100:.2f}%")   
   print(f"Maximum KNN score on the test data: {max(Test_Scores)*100:.2f}%")   
   """

   parameters = { 'n_neighbors' : [5, 7, 9, 11, 13, 15, 17, 21],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan'], 
               'leaf_size' : list(range(1,30))}
   classifier = KNeighborsClassifier()

   #Adotto una k-fold cross validation con k=10
   grid = GridSearchCV(classifier, parameters, refit=True, verbose=2, scoring='accuracy', n_jobs=-1, cv=10)
   grid.fit(X_train,Y_train)                

   best_parameters = grid.best_params_
   best_score = grid.best_score_

   classifier = KNeighborsClassifier(**best_parameters)
   classifier.fit(X_train, Y_train)
   predictions = classifier.predict(X_test)

   print(classification_report(Y_test, predictions))
   print("Accuracy:", classifier.score(X_test, Y_test))
   print(f"Best score: {best_score}")
   print(f"Best parameter: {best_parameters}")

   best_prms.append(best_parameters)

#hyP_KNN()

"""
For the examined diseases dataset with KNeighborsClassifier:
### Best hyperparameter
The output of the hyP_KNN function is {'leaf_size': 1, 'metric': 'minkowski', 'n_neighbors': 5, 'weights': 'distance'}
The classification report is
                                         precision    recall  f1-score   support
                               accuracy                           0.99      1722
                              macro avg       0.99      0.99      0.99      1722
                           weighted avg       0.99      0.99      0.99      1722

Accuracy: 0.9936120789779327
Best score: 0.9953115203761754

### Not the best hyperparameter 
The output of the hyP_KNN function is {'leaf_size': 1, 'metric': 'euclidean', 'n_neighbors': 21, 'weights': 'uniform'}
The classification report is
                                         precision    recall  f1-score   support
                               accuracy                           0.94      1230
                              macro avg       0.94      0.94      0.94      1230
                           weighted avg       0.95      0.94      0.94      1230

Accuracy: 0.9422764227642276                      

"""

@ignore_warnings(category = ConvergenceWarning)
def hyP_LOGREG():
   parameters = {"C": np.logspace(-3, 3, 20), "solver": ['newton-cg', 'liblinear'], 'penalty' : ['l2']}
   classifier = LogisticRegression()

   #Adotto una k-fold cross validation con k=10
   grid = GridSearchCV(classifier, parameters, scoring = "accuracy", n_jobs=-1, verbose=2, cv = 10, refit=True)
   grid.fit(X_train, Y_train)

   best_parameter = grid.best_params_

   classifier = LogisticRegression(**best_parameter)
   classifier.fit(X_train, Y_train)
   predictions = classifier.predict(X_test)

   print(classification_report(Y_test, predictions))
   print("Accuracy:", classifier.score(X_test, Y_test))
   print(f"Best parameter: {best_parameter}")

hyP_LOGREG()

"""
For the examined diseases dataset with Logistic Regression:
### Best hyperparameter
The output of the hyP_LOGREG function is: {'C': 1000.0, 'penalty': 'l2', 'solver': 'newton-cg'}
The classification report is
                                         precision    recall  f1-score   support
                               accuracy                           0.92      1230
                              macro avg       0.93      0.93      0.92      1230
                           weighted avg       0.93      0.92      0.92      1230

Accuracy: 0.9243902439024391

### Not good hyperparameter  
The output of the hyP_LOGREG function is: {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}
The classification report is
                                         precision    recall  f1-score   support
                               accuracy                           0.83      1230
                              macro avg       0.84      0.83      0.82      1230
                           weighted avg       0.85      0.83      0.82      1230                        

"""


def hyP_DECISIONTREE():
   parameters = {"criterion":("gini", "entropy"), 
          "splitter":("best", "random"), 
          "max_depth":(list(range(1, 20))), 
          "min_samples_split":[2, 3, 4], 
          "min_samples_leaf":list(range(1, 20))
          }
   classifier = DecisionTreeClassifier(random_state=35)

   #Adotto una k-fold cross validation con k=10
   grid = GridSearchCV(classifier, parameters, scoring="accuracy", n_jobs=-1, verbose=2, cv=10)
   grid.fit(X_train, Y_train)

   best_parameters = grid.best_params_

   classifier = DecisionTreeClassifier(**best_parameters)
   classifier.fit(X_train, Y_train)
   predictions = classifier.predict(X_test)

   print(classification_report(Y_test, predictions))
   print("Accuracy:", classifier.score(X_test, Y_test))
   print(f"Best parameter: {best_parameters}")

   best_prms.append(best_parameters)

#hyP_DECISIONTREE()

"""
For the examined diseases dataset with Decision Tree:
### Best hyperparameter
The output of the hyP_DECISIONTREE function is {'criterion': 'gini', 'max_depth': 17, 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}
The classification report is
                                         precision    recall  f1-score   support
                               accuracy                           0.99      1722
                              macro avg       0.99      0.99      0.99      1722
                           weighted avg       0.99      0.99      0.99      1722

Accuracy: 0.9936120789779327


### Not good at all hyperparameter  
The output of the hyP_DECISIONTREE function is {'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 5, 'splitter': 'random'}
The classification report is
                                         precision    recall  f1-score   support
                               accuracy                           0.06      1722
                              macro avg       0.01      0.08      0.01      1722
                           weighted avg       0.00      0.06      0.01      1722

Accuracy: 0.06260162601626017     

"""

def hyP_RANDOMFOREST():
   parameters = {
      'n_estimators': [500, 900, 1100, 1500], 
      'max_depth': [2, 3, 5, 10, 15, None], 
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4]}

   classifier = RandomForestClassifier(random_state=35)

   #Adotto una k-fold cross validation con k=10
   grid = GridSearchCV(classifier, parameters, scoring="accuracy", n_jobs=-1, verbose=2, cv=10)
   grid.fit(X_train, Y_train)

   best_parameters = grid.best_params_

   classifier = RandomForestClassifier(**best_parameters)
   classifier.fit(X_train, Y_train)
   predictions = classifier.predict(X_test)

   print(classification_report(Y_test, predictions))
   print("Accuracy:", classifier.score(X_test, Y_test))
   print(f"Best parameter: {best_parameters}")
   best_prms.append(best_parameters)

#hyP_RANDOMFOREST()

"""
For the examined diseases dataset with Random Forest:
### Best hyperparameter
The output of the hyP_RANDOMFOREST function is {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 900}
                                         precision    recall  f1-score   support
                               accuracy                           0.99      1722
                              macro avg       0.99      0.99      0.99      1722
                           weighted avg       0.99      0.99      0.99      1722

Accuracy: 0.9936120789779327

### Not good hyperparameter  
The output of the hyP_RANDOMFOREST function is {'max_depth': 2, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 1500}
The classification report is
                                         precision    recall  f1-score   support
                               accuracy                           0.61      1230
                              macro avg       0.61      0.64      0.59      1230
                           weighted avg       0.60      0.61      0.58      1230

Accuracy: 0.6439024390243903   

"""

def hyP_NAIVEBAYES():
   parameter = {'var_smoothing': np.logspace(0,-9, num=100)}

   classifier = GaussianNB()

   #Adotto una k-fold cross validation con k=10
   grid = GridSearchCV(classifier, parameter, cv = 10, scoring="accuracy", n_jobs=-1, verbose=2)
   grid.fit(X_train, Y_train)

   best_parameter = grid.best_params_
   best_score = grid.best_score_

   classifier = GaussianNB(**best_parameter)
   classifier.fit(X_train, Y_train)
   predictions = classifier.predict(X_test)

   print(classification_report(Y_test, predictions))
   print("Accuracy:", classifier.score(X_test, Y_test))
   print(f"Best parameter: {best_parameter}")
   best_prms.append(best_parameter)
   
#hyP_NAIVEBAYES()

print(best_prms)

"""
For the examined diseases dataset with Naive Bayes:
### Best hyperparameter
The output of the hyP_NAIVEBAYES function is {'var_smoothing': 0.005336699231206307}
                               accuracy                           0.93      1722
                              macro avg       0.94      0.93      0.93      1722
                           weighted avg       0.94      0.93      0.93      1722

Accuracy: 0.9308943089430894

### Not good hyperparameter  
The output of the hyP_NAIVEBAYES function is {'var_smoothing': 12.74}
The classification report is
                                         precision    recall  f1-score   support
                               accuracy                           0.66      1722
                              macro avg       0.66      0.68      0.64      1722
                           weighted avg       0.66      0.66      0.63      1722

Accuracy: 0.6552845528455284 

"""