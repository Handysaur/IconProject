from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

"""
#######################################     HYPERPARAMETERS     ###########################################

Sono stati fatti test di tuning per poter calcolare gli iperparametri migliori per ciascun classificatore, 
in base al dataset su cui si opera.

                                  ####### RISULTATI DEL CALCOLO #######

KNN Classifier: {'leaf_size': 1, 'metric': 'minkowski', 'n_neighbors': 5, 'weights': 'distance'}
BAYESIAN Classifier: {'var_smoothing': 0.02848035868435802}
DECISION TREE Classifier: {'criterion': 'gini', 'max_depth': 17, 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}
RANDOM FOREST Classifier: {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
SVC Classifier: {'C': 0.6, 'gamma': 1, 'kernel': 'rbf'}
LOGISTIC REGRESSION Classifier: {'C': 1000.0, 'penalty': 'l2', 'solver': 'newton-cg'}


###########################################################################################################

"""

####### HYPERPARAMETERS
KNN_C_HYPERP = {'leaf_size': 1, 'metric': 'minkowski', 'n_neighbors': 5, 'weights': 'distance'}
BAYES_C_HYPERP = {'var_smoothing': 0.02848035868435802}
DECISIONTREE_C_HYPERP = {'criterion': 'gini', 'max_depth': 17, 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}
RANDOMFOREST_C_HYPERP = {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
SVC_C_HYPERP = {'C': 0.6, 'gamma': 1, 'kernel': 'rbf'}
LR_C_HYPERP = {'C': 1000.0, 'penalty': 'l2', 'solver': 'newton-cg'}


def knnClassifier(X_train, Y_train, mode):
    print("Building KNN Classifier..............")

    if mode:
        classifier = KNeighborsClassifier(**KNN_C_HYPERP)
    elif not mode:
        classifier = KNeighborsClassifier()

    classifier.fit(X_train, Y_train)
    return classifier


def bayesianClassifier(X_train, Y_train, mode):
    print("Building Bayesian Classifier..............")

    if mode:
        classifier = GaussianNB(**BAYES_C_HYPERP)
    elif not mode:
        classifier = GaussianNB()

    classifier.fit(X_train, Y_train)
    return classifier

def decisionTreeClassifier(X_train, Y_train, mode):
    print("Building Decision Tree Classifier..............")   

    if mode:    
        classifier = DecisionTreeClassifier(**DECISIONTREE_C_HYPERP)
    elif not mode:
        classifier = DecisionTreeClassifier()
          
    classifier.fit(X_train, Y_train)
    return classifier


def randomForestClassifier(X_train, Y_train, mode):
    print("Building Random Forest Classifier..............")

    if mode:     
        classifier = RandomForestClassifier(**RANDOMFOREST_C_HYPERP)
    elif not mode:
        classifier = RandomForestClassifier()

    classifier.fit(X_train, Y_train)
    return classifier


def svcClassifier(X_train, Y_train, mode):
    print("Building SVC Classifier..............")

    if mode: 
        classifier = SVC(**SVC_C_HYPERP)
    elif not mode:    
        classifier = SVC()

    classifier.fit(X_train, Y_train)
    return classifier   

@ignore_warnings(category = ConvergenceWarning)
def logisticRegressionClassifier(X_train, Y_train, mode):
    print("Building Logistic Regression Classifier..............")

    if mode: 
        classifier = LogisticRegression(**LR_C_HYPERP)
    elif not mode:
        classifier = LogisticRegression()

    classifier.fit(X_train, Y_train)

    return classifier
