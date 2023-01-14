import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

"""
Metodo per stampare un report di un classificatore per dati di test e train
"""
def model_completeEvaluation(classifier, X_train, X_test, Y_train, Y_test, train):
    if train:
        prediction = classifier.predict(X_train)
        accuracy_s = accuracy_score(Y_train, prediction)
        #report = classification_report(Y_train, prediction)
        print("Train Result:\n================================================")        

    if not train:
        prediction = classifier.predict(X_test)
        accuracy_s = accuracy_score(Y_test, prediction)
        #report = classification_report(Y_test, prediction)
        print("Test Result:\n================================================")               

    print(f"Accuracy Score: {accuracy_s * 100:.2f}%")
    print("_______________________________________________")
    print('F1-score% =', f1_score(Y_test, prediction, average='macro')*100, '|', 'Precision% =', precision_score(Y_test, prediction, average='macro')*100)
    print("_______________________________________________")  
    #print(f"CLASSIFICATION REPORT:\n{report}")       

    return accuracy_s

"""
Metodo per confrontare i diversi classificatori tra training set e test set, andando a stampare con un grafico i vari dati di accuratezza
"""
def classifiersComparison(train_Accuracy, test_Accuracy, StdDeviation, title):
    algorithms = ('K-Nearest Neighbors', 'Naive Bayes', 'Random Forest', 'Decision Tree', 'SVC', 'Logistic Regression') 
    n_ = len(algorithms)

    fig, ax = plt.subplots(figsize=(15, 10))
    index = np.arange(n_)
    bar_width = 0.3
    opacity = 1
    rects1 = plt.bar(index, train_Accuracy, bar_width, alpha = opacity, color='darkslategrey', label='Train')
    rects2 = plt.bar(index + bar_width, test_Accuracy, bar_width, alpha = opacity, color='mediumturquoise', label='Test')
    rects3 = plt.bar(index + bar_width, StdDeviation, bar_width, alpha = opacity, color='salmon', label='Standard Deviation')
    plt.xlabel('Algorithm') # x axis label
    plt.ylabel('Accuracy (%)') # y axis label
    plt.ylim(0, 115)
    plt.title(f'Comparison of Algorithm Accuracies: {title}') 
    plt.xticks(index + bar_width * 0.5, algorithms) 
    plt.legend(loc = 'upper right') # show legend
    for index, data in enumerate(train_Accuracy):
        plt.text(x = index - 0.035, y = data + 1, s = round(data, 2), fontdict = dict(fontsize = 8))
    for index, data in enumerate(test_Accuracy):
        plt.text(x = index + 0.25, y = data + 1, s = round(data, 2), fontdict = dict(fontsize = 8))
    for index, data in enumerate(StdDeviation):
        plt.text(x = index + 0.25, y = data + 1, s = round(data, 2), fontdict = dict(fontsize = 8)) 

    plt.show()    


