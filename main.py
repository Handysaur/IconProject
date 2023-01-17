import classifiers
from Dataset import data_Handling
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

dataset = data_Handling.getData(0)
dis_desc = data_Handling.getData(1)
dis_prec = data_Handling.getData(2)
symp_weight = data_Handling.getData(3)

dis_desc = dis_desc.to_numpy()
dis_prec = dis_prec.to_numpy()

dataset = data_Handling.flatten_words(dataset)
dataset = data_Handling.datasetProcessing(dataset)

dataset = data_Handling.symptomInDS(dataset, symp_weight)
dataset.fillna(0, inplace = True)

#Divisione del dataset in feature e label. In particolare le feature saranno il nostro input, dunque saranno i sintomi di una malattia. Le label l'output della classificazione, dunque la malattia.
data = dataset.iloc[:, 1:].values
labels = dataset['Disease'].values

#Divisione del dataset in training set e test set. La divisione adottata sarà del 65% per il training set e il 35% per il test set
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size = 0.65, random_state = 35)

#Classificatori non dotati di iperparametro
KNN_Classifier_Unhyper = classifiers.knnClassifier(X_train, Y_train, False)
BAYESIAN_Classifier_Unhyper = classifiers.bayesianClassifier(X_train, Y_train, False)
RF_Classifier_Unhyper = classifiers.randomForestClassifier(X_train, Y_train, False)
DECISIONTREE_Classifier_Unhyper = classifiers.decisionTreeClassifier(X_train, Y_train, False)
SVC_Classifier_Unhyper = classifiers.svcClassifier(X_train, Y_train, False)
LOGREG_Classifier_Unhyper = classifiers.logisticRegressionClassifier(X_train, Y_train, False)

#Classificatori dotati di iperparametro migliore
KNN_Classifier_Hyper = classifiers.knnClassifier(X_train, Y_train, True)
BAYESIAN_Classifier_Hyper = classifiers.bayesianClassifier(X_train, Y_train, True)
RF_Classifier_Hyper = classifiers.randomForestClassifier(X_train, Y_train, True)
DECISIONTREE_Classifier_Hyper = classifiers.decisionTreeClassifier(X_train, Y_train, True)
SVC_Classifier_Hyper = classifiers.svcClassifier(X_train, Y_train, True)
LOGREG_Classifier_Hyper = classifiers.logisticRegressionClassifier(X_train, Y_train, True)

### Confronto in modalità batch dei vari classificatori ###
"""classifier_List = [KNN_Classifier_Unhyper, KNN_Classifier_Hyper, 
                    BAYESIAN_Classifier_Unhyper, BAYESIAN_Classifier_Hyper, 
                    RF_Classifier_Unhyper, RF_Classifier_Hyper, 
                    DECISIONTREE_Classifier_Unhyper, DECISIONTREE_Classifier_Hyper, 
                    SVC_Classifier_Unhyper, SVC_Classifier_Hyper, 
                    LOGREG_Classifier_Unhyper, LOGREG_Classifier_Hyper]"""

"""
#######################################     KNN      ###########################################

print("\nClassificatore KNN: Senza uso di iperparametri otimizzati")

predict_Unhyper_Train = metrics.model_completeEvaluation(KNN_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, True)
predict_Unhyper_Test = metrics.model_completeEvaluation(KNN_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, False)

KNN_train_Unhyper = predict_Unhyper_Train * 100
KNN_test_Unhyper = predict_Unhyper_Test * 100

Unhyper_results = pd.DataFrame(data=[["KNN", KNN_train_Unhyper, KNN_test_Unhyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

                        #########################################
print("\n#########################################\n")                        
print("Classificatore KNN: Con uso di iperparametri otimizzati")

predict_Hyper_Train = metrics.model_completeEvaluation(KNN_Classifier_Hyper, X_train, X_test, Y_train, Y_test, True)
predict_Hyper_Test = metrics.model_completeEvaluation(KNN_Classifier_Hyper, X_train, X_test, Y_train, Y_test, False)

KNN_train_Hyper = predict_Hyper_Train * 100
KNN_test_Hyper = predict_Hyper_Test * 100

Hyper_results = pd.DataFrame(data=[["Hyperparameter KNN", KNN_train_Hyper, KNN_test_Hyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])


print("\n\n\n")

################################################################################################

#######################################     BAYES      ###########################################

print("\nClassificatore Bayes: Senza uso di iperparametri otimizzati")

predict_Unhyper_Train = metrics.model_completeEvaluation(BAYESIAN_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, True)
predict_Unhyper_Test = metrics.model_completeEvaluation(BAYESIAN_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, False)

BAYES_train_Unhyper = predict_Unhyper_Train * 100
BAYES_test_Unhyper = predict_Unhyper_Test * 100

Unhyper_results2 = pd.DataFrame(data=[["Naive Bayes", BAYES_train_Unhyper, BAYES_test_Unhyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
Unhyper_results = Unhyper_results.append(Unhyper_results2, ignore_index=True)

                        #########################################
print("\n#########################################\n")                        
print("Classificatore Bayes: Con uso di iperparametri otimizzati")

predict_Hyper_Train = metrics.model_completeEvaluation(BAYESIAN_Classifier_Hyper, X_train, X_test, Y_train, Y_test, True)
predict_Hyper_Test = metrics.model_completeEvaluation(BAYESIAN_Classifier_Hyper, X_train, X_test, Y_train, Y_test, False)

BAYES_train_Hyper = predict_Hyper_Train * 100
BAYES_test_Hyper = predict_Hyper_Test * 100

Hyper_results2 = pd.DataFrame(data=[["Hyperparameter Naive Bayes", BAYES_train_Hyper, BAYES_test_Hyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
Hyper_results = Hyper_results.append(Hyper_results2, ignore_index=True)

print("\n\n\n")

##################################################################################################

#######################################     RANDOM FOREST      ###########################################

print("\nClassificatore Random Forest: Senza uso di iperparametri otimizzati")

predict_Unhyper_Train = metrics.model_completeEvaluation(RF_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, True)
predict_Unhyper_Test = metrics.model_completeEvaluation(RF_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, False)

RF_train_Unhyper = predict_Unhyper_Train * 100
RF_test_Unhyper = predict_Unhyper_Test * 100

Unhyper_results2 = pd.DataFrame(data=[["Random Forest", RF_train_Unhyper, RF_test_Unhyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
Unhyper_results = Unhyper_results.append(Unhyper_results2, ignore_index=True)

                        #########################################
print("\n#########################################\n")                        
print("Classificatore Random Forest: Con uso di iperparametri otimizzati")

predict_Hyper_Train = metrics.model_completeEvaluation(RF_Classifier_Hyper, X_train, X_test, Y_train, Y_test, True)
predict_Hyper_Test = metrics.model_completeEvaluation(RF_Classifier_Hyper, X_train, X_test, Y_train, Y_test, False)

RF_train_Hyper = predict_Hyper_Train * 100
RF_test_Hyper = predict_Hyper_Test * 100

Hyper_results2 = pd.DataFrame(data=[["Hyperparameter Random Forest", RF_train_Hyper, RF_test_Hyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
Hyper_results = Hyper_results.append(Hyper_results2, ignore_index=True)

print("\n\n\n")

##########################################################################################################

#######################################     DECISION TREE      ###########################################

print("\nClassificatore Decision Tree: Senza uso di iperparametri otimizzati")

predict_Unhyper_Train = metrics.model_completeEvaluation(DECISIONTREE_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, True)
predict_Unhyper_Test = metrics.model_completeEvaluation(DECISIONTREE_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, False)

DT_train_Unhyper = predict_Unhyper_Train * 100
DT_test_Unhyper = predict_Unhyper_Test * 100

Unhyper_results2 = pd.DataFrame(data=[["Decision Tree", DT_train_Unhyper, DT_test_Unhyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
Unhyper_results = Unhyper_results.append(Unhyper_results2, ignore_index=True)

                        #########################################
print("\n#########################################\n")                        
print("Classificatore Decision Tree: Con uso di iperparametri otimizzati")

predict_Hyper_Train = metrics.model_completeEvaluation(DECISIONTREE_Classifier_Hyper, X_train, X_test, Y_train, Y_test, True)
predict_Hyper_Test = metrics.model_completeEvaluation(DECISIONTREE_Classifier_Hyper, X_train, X_test, Y_train, Y_test, False)

DT_train_Hyper = predict_Hyper_Train * 100
DT_test_Hyper = predict_Hyper_Test * 100

Hyper_results2 = pd.DataFrame(data=[["Hyperparameter Decision Tree", DT_train_Hyper, DT_test_Hyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
Hyper_results = Hyper_results.append(Hyper_results2, ignore_index=True)

print("\n\n\n")

##########################################################################################################

#######################################     SVC      ###########################################

print("\nClassificatore Support Vector Classifier: Senza uso di iperparametri otimizzati")

predict_Unhyper_Train = metrics.model_completeEvaluation(SVC_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, True)
predict_Unhyper_Test = metrics.model_completeEvaluation(SVC_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, False)

SVC_train_Unhyper = predict_Unhyper_Train * 100
SVC_test_Unhyper = predict_Unhyper_Test * 100

Unhyper_results2 = pd.DataFrame(data=[["Support Vector Classifier", SVC_train_Unhyper, SVC_test_Unhyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
Unhyper_results = Unhyper_results.append(Unhyper_results2, ignore_index=True)

                        #########################################
print("\n#########################################\n")                        
print("Classificatore Support Vector Classifier: Con uso di iperparametri otimizzati")

predict_Hyper_Train = metrics.model_completeEvaluation(SVC_Classifier_Hyper, X_train, X_test, Y_train, Y_test, True)
predict_Hyper_Test = metrics.model_completeEvaluation(SVC_Classifier_Hyper, X_train, X_test, Y_train, Y_test, False)

SVC_train_Hyper = predict_Hyper_Train * 100
SVC_test_Hyper = predict_Hyper_Test * 100

Hyper_results2 = pd.DataFrame(data=[["Hyperparameter Support Vector Classifier", SVC_train_Hyper, SVC_test_Hyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
Hyper_results = Hyper_results.append(Hyper_results2, ignore_index=True)

print("\n\n\n")

################################################################################################

#######################################     LOGISTIC REGRESSION      ###########################################

print("\nClassificatore Logistic Regression: Senza uso di iperparametri otimizzati")

predict_Unhyper_Train = metrics.model_completeEvaluation(LOGREG_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, True)
predict_Unhyper_Test = metrics.model_completeEvaluation(LOGREG_Classifier_Unhyper, X_train, X_test, Y_train, Y_test, False)

LOGREG_train_Unhyper = predict_Unhyper_Train * 100
LOGREG_test_Unhyper = predict_Unhyper_Test * 100

Unhyper_results2 = pd.DataFrame(data=[["Logistic Regression", LOGREG_train_Unhyper, LOGREG_test_Unhyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
Unhyper_results = Unhyper_results.append(Unhyper_results2, ignore_index=True)

                        #########################################
print("\n#########################################\n")                        
print("Classificatore Logistic Regression: Con uso di iperparametri otimizzati")

predict_Hyper_Train = metrics.model_completeEvaluation(LOGREG_Classifier_Hyper, X_train, X_test, Y_train, Y_test, True)
predict_Hyper_Test = metrics.model_completeEvaluation(LOGREG_Classifier_Hyper, X_train, X_test, Y_train, Y_test, False)

LOGREG_train_Hyper = predict_Hyper_Train * 100
LOGREG_test_Hyper = predict_Hyper_Test * 100

Hyper_results2 = pd.DataFrame(data=[["Hyperparameter Logistic Regression", LOGREG_train_Hyper, LOGREG_test_Hyper]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
Hyper_results = Hyper_results.append(Hyper_results2, ignore_index=True)

print("\n\n\n")

################################################################################################################

#######################################     CLASSIFIERS COMPARISON     ###########################################

## Without Hyperparameters
train_Classifiers_Accuracies = ((KNN_train_Unhyper / 100).mean() * 100, (BAYES_train_Unhyper / 100).mean() * 100, (RF_train_Unhyper / 100).mean() * 100, (DT_train_Unhyper / 100).mean() * 100, (SVC_train_Unhyper / 100).mean() * 100, (LOGREG_train_Unhyper / 100).mean() * 100)
test_Classifiers_Accuracies = ((KNN_test_Unhyper / 100).mean() * 100, (BAYES_test_Unhyper / 100).mean() * 100, (RF_test_Unhyper / 100).mean() * 100, (DT_test_Unhyper / 100).mean() * 100, (SVC_test_Unhyper / 100).mean() * 100, (LOGREG_test_Unhyper / 100).mean() * 100)
Classifiers_StdDeviation = ((KNN_test_Unhyper / 100).std() * 100, (BAYES_test_Unhyper / 100).std() * 100, (RF_test_Unhyper / 100).std() * 100, (DT_test_Unhyper / 100).std() * 100, (SVC_test_Unhyper / 100).std() * 100, (LOGREG_test_Unhyper / 100).std() * 100)
metrics.classifiersComparison(train_Classifiers_Accuracies, test_Classifiers_Accuracies, Classifiers_StdDeviation, 'No Hyperparameters') 

## With Hyperparameters
train_Classifiers_Accuracies = ((KNN_train_Hyper / 100).mean() * 100, (BAYES_train_Hyper / 100).mean() * 100, (RF_train_Hyper / 100).mean() * 100, (DT_train_Hyper / 100).mean() * 100, (SVC_train_Hyper / 100).mean() * 100, (LOGREG_train_Hyper / 100).mean() * 100)
test_Classifiers_Accuracies = ((KNN_test_Hyper / 100).mean() * 100, (BAYES_test_Hyper / 100).mean() * 100, (RF_test_Hyper / 100).mean() * 100, (DT_test_Hyper / 100).mean() * 100, (SVC_test_Hyper / 100).mean() * 100, (LOGREG_test_Hyper / 100).mean() * 100)
Classifiers_StdDeviation = ((KNN_test_Hyper / 100).std() * 100, (BAYES_test_Hyper / 100).std() * 100, (RF_test_Hyper / 100).std() * 100, (DT_test_Hyper / 100).std() * 100, (SVC_test_Hyper / 100).std() * 100, (LOGREG_test_Hyper / 100).std() * 100)
metrics.classifiersComparison(train_Classifiers_Accuracies, test_Classifiers_Accuracies, Classifiers_StdDeviation, 'With Hyperparameters') 

#TODO Fare i grafici condivisi di test e train, con e senze iperparametri. Quindi due plot: uno per il train dove si confronta per ogni class. con e senza iperparametri e l'altro plot del test dove si fa la stessa cosa

print(Unhyper_results)
print("\n")
print(Hyper_results)

###################################################################################################################

"""


def pred_Disease(classifier, symptoms):
    all_symptoms = np.array(symp_weight["Symptom"])
    symptoms_weight = np.array(symp_weight["weight"])

    user_symp = symptoms

    for j in range(len(user_symp)):
        for k in range(len(all_symptoms)):
            if user_symp[j] == all_symptoms[k]:
                user_symp[j] = symptoms_weight[k]   

    Y_ = [user_symp]

    prediction = classifier.predict(Y_) 

    #Disease description creation, GUI Ready
    disease_Description = ""
    disease_Precautions = []

    for disease in dis_desc:
        if disease[0] == prediction[0]:
            disease_Description = disease[1]    

    for disease in dis_prec:
        if disease[0] == prediction[0]:
            disease_Precautions.append(str(disease[1]))
            disease_Precautions.append(str(disease[2]))  
            disease_Precautions.append(str(disease[3]))
            disease_Precautions.append(str(disease[4]))          

    guiPredict_Msg = "Dati i sintomi immessi, è stata predetta la seguente malattia: {" + prediction[0] + "}\n"
    guiPredict_Msg += "Descrizione malattia:\n{" + disease_Description + "}\n"
    guiPredict_Msg += "-------------------------------------------------\n"
    guiPredict_Msg += "Si consiglia di adottare le seguenti precauzioni:\n"
    guiPredict_Msg += "     - {" + disease_Precautions[0] + "}\n"
    guiPredict_Msg += "     - {" + disease_Precautions[1] + "}\n"
    guiPredict_Msg += "     - {" + disease_Precautions[2] + "}\n"
    guiPredict_Msg += "     - {" + disease_Precautions[3] + "}\n"
    guiPredict_Msg += "-------------------------------------------------\n"         
    
    #print(guiPredict_Msg)

    print("Prediction: ", prediction[0])  
       
    return guiPredict_Msg 

### Test in modalità batch del predict ###
"""

#symptoms_List = symp_weight["Symptom"].to_list()
#print(symptoms_List)
symptyomsssss = ['shivering', 'chills', 'watering_from_eyes', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     #Allergy
symptyomsssss2 = ['stomach_pain', 'acidity', 'vomiting', 'cough', 'chest_pain', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     #GERD
symptyomsssss3 = ['vomiting', 'yellowish_skin', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes', 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0]    #Chronic cholestasis
symptyomsssss4 = ['vomiting', 'indigestion', 'loss_of_appetite', 'abdominal_pain', 'passage_of_gases',0,0,0,0,0,0,0,0,0,0,0,0]#Peptic ulcer diseae
symptyomsssss5 = ['vomiting', 'sunken_eyes', 'dehydration', 'diarrhoea',0,0,0,0,0,0,0,0,0,0,0,0,0] #Gastroenteritis
symptyomsssss6 = ['headache', 'chest_pain', 'loss_of_balance', 'lack_of_concentration',0,0,0,0,0,0,0,0,0,0,0,0,0] #Hypertension
symptyomsssss7 = ['acidity', 'indigestion', 'headache', 'excessive_hunger', 'stiff_neck', 'depression', 'irritability', 'visual_disturbances',0,0,0,0,0,0,0,0,0] #Migraine
symptyomsssss8 = ['stomach_pain', 'chills', 'nodal_skin_eruptions', 'muscle_weakness', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #Paralysis (brain hemorrhage)

i = 0
for classifier in classifier_List:
    print(symptyomsssss)
    print(classifier)
    if i % 2 == 0:
        print("######## Unhyper ######## ")
        pred_Disease(classifier, symptyomsssss)
        pred_Disease(classifier, symptyomsssss2)
        pred_Disease(classifier, symptyomsssss3)
        pred_Disease(classifier, symptyomsssss4)
        pred_Disease(classifier, symptyomsssss5)
        pred_Disease(classifier, symptyomsssss6)
        pred_Disease(classifier, symptyomsssss7)
        pred_Disease(classifier, symptyomsssss8)        
    else:
        print("######## Hyper ######## ")
        pred_Disease(classifier, symptyomsssss)
        pred_Disease(classifier, symptyomsssss2)
        pred_Disease(classifier, symptyomsssss3)
        pred_Disease(classifier, symptyomsssss4)
        pred_Disease(classifier, symptyomsssss5)
        pred_Disease(classifier, symptyomsssss6)
        pred_Disease(classifier, symptyomsssss7)
        pred_Disease(classifier, symptyomsssss8)  

        print("\n")    

    i += 1

"""