import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing

DATASET = 'Dataset/CSVs/dataset.csv'
SYMP_DESC = 'Dataset/CSVs/disease_Description.csv'
SYMP_PREC = 'Dataset/CSVs/disease_precaution.csv'
SYMP_WEIGHT = 'Dataset/CSVs/Symptom_severity.csv'

"""
Estrazione dei dati da un file csv
"""
def getData(type_of_file):
    
    if(type_of_file == 0):
        path = DATASET
    elif(type_of_file == 1):
        path = SYMP_DESC
    elif(type_of_file == 2):
        path = SYMP_PREC   
    elif(type_of_file == 3):
        path = SYMP_WEIGHT     

    data = pd.read_csv(path)

    return data

"""
Metodo per preprocessare il dataset, andando a fare uno shuffle e riorganizzare i dati.
"""
def datasetProcessing(dataset):

    dataset = shuffle(dataset, random_state = 35)

    dataset = dataset.fillna(0) #Termini NaN portati a 0
    
    cols = dataset.columns
    data = dataset[cols].values.flatten()

    processed = pd.Series(data)
    processed = processed.str.strip()
    processed = processed.values.reshape(dataset.shape)

    df = pd.DataFrame(processed, columns = cols)  

    return df 

"""
Metodo per codificare la severità di un sintomo all'interno del dataset
"""
def symptomInDS(dataset, symptomsDS):
    cols = dataset.columns

    values = dataset.values                                 
    symptoms = symptomsDS['Symptom'].unique() #Prendo i sintomi una sola volta

    for i in range(len(symptoms)):
        values[values == symptoms[i]] = symptomsDS[symptomsDS['Symptom'] == symptoms[i]]['weight'].values[0]

    df = pd.DataFrame(values, columns = cols)  

    #Sostituzione dei sintomi che non hanno rank con 0
    df = df.replace('dischromic__patches', 0)
    df = df.replace('spotting__urination', 0)
    df = df.replace('foul_smell_of_urine', 0)  

    return df    

def standardizeData(X_train, X_test):

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_train, X_test

"""
Metodo per rimuovere spazi bianchi tra le parole, sostituendo con un tratto basso
"""
def flatten_words(dataset):

    for col in dataset.columns:
        for i in range(len(dataset[col])):
            if (type(dataset[col][i]) == str ):
                dataset[col][i] = dataset[col][i].strip()
                dataset[col][i] = dataset[col][i].replace(" ", "_")

    return dataset   