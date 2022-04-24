import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 

# on récupère les donnees de fichier en utilisant la bibliotheque pandas
def LoadFile(file):
    data = pd.read_csv(file)
    return data 
# fonction d'activation sigmoïde 
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# fonction d'activation tangent hyperbolic
def tanh(x):
    return np.exp(-x)-np.exp(-x)/np.exp(-x)+ np.exp(-x)

# prétraitement les données 
def DataPreprocessing(data):
     # X (les entrees) => colonne Sex & Nbsem
    X = data.iloc[:,0:2]
    # Y (la sortie) => colonne PoidsBB
    Y = data.iloc[:,2:3]
    # convertir la colonne Sexe dans x de M F à des données binaires (0/1) pour pouvoir les manipuler dans l'apprentissage
    convert = preprocessing.LabelEncoder()
    X.iloc[:,0:1] = X.apply(convert.fit_transform)
    # on utilise les données 1 à 350 pour l'apprentissage et le reste pour le test de validation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.295)
    return X_train, X_test, Y_train, Y_test 

# on applique l'algorithme MLPClassifier (perceptron multicouche) pour predire les poids des bebe
def learning(X_train, X_test, Y_train, Y_test ):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test) 
    """   
    la fonction MLPClassifier() est une fonction pédfinie sous Python dans la classe MLPClassifier elle prend en paramètres la fonction d'activation tanh
    et une couche cachée à deux neuronnes et le nombre d'iteraion """
    mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(len(X_train), len(Y_train)), max_iter=1000)  
    mlp.fit(X_train, Y_train.values.ravel()) 
    predictions = mlp.predict(X_test) 
    return predictions

# main 
def main():
    data = LoadFile('data/bebe.csv')
    X_train, X_test, Y_train, Y_test = DataPreprocessing(data)
    predictions = learning(X_train, X_test, Y_train, Y_test )
    print('voici les poids predit : \n', predictions)
    
if __name__ == "__main__":
    print("veuillez patienter quelques instants")

    main()