### TP GUIQUERRO Nathaniel regression linéaire
### 14/03/2025

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt

#Charger les données depuis le fichier CSV de boston
data = pd.read_csv("boston.csv")

"""
function qui permet de générer un rapport html en function des données
"""
def generateReport (data):
    # Générer le rapport
    profile = ProfileReport(data, title='Boston Housing Data Report')
    profile.to_file('boston_housing_report.html')

#generateReport(data)
# on peut voir des histogramme, matrice de correlation des paramètres (ex: nombre de pièce) et son influence sur le prix
# on constate également des nuages de points répresentant l'intéranctions entre les différents paramètres.

def part4(data):
    # Generate the correlation matrix
    corr_matrix = data.corr()

    # Display the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.savefig("correlation_matrix.png")

    # Identify top 3 correlated features
    print(corr_matrix['medv'].sort_values(ascending=False).head(4))


#part4(data)
#on voit la mattrice de corellation qui est la même que dans le html ...
#est donc qu'ils y a plusieurs paramètres qui rentre dans la régression linéaire
#certains paramètres auront des 'apports' positif sur le prix et d'autres négatif
# comme le nobre de pièce (qui sera postif) ou au contraire la criminalité (qui apportera un aspect négatif sur le prix)


def part5_slr(data):
    # Select the feature and target
    X = data[['rm']]
    y = data['medv']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create the model and train it
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')


part5_slr(data)
#comment : Le modèle fait bcp d'erreur car MSE : 46 => un seul paramètre ne suffit pas pour le MEDV
#le nombre de pièce n'est pas suffisent pour expliquer le prix.

def part5_mlr(data):
    X = data.drop('medv', axis=1)
    y = data['medv']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create the model and train it
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

part5_mlr(data)
#comment : on constante qu'en ajoutant plusieurs paramètres alors le MSE est diminué de 46 à 24 quasiment de moitié
# ex de param : RM (Nombre de pièces) + prix
# ex de param : LSTAT (% de personnes à faible revenu) - prix
# ... Polution etc ...
# également le R² est doublé donc la variance est expliqué plus que avec la Simple Linear Regression

