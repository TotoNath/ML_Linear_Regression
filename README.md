# Rapport de réponses TP : Régression Linéaire
### Rapport de : GUIQUERRO Nathaniel
### fait le : 14/03/2025

## Partie 1 :
Je setup mon environment sur Intellij IDEA Ultimate 2025
> venv
> en utilisant les \n imports suivant\
> import pandas as pd\
from sklearn.linear_model import LinearRegression\
from sklearn.metrics import mean_squared_error, r2_score\
from sklearn.model_selection import train_test_split\
from ydata_profiling import ProfileReport\
import seaborn as sns\
import matplotlib.pyplot as plt\

## Partie 2 :
on importe via le csv sur myges car le dataset n'est plus présent
dans les nouvelles version à cause d'un problème d'éthique apparement

## Partie 3 :

On explore le fichier HTML crée je constate qu'il y a plusieurs parramètres qui sont analysé
- RM : Nombre de pièce  (aspect probablement positif sur le prix)
- LSTAT : % de personne a faible revenu (influence probablement negatif)
- NOX : polution (négatif)
- DIS : distance par rapport a l'emploi (peut être positf mais négatif aussi d'après moi)
et d'autres paramètres et leurs aspect de corélation sur le MEDV


## Partie 4 :
4.2 il y a 3 paramètres qui influe le plus : 
```
rm      0.695360
zn      0.360445
b       0.333461
```
et donc leurs influence sur le medv

donc rm => positif sur le prix
zn => negatif
b => negatif

## Partie 5:

### SLR
comment : Le modèle fait bcp d'erreur car MSE : 46 => un seul paramètre ne suffit pas pour le MEDV
le nombre de pièce n'est pas suffisent pour expliquer le prix.

### MLR

comment : on constante qu'en ajoutant plusieurs paramètres alors le MSE est diminué de 46 à 24 quasiment de moitié
 ex de param : RM (Nombre de pièces) + prix
 ex de param : LSTAT (% de personnes à faible revenu) - prix
 ... Polution etc ...
 également le R² est doublé donc la variance est expliqué plus que avec la Simple Linear Regression