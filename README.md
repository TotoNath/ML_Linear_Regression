# Rapport de réponses TP : Régression Linéaire

### Rapport de : GUIQUERRO Nathaniel

### fait le : 14/03/2025

## Partie 1 :

Je setup mon environment sur Intellij IDEA Ultimate 2025
> venv
> en utilisant les
> \n imports suivant\
> import pandas as pd\
> from sklearn.linear_model import LinearRegression\
> from sklearn.metrics import mean_squared_error, r2_score\
> from sklearn.model_selection import train_test_split\
> from ydata_profiling import ProfileReport\
> import seaborn as sns\
> import matplotlib.pyplot as plt\

## Partie 2 :

on importe via le csv sur myges car le dataset n'est plus présent
dans les nouvelles version à cause d'un problème d'éthique apparement

## Partie 3 :

On explore le fichier HTML crée je constate qu'il y a plusieurs parramètres qui sont analysé

- RM : Nombre de pièce  (aspect probablement positif sur le prix) et suit une loi normal
- LSTAT : % de personne a faible revenu (influence probablement negatif)
- Crim : dans la plus part de quartiers il y a aucun crim mais il y a peu de ville où il y a la plus part des crimes
- NOX : polution (négatif) n'est pas normal
- DIS : distance par rapport a l'emploi (peut être positf mais négatif aussi d'après moi) la plus part sort a 2.5 miles
  de leurs travails, il y a très peu d'habitations à + 12 miles du centre de travail
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

### Correction =>

Corelation positive : si la première variable augemente (resp.diminue), la 2ème variable as une tendance d'augementer (
resp.diminiuer).
Corrélation négative : si la première variable augemente, la deuxième a une tendance a diminuer.

||ro|| <= 1

- si ro € [-0.3;0.3] => faible corrélation
- si ro = 0 => pas de corrélation
- si ro € [-0.3;-0.75] | [0.3;0.75] => goot to very good corrélation
- si ro € ~= 1 => excellent corrélation

donc fallait utiliser RM, lstat.

## Partie 5:

### SLR

comment : Le modèle fait bcp d'erreur car MSE : 46 => un seul paramètre ne suffit pas pour le MEDV
le nombre de pièce n'est pas suffisent pour expliquer le prix.

### Correction =>

R² => variance expliqué => plus c'est proche de 1 mieux c'est si c'est en dessous de 0 le modèle n'est pas bon.
dans le code .fit => descente du gradient
R² € [0;1] on peut expliquer R²*100 % des résultats prédits
Variance expliqué dans la variable par le modèle(ex : medv ici)
Ici 37% des variations sont expliqués par le modèle ce qui n'est pas suffisant

### MLR

comment : on constante qu'en ajoutant plusieurs paramètres alors le MSE est diminué de 46 à 24 quasiment de moitié
ex de param : RM (Nombre de pièces) + prix
ex de param : LSTAT (% de personnes à faible revenu) - prix
... Polution etc ...

également le R² € [0;1] on peut expliquer R²*100 % des résultats prédits
Variance expliqué dans la variable par le modèle(ex : medv ici)
Ici 67% des variations sont expliqués par le modèle