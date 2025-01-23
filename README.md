
# Projet IA : Navigation d’un robot dans un labyrinthe (MDP)

Ce projet d’intelligence artificielle explore les **processus décisionnels markoviens (MDP)** et implémente un **algorithme d’itération sur les valeurs** 
pour déterminer une politique optimale permettant à un robot de naviguer dans un labyrinthe.

## Objectifs
- Comprendre les concepts des MDP : <S, A, P, R, T>
- Implémenter et analyser l’impact des paramètres comme le facteur d’actualisation (γ) sur les décisions.
- Trouver une **politique optimale** et afficher le **chemin le plus court** dans un environnement avec obstacles (marécages).

## Fonctionnalités
- **Calcul de la politique optimale** basée sur les valeurs des états (π*).
- **Personnalisation des paramètres** :
  - Dimensions du labyrinthe.
  - Positions de départ et d’arrivée.
  - Récompenses attribuées aux marécages et à la destination.
  - Facteur d’actualisation (γ) et seuil de convergence (ε).
- **Visualisation graphique** :
  - Affichage des valeurs des états.
  - Représentation graphique du labyrinthe avec le chemin optimal.

## Organisation des fichiers
- **`algo.py`** : Implémentation de l’algorithme d’itération sur les valeurs et calcul des politiques optimales.
- **`main.py`** : Exemple de configuration avec un labyrinthe 5x5.
- **`main_lunch.py`** : Script principal avec des paramètres ajustables pour tester différents scénarios.

## Résultats affichés
1. Les valeurs des états.
2. La politique optimale.
3. Le chemin optimal.

## Usage
1. Cloner ce dépôt :
    ```bash
    git clone <URL_DU_DEPOT>
    ```
2. Lancer le script principal :
    ```bash
    python main_lunch.py
    ```
3. Modifier les paramètres dans `main_lunch.py` pour tester différents scénarios.

## Aperçu
![Illustration](https://via.placeholder.com/600x400.png?text=Exemple+de+visualisation)

## Auteurs
- **Thomas Jeanjacquot**
- **Amandine Lapique-Favre**

