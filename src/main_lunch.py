from algo import lunch_algo
def lunch():
    # Dimensions de la grille
    n, m = 10, 5

    # Positions de départ et d'arrivée
    start = (4, 1)
    goal = (0, 3)

    # Positions des marécages
    marecages = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]
    # Récompenses
    reward_goal = 5
    reward_marecage = -2
    reward_default =0

    # Gamma et epsilon
    gamma = 0.5 # Un γ proche de 1 fait en sorte que l'agent valorise fortement les récompenses futures, ce qui le rend plus stratégique avec une planification à long terme. Un γ proche de 0 rend l'agent myope aux récompenses futures, se concentrant presque exclusivement sur les récompenses immédiates.
    epsilon = 0.1 #  un seuil de convergence dans l'itération de valeur. Il détermine à quel point la différence entre les valeurs estimées des états à travers les itérations doit être petite avant que l'algorithme puisse arrêter de s'exécuter.

    lunch_algo(n, m, start, goal, marecages, reward_goal, reward_marecage, gamma, epsilon)
    
    # Attendre que l'utilisateur appuie sur une touche avant de terminer le programme
    input("Appuyez sur une touche pour quitter...")

if __name__ == "__main__":
    lunch()
