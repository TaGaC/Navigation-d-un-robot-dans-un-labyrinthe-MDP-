import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# Dimensions de la grille
n, m = 20,20

# Récompenses
reward_goal = 5
reward_marecage = -2
reward_default = 0

# Gamma et epsilon
epsilon = 0.01 # >0  un seuil de convergence dans l'itération de valeur. Il détermine à quel point la différence entre les valeurs estimées des états à travers les itérations doit être petite avant que l'algorithme puisse arrêter de s'exécuter.

# Actions possibles
actions = ['Haut', 'Bas', 'Gauche', 'Droite']
action_effects = {
    'H': (-1, 0),  # Haut
    'B': (1, 0),   # Bas
    'G': (0, -1),  # Gauche
    'D': (0, 1)    # Droite
}

# Initialisation des valeurs des états
V = np.zeros((n, m))

# Fonction de récompense
def get_reward(state, goal, marecages):
    if state == goal:
        return reward_goal
    elif state in marecages:
        return reward_marecage
    else:
        return reward_default

# Vérifie si l'état est valide
def is_valid(state):
    x, y = state
    return 0 <= x < n and 0 <= y < m

# Calcul de l'état suivant en fonction de l'action choisi et de l'état actuel, retourne None si l'action fait sortir de la grille, le noeud suivant sinon
def get_next_state(state, action):
    effect = action_effects[action]
    next_state = (state[0] + effect[0], state[1] + effect[1])
    if is_valid(next_state):
        return next_state
    return None  # Retourne None si le déplacement est hors limites




# Algorithme d'itération sur la Valeur (Vi), permet de déterminer les valeurs des états en fonction des récompenses et des valeurs des états voisins, on applique la formule de Bellman pour mettre à jour la valeur de l'état, on répète l'opération jusqu'à ce que sa se stabilise (V(n-1) = V(n)), si c'est pas parfait, le epsilon permet de déterminer la marge d'erreur pour arrêter l'itération
def value_iteration(V,goal,marecages,gamma):  # Voir diapo 31 du cours pour le détail de l'algorithme
    V[goal[0], goal[1]] = reward_goal  # Initialise la valeur de l'état du but avec sa récompense
    max_iterations = 100  # Limite à 100 itérations pour éviter les boucles infinies
    epsilon_threshold = epsilon * (1 - gamma) / (2 * gamma)  # Seuil de convergence basé sur la condition donnée
    i = 0
    while i < max_iterations:
        new_V = np.copy(V)  # Copie des valeurs pour éviter les mises à jour simultanées
        i += 1
        for i in range(n):
            for j in range(m):
                state = (i, j)
                if state == goal:
                    continue  # Continue d'ignorer la mise à jour du but durant les itérations
                max_value = float('-inf')
                for action in actions:
                    action_key = action[0]  # Prendre la première lettre de l'action comme clé
                    next_state = get_next_state(state, action_key)
                    if next_state is None:
                        continue  # Si next_state renvoie None, alors l'action sort des limites, on l'ignore

                    # Définir les états latéraux basés sur l'action principale
                    if action_key in ['H', 'B']:  # Mouvements verticaux
                        side_states = [
                            get_next_state(next_state, 'G'),  # Mouvement latéral gauche
                            get_next_state(next_state, 'D')   # Mouvement latéral droit
                        ]
                    else:  # 'G' ou 'D' - Mouvements horizontaux
                        side_states = [
                            get_next_state(next_state, 'H'),  # Mouvement vertical haut
                            get_next_state(next_state, 'B')   # Mouvement vertical bas
                        ]

                    value = 0.8 * V[next_state]  # Probabilité principale
                    for side_state in side_states:  # Ici on applique les 0,1 de probabilité pour les mouvements en cas de vent
                        if side_state is not None:
                            value += 0.1 * V[side_state]  # Probabilités latérales
                        else:
                            value += 0.1 * V[next_state]  # Si mouvement latéral invalide, ajouter valeur de l'état principal  on passerait donc à 0,9 pour la probabilité de l'action principale

                    value = get_reward(state,goal,marecages) + gamma * value  # On applique la formule de Bellman pour mettre à jour la valeur de l'état
                    max_value = max(max_value, value)
                new_V[state] = max_value

        # Vérifier la condition d'arrêt basée sur la norme
        if np.linalg.norm(new_V - V) < epsilon_threshold:
            V[:] = new_V
            break 
        V[:] = new_V
        
    if i == max_iterations-1:
        print("L'algorithme n'a pas convergé après 100 itérations, vérifiez que Epsilon n'est pas trop petit.")
    else:
        print(f"Convergence après {i} itérations.")
    return V

# Calcul des Q-valeurs, permet de déterminer la politique optimale, va chercher à maximiser la valeur de l'état suivant à partir des V-valeurs calculées
def calculate_q_values(V,goal,marecages,gamma):
    Q = np.zeros((n, m, len(actions)))
    for i in range(n):
        for j in range(m):
            state = (i, j)
            if state == goal:
                continue
            for k, action in enumerate(actions): # On a rajouté un index k pour les actions pour pouvoir ranger les Q-valeurs dans un tableau 3D
                action_key = action[0]  # Prendre la première lettre de l'action
                next_state = get_next_state(state, action_key)
                if next_state is None:
                    Q[i, j, k] = float('-inf')  # Attribuer une pénalité extrême pour ne pas aller hors limites
                    continue

                # Définir les états latéraux basés sur l'action principale
                if action_key in ['H', 'B']:  # Mouvements verticaux
                    side_states = [
                        get_next_state(next_state, 'G'),  # Mouvement latéral gauche
                        get_next_state(next_state, 'D')   # Mouvement latéral droit
                    ]
                else:  # 'G' ou 'D' - Mouvements horizontaux
                    side_states = [
                        get_next_state(next_state, 'H'),  # Mouvement vertical haut
                        get_next_state(next_state, 'B')   # Mouvement vertical bas
                    ]

                value = 0.8 * V[next_state]  # Probabilité principale
                for side_state in side_states:
                    if side_state is not None:
                        value += 0.1 * V[side_state]  # Probabilités latérales
                    else:
                        value += 0.1 * V[next_state]  # Si mouvement latéral invalide, ajouter valeur de l'état principal

                Q[i, j, k] = get_reward(state,goal,marecages) + gamma * value
    return Q


# Extraction de la politique optimale basée sur les Q-valeurs, on parcourt chaque état et on choisit l'action qui maximise la Q-valeur
def extract_policy(Q,goal):
    policy = np.empty((n, m), dtype=str)
    for i in range(n):
        for j in range(m):
            if (i, j) == goal:
                policy[i, j] = 'F'
                continue
            best_action = np.argmax(Q[i, j]) # Choisit parmis les actions celle qui maximise la Q-valeur
            policy[i, j] = actions[best_action][0] # On ne garde que la première lettre de la meilleur action pour l'écrire
    return policy

# Extraire le chemin optimal à partir de la politique, on part de l'état de départ et on suit les actions jusqu'à l'arrivée en suivant les actions de la politique
def extract_path(policy,start,goal):
    path = []
    state = start
    result = 1
    while state != goal:
        path.append(state)
        action = policy[state]
        next_state = get_next_state(state, action)
        if next_state is None or next_state == state or next_state in path:  # Vérifiez la validité et évitez les boucles, attention ici on a fait en sorte qu'il ne puisse plus faire revenir sur ses pieds, voir pour triater le problème ou il y a des égalités dans les Q-valeurs
            print("La politique n'est pas optimale, le chemin est bloqué.")
            result = 0
            break
        state = next_state
    path.append(goal)
    return path,result


# Fonction qui plot les valeurs des états et la politique optimale avec couleurs pour départ, arrivée, et chemin
def plot_values_and_policy(V, policy, path,start,goal,marecages):
    n, m = V.shape
    fig, ax = plt.subplots()
    
    # Créer une liste de couleurs pour la colormap
    cmap_list = ['white', 'darkgreen', 'blue', 'red', 'yellow']  # normal, marécage, départ, arrivée, chemin
    cmap = ListedColormap(cmap_list)
    
    # Préparer un tableau pour les indices des couleurs
    color_indices = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (i, j) == start:
                color_indices[i, j] = 2  # Départ en vert
            elif (i, j) == goal:
                color_indices[i, j] = 3  # Arrivée en rouge
            elif (i, j) in marecages:
                color_indices[i, j] = 1  # Marécages en vert foncé
            elif (i, j) in path:
                color_indices[i, j] = 4  # Chemin en jaune
            else:
                color_indices[i, j] = 0  # Autres cases en blanc

    # Afficher les cases avec la colormap personnalisée
    ax.imshow(color_indices, cmap=cmap, interpolation='nearest')

    # Ajouter les textes de valeur et politique
    for i in range(n):
        for j in range(m):
            text = ax.text(j, i, f'{V[i, j]:.2f}\n{policy[i, j]}',
                           ha='center', va='center', color='black' if (i, j) not in path else 'black')

    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(np.arange(m))
    ax.set_yticklabels(np.arange(n))
    ax.set_title("Valeurs des États et Politique Optimale")
    plt.show()


# Génération aléatoire de la grille
def generate_grid():
    start = (random.randint(0, n-1), random.randint(0, m-1))
    goal = (random.randint(0, n-1), random.randint(0, m-1))
    while start == goal:
        goal = (random.randint(0, n-1), random.randint(0, m-1))
    
    num_marecages = random.randint(1, (n*m)//(10/9))  # Limite le nombre de marécages à la moitié des cases
    marecages = set()
    while len(marecages) < num_marecages:
        marecage = (random.randint(0, n-1), random.randint(0, m-1))
        if marecage != start and marecage != goal:
            marecages.add(marecage)
    
    return start, goal, list(marecages)

# Initialisation et exécution de l'algorithme
def run_simulation():
    start, goal, marecages = generate_grid()
    V = np.zeros((n, m))  # Initialisation des valeurs des états
    gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Différents niveaux de gamma à tester

    for gamma in gamma_values:
        print(f"Testing with gamma = {gamma}")
        V = value_iteration(V, goal,marecages,gamma) 
        Q = calculate_q_values(V, goal,marecages,gamma) 
        policy = extract_policy(Q,goal)
        path,result = extract_path(policy, start, goal) #renvoie 1 si le chemin est valide, 0 sinon
        #plot_values_and_policy(V, policy, path, start, goal, marecages)
        if result != 0 :
            return gamma # On renvoie le premier gamma pour lequel le chemin est trouvé 
    return None

# Exécution de l'algorithme 50 fois et calcul du gamma
gamma_somme = 0
chemin_trouve = 0 
for i in range(50) :
    gamma_sim = run_simulation()
    if gamma_sim != None :
        gamma_somme = gamma_somme + gamma_sim
        chemin_trouve = chemin_trouve + 1
moyenne = gamma_somme/chemin_trouve
print(moyenne,chemin_trouve)

