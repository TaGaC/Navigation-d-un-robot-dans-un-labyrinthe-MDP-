import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Dimensions de la grille
n, m = 5, 5

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
def get_reward(state):
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

# Transition
def get_next_state(state, action):
    effect = action_effects[action]
    next_state = (state[0] + effect[0], state[1] + effect[1])
    if is_valid(next_state):
        return next_state
    return None  # Retourne None si le déplacement est hors limites




# Itération sur les valeurs
def value_iteration():
    global V
    V[goal[0], goal[1]] = reward_goal  # Initialise la valeur de l'état du but avec sa récompense
    while True:
        delta = 0
        new_V = np.copy(V)
        for i in range(n):
            for j in range(m):
                state = (i, j)
                if state == goal:
                    continue  # Continue d'ignorer la mise à jour du but durant les itérations
                max_value = float('-inf')
                for action in actions:
                    action_key = action[0]  # Prendre la première lettre de l'action
                    next_state = get_next_state(state, action_key)
                    if next_state is None:
                        continue  # Ignore les actions non valides

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

                    value = get_reward(state) + gamma * value
                    max_value = max(max_value, value)
                new_V[state] = max_value
                delta = max(delta, abs(new_V[state] - V[state]))
        V[:] = new_V
        if delta <= epsilon:
            break



# Calcul des Q-valeurs
def calculate_q_values(V):
    Q = np.zeros((n, m, len(actions)))
    for i in range(n):
        for j in range(m):
            state = (i, j)
            if state == goal:
                continue
            for k, action in enumerate(actions):
                action_key = action[0]  # Prendre la première lettre de l'action
                next_state = get_next_state(state, action_key)
                if next_state is None:
                    Q[i, j, k] = float('-inf')  # Attribuer une pénalité extrême pour sortir des limites
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

                Q[i, j, k] = get_reward(state) + gamma * value
    return Q


# Extraction de la politique optimale basée sur les Q-valeurs
def extract_policy(Q):
    policy = np.empty((n, m), dtype=str)
    for i in range(n):
        for j in range(m):
            if (i, j) == goal:
                policy[i, j] = 'F'
                continue
            best_action = np.argmax(Q[i, j])
            policy[i, j] = actions[best_action][0]
    return policy

# Extraire le chemin optimal à partir de la politique
def extract_path(policy):
    path = []
    state = start
    while state != goal:
        path.append(state)
        action = policy[state]
        next_state = get_next_state(state, action)
        if next_state is None or next_state == state or next_state in path:  # Vérifiez la validité et évitez les boucles
            break
        state = next_state
    path.append(goal)
    return path


# Fonction qui plot les valeurs des états et la politique optimale avec couleurs pour départ, arrivée, et chemin
def plot_values_and_policy(V, policy, path):
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

# Exécution de l'algorithme
value_iteration()
Q = calculate_q_values(V)
policy = extract_policy(Q)
path = extract_path(policy)

# Affichage des résultats
print("Valeurs des états:")
print(V)
print("\nQ-valeurs:")
print(Q)
print("\nPolitique optimale:")
print(policy)
print("\nChemin optimal:")
print(path)

# Afficher les résultats
plot_values_and_policy(V, policy, path)
