import numpy as np
import matplotlib.pyplot as plt

# Dimensions de la grille
n, m = 5, 5

# Positions de départ et d'arrivée
start = (4, 1)
goal = (0, 3)

# Positions des marécages
marecages = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]

# Récompenses
reward_goal = 5
reward_marecage = -1
reward_default = -0.1

# Gamma et epsilon
gamma = 0.9
epsilon = 0.01

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
    return state

# Itération sur les valeurs
def value_iteration():
    while True:
        delta = 0
        new_V = np.copy(V)
        for i in range(n):
            for j in range(m):
                state = (i, j)
                if state == goal:
                    continue
                max_value = float('-inf')
                for action in actions:
                    action_key = action[0]  # Prendre la première lettre de l'action
                    next_state = get_next_state(state, action_key)
                    transition_prob = 0.8
                    side_states = [
                        get_next_state(state, 'G') if action in ['Haut', 'Bas'] else get_next_state(state, 'H'),
                        get_next_state(state, 'D') if action in ['Haut', 'Bas'] else get_next_state(state, 'B')
                    ]
                    value = transition_prob * V[next_state]
                    for side_state in side_states:
                        value += 0.1 * V[side_state]
                    max_value = max(max_value, get_reward(state) + gamma * value)
                new_V[state] = max_value
                delta = max(delta, abs(new_V[state] - V[state]))
        V[:] = new_V
        if delta < epsilon:
            break

# Calcul de la politique optimale
def extract_policy():
    policy = np.empty((n, m), dtype=str)
    for i in range(n):
        for j in range(m):
            state = (i, j)
            if state == goal:
                policy[state] = 'F'
                continue
            max_value = float('-inf')
            best_action = None
            for action in actions:
                action_key = action[0]  # Prendre la première lettre de l'action
                next_state = get_next_state(state, action_key)
                transition_prob = 0.8
                side_states = [
                    get_next_state(state, 'G') if action in ['Haut', 'Bas'] else get_next_state(state, 'H'),
                    get_next_state(state, 'D') if action in ['Haut', 'Bas'] else get_next_state(state, 'B')
                ]
                value = transition_prob * V[next_state]
                for side_state in side_states:
                    value += 0.1 * V[side_state]
                value = get_reward(state) + gamma * value
                if value > max_value:
                    max_value = value
                    best_action = action_key
            policy[state] = best_action
    return policy

# Extraire le chemin optimal à partir de la politique
def extract_path(policy):
    path = []
    state = start
    while state != goal:
        path.append(state)
        action = policy[state]
        if action == 'F':  # État d'arrivée atteint
            break
        state = get_next_state(state, action)
        if state in path:  # En cas de boucle infinie
            break
    path.append(goal)
    return path

# Fonction qui plot les valeurs des états et la politique optimale avec couleurs pour départ, arrivée, et chemin
def plot_values_and_policy(V, policy, path):
    n, m = V.shape
    fig, ax = plt.subplots()
    im = ax.imshow(V, cmap='coolwarm')

    for i in range(n):
        for j in range(m):
            if (i, j) == start:
                color = 'blue'
            elif (i, j) == goal:
                color = 'green'
            elif (i, j) in path:
                color = 'yellow'
            else:
                color = 'black'
            text = ax.text(j, i, f'{V[i, j]:.2f}\n{policy[i, j]}',
                           ha='center', va='center', color=color)

    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(np.arange(m))
    ax.set_yticklabels(np.arange(n))
    ax.set_title("Valeurs des États et Politique Optimale")
    fig.tight_layout()
    plt.show()

# Exécution de l'algorithme
value_iteration()
policy = extract_policy()
path = extract_path(policy)

# Affichage des résultats
print("Valeurs des états:")
print(V)
print("\nPolitique optimale:")
print(policy)
print("\nChemin optimal:")
print(path)

# Afficher les résultats
plot_values_and_policy(V, policy, path)
