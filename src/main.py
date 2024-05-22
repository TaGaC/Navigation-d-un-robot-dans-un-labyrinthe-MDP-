import numpy as np
import matplotlib.pyplot as plt

# Dimensions de la grille
n, m = 5, 5

# Positions de départ et d'arrivée
start = (4, 1)
goal = (0, 3)

# Positions des marécages
marecages = [(1, 1), (1, 2), (1,3), (2,3), (3,3)]

# Récompenses
reward_goal = 5
reward_marecage = -1
reward_default = -0.1

# Gamma et epsilon
gamma = 0,9
epsilon = 0.01

# Actions possibles, on part du principe que notre case 0,0 est en haut à gauche
actions = ['Haut', 'Bas', 'Gauche', 'Droite']
action_effects = {
    'Haut': (-1, 0),
    'Bas': (1, 0),
    'Gauche': (0, -1),
    'Droite': (0, 1)
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
                    next_state = get_next_state(state, action)
                    transition_prob = 0.8
                    side_states = [
                        get_next_state(state, 'Gauche') if action in ['Haut', 'Bas'] else get_next_state(state, 'Haut'),
                        get_next_state(state, 'Droite') if action in ['Haut', 'Bas'] else get_next_state(state, 'Bas')
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
                policy[state] = 'G'
                continue
            max_value = float('-inf')
            best_action = None
            for action in actions:
                next_state = get_next_state(state, action)
                transition_prob = 0.8
                side_states = [
                    get_next_state(state, 'Gauche') if action in ['Haut', 'Bas'] else get_next_state(state, 'Haut'),
                    get_next_state(state, 'Droite') if action in ['Haut', 'Bas'] else get_next_state(state, 'Bas')
                ]
                value = transition_prob * V[next_state]
                for side_state in side_states:
                    value += 0.1 * V[side_state]
                if value > max_value:
                    max_value = value
                    best_action = action
            policy[state] = best_action
    return policy


# Fonction qui plot les valeurs des états et la politique optimale
def plot_values_and_policy(V, policy):
    n, m = V.shape
    fig, ax = plt.subplots()
    im = ax.imshow(V, cmap='coolwarm')

    for i in range(n):
        for j in range(m):
            text = ax.text(j, i, f'{V[i, j]:.2f}\n{policy[i, j]}',
                           ha='center', va='center', color='black')

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

# Affichage des résultats
print("Valeurs des états:")
print(V)
print("\nPolitique optimale:")
print(policy)

# Afficher les résultats
plot_values_and_policy(V, policy)
