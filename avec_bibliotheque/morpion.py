import pickle
import numpy as np
from math import inf
import random
from tqdm import tqdm



class Morpion:
    def __init__(self, markerJ1="X", markerJ2='O'):
        self.__grille = None
        self.__cases_vides = None
        self.__grille_joueurs = None
        self.__grille_points = None  # Vérifier rapidement un alignement : J1 = 1, J2 = -1
        self.__done = False
        self.current_player = 0
        self.__markers = [markerJ1, markerJ2]
        self.__case2point = {c: (c // 3, c % 3) for c in range(9)}
        self.__info = {}
        self.reset()

    def alignement(self, case):
        ligne, colonne = self.__case2point[case]
        if abs(sum(self.__grille_points[ligne, :])) == 3:
            return True
        if abs(sum(self.__grille_points[:, colonne])) == 3:
            return True
        if (ligne + colonne) % 2 == 0:
            if (ligne + colonne) % 4 == 0:
                return abs(sum([self.__grille_points[n, n] for n in [0, 1, 2]])) == 3
            elif ligne == 1:
                return abs(sum([self.__grille_points[n, n] for n in [0, 1, 2]])) == 3 or abs(sum([self.__grille_points[n, 2 - n] for n in [0, 1, 2]])) == 3
            else:
                return abs(sum([self.__grille_points[n, 2 - n] for n in [0, 1, 2]])) == 3
        return False

    def info(self):
        self.__info['win_actions'] = []
        self.__info['def_actions'] = []
        for case in self.__info['legal_actions']:
            ligne, colonne = self.__case2point[case]
            self.__grille_points[ligne, colonne] = 1 - 2 * self.current_player
            if self.alignement(case):
                self.__info['win_actions'].append(case)
            self.__grille_points[ligne, colonne] = 2 * self.current_player - 1
            if self.alignement(case):
                self.__info['def_actions'].append(case)
            self.__grille_points[ligne, colonne] = 0
        return self.__info

    def obs(self):
        return np.array([[self.__cases_vides, self.__grille_joueurs[0], self.__grille_joueurs[1]], [self.__cases_vides, self.__grille_joueurs[1], self.__grille_joueurs[0]]])

    def render(self):
        for ligne in self.__grille.reshape(3, 3):
            print(" ".join([c if c != None else "-" for c in ligne]))

    def reset(self):
        self.__grille = np.full(9, None)
        self.__cases_vides = np.full(9, 1)
        self.__grille_joueurs = np.full([2, 9], 0)
        self.__grille_points = np.full([3, 3], 0)
        self.__done = False
        self.__info['legal_actions'] = [i for i in range(9)]
        self.current_player = 0
        self.__info['current_player'] = self.current_player
        return self.obs()

    def step(self, case, verbose=False):
        reward = [0, 0]
        joueur_suivant = 1 - self.current_player
        if case in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if self.__grille[case] == None:
                self.__grille[case] = self.__markers[self.current_player]
                self.__cases_vides[case] = 0
                self.__grille_joueurs[self.current_player, case] = 1
                self.__grille_points[self.__case2point[case]] = 1 - 2 * self.current_player
                self.__info['legal_actions'].remove(case)
                if self.alignement(case):
                    self.__done = True
                    reward[self.current_player] = 1
                    reward[joueur_suivant] = -1
                    if verbose:
                        print("Alignement !")
            else:
                self.__done = True
                reward[self.current_player] = -10
                if verbose:
                    print("Case déjà jouée !")
        else:
            self.__done = True
            reward[self.current_player] = -10
        if len(self.__info['legal_actions']) == 0 and not self.__done:
            self.__done = True
            if verbose:
                print("Match nul !")
        self.current_player = joueur_suivant
        self.__info['current_player'] = self.current_player
        return self.obs(), reward, self.__done, self.info()


class Joueur:
    def __init__(self, nb_states, nb_actions, file=None):
        self.__Q = np.zeros([nb_states, nb_actions])
        self.__nb_actions = nb_actions
        self.__file = file
        if file:
            self.load()

    def jouer(self, state, allowed_actions, coeff_random=0):
        Q2 = (self.__Q[state, :] + np.random.randn(1, self.__nb_actions) * coeff_random)[0]
        return np.argmax([Q2[a] if a in allowed_actions else -inf for a in range(self.__nb_actions)])

    def load(self):
        try:
            with open(self.__file, 'rb') as f:
                self.__Q = pickle.load(f)
                print("IA Chargée :", f)
        except:
            print("Chargement impossible")

    def recompenser(self, state, action, reward, new_state, learning_rate=0.85, actualisation_factor=0.99):
        self.__Q[state, action] = self.__Q[state, action] + learning_rate * (
            reward + actualisation_factor * np.max(self.__Q[new_state, :]) - self.__Q[state, action])


    def entrainer(self, data, reward, learning_rate=0.85, actualisation_factor=0.99):
        for state, action, new_state in data[::-1]:
            self.recompenser(state, action, reward, new_state, learning_rate=learning_rate, actualisation_factor=actualisation_factor)

    def obs2state(self, obs):
        array = np.array(obs)
        state = sum([n * 3**i for i, n in enumerate((array[1] + 2 * array[2]).reshape(1, 9)[0])])
        return state

    def save(self, file=None):
        self.__file = file if file else self.__file
        with open(self.__file, 'wb') as f:
            pickle.dump(self.__Q, f)

    def Q_table(self):
        return np.array(self.__Q)
    


class Joueur_Aleatoire:
    def __init__(self):
        pass

    def __repr__(self):
        return "Joueur Aléatoire"

    def jouer(self, info):
        return random.choice(info['legal_actions'])

class Joueur_Simple:
    def __init__(self):
        pass

    def __repr__(self):
        return "Joueur Simple"

    def jouer(self, info):
        if len(info['win_actions']) > 0:
            return info['win_actions'][0]
        elif len(info['def_actions']) > 0:
            return info['def_actions'][0]
        else:
            return random.choice(info['legal_actions'])

class Joueur_Avance:
    def __init__(self, random_factor=0):
        self.__random_factor = random_factor

    def __repr__(self):
        return f"Joueur Avancé niv.{round((1 - self.__random_factor) * 100, 2)}"

    def jouer(self, info):
        if len(info['legal_actions']) == 9 and random.random() > self.__random_factor:
            return random.choice([0, 2, 6, 8])
        elif len(info['legal_actions']) == 8 and 4 in info['legal_actions'] and random.random() > self.__random_factor:
            return 4
        elif len(info['win_actions']) > 0 and random.random() > self.__random_factor:
            return info['win_actions'][0]
        elif len(info['def_actions']) > 0 and random.random() > self.__random_factor:
            return info['def_actions'][0]
        else:
            return random.choice(info['legal_actions'])

    def set_random_factor(self, random_factor):
        self.__random_factor = random_factor

class Joueur_Intermediaire:
    def __init__(self):
        pass

    def __repr__(self):
        return "Joueur Intermediaire"

    def jouer(self, info):
        if len(info['legal_actions']) == 9:
            return 4
        elif len(info['win_actions']) > 0:
            return info['win_actions'][0]
        elif len(info['def_actions']) > 0:
            return info['def_actions'][0]
        else:
            return random.choice(info['legal_actions'])



# Définir le nombre d'épisodes d'entraînement
episodes_entrainement = [100, 500, 1000, 5000, 10000]

# Définir les adversaires pour l'entraînement
liste_adversaires_entrainement = [
    Joueur_Aleatoire(),
    Joueur_Simple(),
    Joueur_Intermediaire()
]

# Définir les adversaires pour les tests
liste_adversaires_tests = [
    Joueur_Aleatoire(),
    Joueur_Simple(),
    Joueur_Intermediaire()
]

# Entraînement et évaluation
for ep in episodes_entrainement:
    print(f"Entraînement avec {ep} épisodes...")

    # Créer une nouvelle instance de l'IA
    NOM_IA = f"IA_QL_{ep}"
    joueur = Joueur(nb_states=3**9, nb_actions=9, file=NOM_IA)

    # Entraînement de l'IA
    for _ in tqdm(range(ep)):
        for adversaire in liste_adversaires_entrainement:
            env = Morpion()
            obs = env.reset()
            done = False
            data = []

            while not done:
                if env.current_player == 0:
                    state = joueur.obs2state(obs[0])
                    action = joueur.jouer(state, env.info()['legal_actions'])
                    new_obs, reward, done, info = env.step(action)
                    new_state = joueur.obs2state(new_obs[0])
                    data.append((state, action, new_state))
                    obs = new_obs
                else:
                    action = adversaire.jouer(env.info())
                    obs, reward, done, info = env.step(action)

            joueur.entrainer(data, reward[0])

    joueur.save(file=NOM_IA)

    # Évaluation de l'IA
    scores = {adversaire.__repr__(): 0 for adversaire in liste_adversaires_tests}
    for adversaire in liste_adversaires_tests:
        for _ in range(100):
            env = Morpion()
            obs = env.reset()
            done = False
            while not done:
                if env.current_player == 0:
                    state = joueur.obs2state(obs[0])
                    action = joueur.jouer(state, env.info()['legal_actions'])
                    obs, reward, done, info = env.step(action)
                else:
                    action = adversaire.jouer(env.info())
                    obs, reward, done, info = env.step(action)

            if reward[0] > 0:
                scores[adversaire.__repr__()] += 1
            elif reward[0] == 0:
                scores[adversaire.__repr__()] += 0.5

    print(f"Scores après {ep} épisodes : {scores}")

# Script pour jouer contre l'IA entraînée
def jouer_contre_ia(nom_ia):
    env = Morpion()
    joueur = Joueur(nb_states=3**9, nb_actions=9, file=nom_ia)

    obs = env.reset()
    env.current_player = random.choice([0, 1])
    info = env.info()
    done = False

    while not done:
        if env.current_player == 0:
            state = joueur.obs2state(obs[0])
            case = joueur.jouer(state, info['legal_actions'])
        else:
            case = -1
            while case not in info['legal_actions']:
                case = int(input('Coup ? (Entre 0 et 8)'))

        obs, reward, done, info = env.step(case)
        env.render()
        print("")

    if reward[0] > 0:
        print("Victoire de l'IA !")
    elif reward[0] < 0:
        print("Défaite de l'IA !")
    else:
        print("Match nul...")

# Pour jouer contre l'IA, décommentez et remplacez 'IA_QL' par le nom du fichier de l'IA entraînée

jouer_contre_ia('IA_QL_10000')