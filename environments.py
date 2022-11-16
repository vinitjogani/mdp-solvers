from functools import cached_property
import random
import numpy as np
import pickle


def get_action(policy, state):
    a = policy[state]
    if isinstance(a, dict):
        actions, probs = zip(*a.items())
        out = np.empty(len(actions), dtype=object)
        out[:] = actions
        a = np.random.choice(out, p=probs)
    return a


class Environment:
    def reset(self):
        raise NotImplementedError()

    # def actions(self, state):
    #     raise NotImplementedError()

    def transitions(self, state, action, V):
        raise NotImplementedError()

    def reward(self, state):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def simulate(self, V, n):
        rewards = []
        for _ in range(n):
            self.reset()

            R = 0
            while True:
                actions = self.actions(self.state)
                if not actions:
                    break
                best = np.argmax(
                    [
                        sum(p * V[s] for s, p in self.transitions(self.state, a, None))
                        for a in actions
                    ]
                )
                reward, done = self.step(actions[best], None)
                R += reward
                if done:
                    break
            rewards.append(R)
        return np.mean(rewards)

    def simulate_policy(self, policy, n):
        rewards = []
        for _ in range(n):
            self.reset()

            R = 0
            while True:
                actions = self.actions(self.state)
                if not actions:
                    break
                action = get_action(policy, self.state)
                reward, done = self.step(action, None)
                R += reward
                if done:
                    break
            rewards.append(R)
        return np.mean(rewards)


def sanitize(i, j, k, l):
    return min(i, j), max(i, j), min(k, l), max(k, l)


def rev(state):
    k, l, i, j = state
    return i, j, k, l


class Chopsticks(Environment):
    """
    Chopsticks is a 2-player game where each person starts with
    1 raised finger (aka point) on both hands and player alternate taking turns
    until one player has no points left on either hand.

    At each turn, a player may either:
    (1) Re-allocate their points between their two hands, e.g. a player
        with 4 points on one hand and 1 points on another may distribute
        them to make it 2 points on one hand and 3 on the other.
    (2) Hit one of the opponent's hands to add on points e.g. when a player
        has 1 point in a hand and uses it to hit an opponent's hand with 3
        points, then that hand will now have 4 points.

    Typically, the maximum number of points on each hand is restricted to 4
    beyond which the points roll back from 0 (i.e. the points are always modulo 5).
    However, we can generalize this to an n-point game.
    """

    def __init__(self, n=6, gamma=0.99, policy="chopsticks_opponent.pkl"):
        self.n = n
        self.gamma = gamma
        self.T = {}
        self.opponent = pickle.load(open(policy, "rb"))

    def reset(self):
        self.state = (1, 1, 1, 1)

    def done(self, state):
        i, j, k, l = state
        if (i == 0 and j == 0) or (k == 0 and l == 0):
            return True
        return False

    @cached_property
    def states(self):
        """
        Each state is represented as a 4-tuple (a, b, c, d)
        where (a, b) are sorted list of points for player 1 and
        (c, d) are sorted list of points for player 2.
        """
        S = []
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        if i > j or k > l:
                            continue
                        S.append((i, j, k, l))
        return S

    # @cached_property
    def actions(self, state):
        """
        The possible actions that player 1 can take in this state.
        """
        if state in self.T:
            return self.T[state]

        if self.done(state):
            return []

        i, j, k, l = state
        t = []
        if j > 0:
            for i2 in range(i + 1, min(i + j + 1, self.n)):
                t.append((i2, j + i - i2, k, l))
            if k > 0:
                t.append((i, j, (k + j) % self.n, l))
            if l > 0:
                t.append((i, j, k, (l + j) % self.n))
        if i > 0:
            for j2 in range(j + 1, min(i + j + 1, self.n)):
                t.append((i + j - j2, j2, k, l))
            if k > 0:
                t.append((i, j, (k + i) % self.n, l))
            if l > 0:
                t.append((i, j, k, (l + i) % self.n))

        sanitized = {sanitize(*x) for x in t}
        self.T[state] = [x for x in sanitized if x != (i, j, k, l)]
        return self.T[state]

    def transitions(self, state, action, V):
        """
        Return (s', p) tuples containing the next states and probabilities
        for taking <action> in <state>.
        """
        if self.done(action):
            return [(action, 1)]

        # Possible action from player 2's perspective
        options = self.opponent[rev(action)]

        # if V is None:
        return [(rev(s), p) for s, p in options.items()]
        # else:
        #     top_options = np.argsort([V[rev(s)] for s in options])[:10]
        #     return [(rev(options[o]), 1 / len(top_options)) for o in top_options]

    def reward(self, state):
        i, j, k, l = state
        if i + j == 0:
            return -1
        if k + l == 0:
            return 1
        return 0

    def step(self, action, V):
        options = self.transitions(None, action, V)
        self.state = random.choice(options)[0]
        return self.reward(self.state), self.done(self.state)
