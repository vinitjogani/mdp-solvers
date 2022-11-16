from functools import cached_property, lru_cache
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

    def actions(self, state):
        raise NotImplementedError()

    def transitions(self, state, action):
        raise NotImplementedError()

    def reward(self, state):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def simulate(self, policy, n):
        rewards = []
        for _ in range(n):
            self.reset()

            R = 0
            while True:
                actions = self.actions(self.state)
                if not actions:
                    break
                action = get_action(policy, self.state)
                reward, done = self.step(action)
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

    def __init__(self, n=6, gamma=0.99, policy="chopsticks.policy"):
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

    @lru_cache()
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

    def transitions(self, state, action):
        """
        Return (s', p) tuples containing the next states and probabilities
        for taking <action> in <state>.
        """
        if self.done(action):
            return [(action, 1)]

        # Possible action from player 2's perspective
        options = self.opponent[rev(action)]
        return [(rev(s), p) for s, p in options.items()]

    def reward(self, state):
        i, j, k, l = state
        if (i + j) == 0:
            return -1
        if (k + l) == 0:
            return 1
        return 0.0

    def step(self, action):
        options = self.transitions(None, action)
        self.state = random.choice(options)[0]
        return self.reward(self.state), self.done(self.state)


class Racecar(Environment):
    def __init__(self):
        self.max_speed = 3
        self.max_fuel = 3
        self.steps = 0
        self.gamma = 0.9

    def reset(self):
        self.steps = 0
        self.state = (0, self.max_fuel, self.max_speed)

    def pit_stop(self, temp, fuel, speed):
        if fuel == 0:
            return 0.95
        elif temp >= 2:
            return temp / 10
        elif temp >= 1:
            return temp / 20
        elif temp >= 0:
            return temp / 30
        return 0

    @cached_property
    def states(self):
        output = []
        for temp in np.arange(0, 2.1, 0.5):
            for fuel in np.arange(0, self.max_fuel + 0.1, 0.5):
                for speed in range(self.max_speed + 1):
                    output.append((temp, fuel, speed))
        return output

    def actions(self, state):
        return (-1, 0, 1)

    def transitions(self, state, action):
        temp, fuel, speed = state
        output = []

        p_heat = 0.5

        if action == -1:
            speed = max(speed - 1, 0)
            if speed <= 1:
                fuel = min(fuel + 1, self.max_fuel)

            temp_ = max(temp - 1, 0)
            p_blowout = self.pit_stop(temp_, fuel, speed)
            output.append(((temp_, fuel, speed), p_heat * (1 - p_blowout)))
            output.append(((0, self.max_fuel, 0), p_heat * p_blowout))

            temp_ = temp
            p_blowout = self.pit_stop(temp_, fuel, speed)
            output.append(((temp_, fuel, speed), (1 - p_heat) * (1 - p_blowout)))
            output.append(((0, self.max_fuel, 0), (1 - p_heat) * p_blowout))

        elif action == 1 and fuel >= 1:
            fuel = max(fuel - 1, 0)
            speed = min(speed + 1, self.max_speed)

            temp_ = min(temp + 1, 2)
            p_blowout = self.pit_stop(temp_, fuel, speed)
            output.append(((temp_, fuel, speed), p_heat * (1 - p_blowout)))
            output.append(((0, self.max_fuel, 0), p_heat * p_blowout))

            temp_ = temp
            p_blowout = self.pit_stop(temp_, fuel, speed)
            output.append(((temp_, fuel, speed), (1 - p_heat) * (1 - p_blowout)))
            output.append(((0, self.max_fuel, 0), (1 - p_heat) * p_blowout))

        else:
            fuel = max(fuel - 0.5, 0)
            if speed <= 1:
                fuel = min(fuel + 0.5, self.max_fuel)

            temp_ = min(temp + 0.5, 2)
            p_blowout = self.pit_stop(temp_, fuel, speed)
            output.append(((temp_, fuel, speed), p_heat * (1 - p_blowout)))
            output.append(((0, self.max_fuel, 0), p_heat * p_blowout))

            temp_ = temp
            p_blowout = self.pit_stop(temp_, fuel, speed)
            output.append(((temp_, fuel, speed), (1 - p_heat) * (1 - p_blowout)))
            output.append(((0, self.max_fuel, 0), (1 - p_heat) * p_blowout))

        return [s for s in output if s[1] > 0]

    def reward(self, state):
        return state[-1]

    def step(self, action):
        self.steps += 1
        options = np.array(self.transitions(self.state, action), dtype=object)
        self.state = np.random.choice(options[:, 0], p=options[:, 1].astype("float64"))
        reward = self.reward(self.state)
        done = self.steps > 2000
        return reward, done
