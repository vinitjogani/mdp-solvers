from environments import Chopsticks, get_action
import numpy as np
import random
import pickle


def value_iteration(env, iters=10):

    V = {}
    for s in env.states:
        V[s] = 0

    for i in range(iters):
        print(i, env.simulate(V, 1000))

        V_ = dict(V)
        for s in env.states:
            V_[s] = env.reward(s)
            if not env.actions(s):
                continue
            V_[s] += env.gamma * max(
                sum(V[s_] * p for s_, p in env.transitions(s, a, None))
                for a in env.actions(s)
            )

        V = V_


def policy_evaluation(policy, env):
    V = {}
    for s in env.states:
        V[s] = 0

    delta = 1
    while delta > 0:
        delta = 0

        V_ = dict(V)
        for s in env.states:
            V_[s] = env.reward(s)
            if env.actions(s):
                V_[s] += env.gamma * sum(
                    V[s_] * p * p_
                    for a, p_ in policy[s].items()
                    for s_, p in env.transitions(s, a, None)
                )

            delta += abs(V_[s] - V[s])

        V = V_
    # print("\n\n\n\n")
    return V


def policy_iteration(env, iters=10):

    V = {}
    policy = {}
    for s in env.states:
        V[s] = 0
        actions = env.actions(s)
        policy[s] = {a: 1 / len(actions) for a in actions}

    for i in range(iters):
        print(i, env.simulate_policy(policy, 1000))

        V = policy_evaluation(policy, env)
        for s in env.states:
            best_a, best_v = None, None
            for a in env.actions(s):
                v = sum(V[s_] * p for s_, p in env.transitions(s, a, None))
                if best_v is None or v > best_v:
                    best_a, best_v = a, v
            policy[s] = {best_a: 1.0}


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    c = Chopsticks(12, 0.9)
    value_iteration(c)

    np.random.seed(0)
    random.seed(0)
    c = Chopsticks(12, 0.9, "chopsticks_opponent.pkl")
    policy_iteration(c)
