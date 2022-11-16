from environments import *
import numpy as np
import random
import time


def derive_policy(env, V):
    policy = {}
    for s in env.states:
        best_a, best_v = None, None
        for a in env.actions(s):
            v = sum(V[s_] * p for s_, p in env.transitions(s, a))
            if best_v is None or v > best_v:
                best_a, best_v = a, v
        policy[s] = {best_a: 1.0}
    return policy


def compare_policies(p1, p2):
    delta = 0
    for k in p1:
        if p2[k] != p1[k]:
            delta += 1
    return delta


def epsilon_greedy(policy, env, eps=0.001):
    policy = dict(policy)
    for s in policy:
        all_actions = env.actions(s)
        actions = dict(policy[s])
        if len(actions) == len(all_actions):
            continue

        residual = eps / (len(all_actions) - len(actions))
        normalizer = sum(actions.values()) + eps
        for a in all_actions:
            if a not in actions:
                actions[a] = residual / normalizer
            else:
                actions[a] /= normalizer

        policy[s] = actions
    return policy


def value_iteration(env, max_iters=500):

    V = {}
    for s in env.states:
        V[s] = 0
    policy = derive_policy(env, V)
    change_history = []
    reward_history = []
    time_history = []
    start = time.time()

    delta = 1
    while delta > 1e-3 and max_iters > 0:
        max_iters -= 1
        reward_history.append(env.simulate(derive_policy(env, V), 100))
        delta = 0

        V_ = dict(V)
        for s in env.states:
            V_[s] = env.reward(s)
            if env.actions(s):
                V_[s] += env.gamma * max(
                    sum(V[s_] * p for s_, p in env.transitions(s, a))
                    for a in env.actions(s)
                )
            delta += abs(V_[s] - V[s])

        V = V_

        policy_ = policy
        policy = derive_policy(env, V)
        changes = compare_policies(policy_, policy)
        change_history.append(changes)
        time_history.append(time.time() - start)
        if changes == 0:
            print("Converged!")
            break

    return V, policy, change_history, reward_history, time_history


def policy_evaluation(policy, env):
    V = {}
    for s in env.states:
        V[s] = 0

    delta = 1
    while delta > 1e-1:
        delta = 0

        V_ = dict(V)
        for s in env.states:
            V_[s] = env.reward(s)
            if env.actions(s):
                V_[s] += env.gamma * sum(
                    V[s_] * p * p_
                    for a, p_ in policy[s].items()
                    for s_, p in env.transitions(s, a)
                )

            delta += abs(V_[s] - V[s])

        V = V_
    return V


def policy_iteration(env, iters=15):
    V = {}
    policy = {}
    for s in env.states:
        V[s] = 0
        actions = env.actions(s)
        policy[s] = {a: 1 / len(actions) for a in actions}
    change_history = []
    reward_history = []
    time_history = []
    start = time.time()

    for _ in range(iters):
        reward_history.append(env.simulate(policy, 100))
        V = policy_evaluation(policy, env)

        policy_ = policy
        policy = derive_policy(env, V)
        changes = compare_policies(policy_, policy)
        change_history.append(changes)
        time_history.append(time.time() - start)
        if changes == 0:
            print("Converged!")
            break

    return V, policy, change_history, reward_history, time_history


if __name__ == "__main__":
    # print("Value Iteration:")
    # np.random.seed(42)
    # random.seed(42)
    # c = Chopsticks(12, 0.9)
    # value_iteration(c)

    # print("Policy Iteration")
    # np.random.seed(42)
    # random.seed(42)
    # c = Chopsticks(12, 0.9)
    # policy_iteration(c)

    print("Value Iteration:")
    np.random.seed(42)
    random.seed(42)
    c = Racecar()
    value_iteration(c)

    print("Policy Iteration")
    np.random.seed(42)
    random.seed(42)
    c = Racecar()
    policy_iteration(c)
