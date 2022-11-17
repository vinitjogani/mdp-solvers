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

        if not all_actions: continue

        residual = eps / len(all_actions)
        for a in all_actions:
            if a not in actions:
                actions[a] = eps / len(all_actions)
            else:
                actions[a] = (1 - eps) * actions[a] + residual

        policy[s] = actions
    return policy


def value_iteration(env, max_iters=500, rsims=100):

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
        reward_history.append(env.simulate(derive_policy(env, V), rsims))
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
        time_history.append(time.time() - start)

        changes = compare_policies(policy_, policy)
        change_history.append(changes)
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


def policy_iteration(env, iters=15, rsims=100):
    V = {}
    policy = {}
    for s in env.states:
        V[s] = 0
        actions = env.actions(s)
        policy[s] = {a: 1 / len(actions) for a in actions}
    change_history = []
    reward_history = []
    time_history = []
    timer = 0

    for _ in range(iters):
        reward_history.append(env.simulate(policy, rsims))

        start = time.time()
        V = policy_evaluation(policy, env)
        policy_ = policy
        policy = derive_policy(env, V)
        timer += time.time() - start
        time_history.append(timer)

        changes = compare_policies(policy_, policy)
        change_history.append(changes)
        if changes == 0:
            print("Converged!")
            break

    return V, policy, change_history, reward_history, time_history


def generate_episode(policy, env, max_iters=2000):
    env.reset()
    done = False 
    episode = []

    while not done and max_iters > 0:
        max_iters -= 1
        state = env.state 
        action = get_action(policy, state)
        reward, done = env.step(action)
        episode.append((state, action, reward, env.state, done))

    return episode



def q_learning(env, iters=100, eps=0.2, eps_decay=0.95, min_eps=0.001, lr=0.1, replay_size=100_000, rsims=100):
    Q = {}
    policy = {}
    for s in env.states:
        actions = env.actions(s)
        Q[s] = {a: 0 for a in actions}
        policy[s] = {a: 1 / len(actions) for a in actions}
    change_history, reward_history, time_history = [], [], []
    timer = 0
    unchanged = 0

    replay_memory = []
    for _ in range(iters):
        reward_history.append(env.simulate(policy, rsims))
        print(reward_history[-1])
        
        start = time.time()

        Q_ = dict(Q)
        policy_ = policy
        policy = dict(policy_)
        for s in Q:
            if not Q[s]: continue
            best = max((v, a) for a, v in Q[s].items())[1]
            policy[s] = {best:1.0}

        episode = generate_episode(epsilon_greedy(policy, env, eps), env, 50)
        replay_memory.extend(episode)
        replay_memory = replay_memory[-replay_size:]
        for s, a, r, s_, d in replay_memory: 
            if d:
                new = r 
            else:
                new = r + env.gamma * max(Q[s_].values())
            old = Q_[s][a]
            Q_[s][a] += lr * (new - old)
        Q = Q_

        timer += time.time() - start
        time_history.append(timer)
        eps = max(eps * eps_decay, min_eps)

        changes = compare_policies(policy_, policy)
        change_history.append(changes)
        if changes == 0:
            unchanged += 1
            if unchanged >= 10:
                print("Converged!")
                break
        else:
            unchanged = 0

    
    V = {s:(max(Q[s].values()) if Q[s] else 0) for s in Q}
    policy = derive_policy(env, V)
    return V, policy, change_history, reward_history, time_history, Q
    


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
