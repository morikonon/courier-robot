"""Baselines, the SARSA agent, and value iteration (model-based reference)."""
import numpy as np

from env import MOVES


def greedy_action(Q, s, rng):
    """Action with the highest Q value; ties are broken at random."""
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))


def epsilon_greedy(Q, s, epsilon, rng):
    """With probability epsilon explore (random action), otherwise exploit (greedy action)."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return greedy_action(Q, s, rng)


# ---------------- baselines ----------------
def random_policy(env, s, rng):
    return int(rng.integers(env.n_actions))


def shortest_path_rule(env, s, rng):
    """Fixed rule: pick a move that reduces Manhattan distance to the goal. Ignores hazards and slip."""
    r, c = env.to_pos(s)
    gr, gc = env.to_pos(env.goal)
    good = [a for a, (dr, dc) in MOVES.items()
            if abs(r + dr - gr) + abs(c + dc - gc) < abs(r - gr) + abs(c - gc)]
    return int(rng.choice(good))


# ---------------- SARSA (on-policy TD control) ----------------
def train_sarsa(env, episodes, alpha, gamma, epsilon, rng):
    """Tabular SARSA with an epsilon-greedy behaviour policy.

    Update: Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
    Returns the learned Q table and the (undiscounted) return of every training episode.
    """
    Q = np.zeros((env.n_states, env.n_actions))
    episode_returns = np.zeros(episodes)
    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon, rng)
        total = 0.0
        while True:
            s2, r, terminated, truncated = env.step(a)
            total += r
            if terminated:                     # goal: no future value
                Q[s, a] += alpha * (r - Q[s, a])
                break
            a2 = epsilon_greedy(Q, s2, epsilon, rng)
            Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
            if truncated:                      # time limit is not a real end: we still bootstrapped
                break
            s, a = s2, a2
        episode_returns[ep] = total
    return Q, episode_returns


# ---------------- value iteration (needs the model; non-RL reference) ----------------
def value_iteration(env, gamma, tol=1e-8):
    V = np.zeros(env.n_states)
    while True:
        Q = np.zeros((env.n_states, env.n_actions))
        for s in range(env.n_states):
            for a in range(env.n_actions):
                Q[s, a] = sum(p * (r + (0 if done else gamma * V[s2]))
                              for p, s2, r, done in env.outcomes(s, a))
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return Q
        V = V_new


# ---------------- evaluation ----------------
def evaluate(env, policy, episodes, rng):
    """Run `policy(env, s, rng)` without learning. Returns per-episode returns, successes, lengths."""
    returns, successes, lengths = [], [], []
    for _ in range(episodes):
        s = env.reset()
        total, done, success = 0.0, False, False
        while not done:
            s, r, terminated, truncated = env.step(policy(env, s, rng))
            total += r
            success = terminated
            done = terminated or truncated
        returns.append(total)
        successes.append(success)
        lengths.append(env.t)
    return np.array(returns), np.array(successes), np.array(lengths)


def q_policy(Q):
    """Greedy policy (no exploration) derived from a Q table."""
    return lambda env, s, rng: greedy_action(Q, s, rng)
