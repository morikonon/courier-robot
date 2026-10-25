"""Runs every experiment in the report and writes figures/ and results/.

Usage:  python run_experiments.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env import CourierGrid, LAYOUT, ACTION_NAMES
from agents import (train_sarsa, value_iteration, evaluate, q_policy,
                    random_policy, shortest_path_rule)

SEEDS = list(range(10))          # 10 independent runs (brief requires >= 5)
EVAL_SEED_OFFSET = 10_000        # evaluation uses a different random stream than training
EPISODES = 1000                  # training episodes per run
EVAL_EPISODES = 200              # greedy evaluation episodes per run
ALPHA, GAMMA, EPSILON = 0.1, 0.99, 0.1
EPSILON_STUDY = [0.01, 0.1, 0.3]
SMOOTH = 20                      # moving-average window for training curves

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
RES = Path("results"); RES.mkdir(exist_ok=True)


def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def eval_stats(env_seed, policy):
    env = CourierGrid(seed=env_seed)
    rets, succ, lens = evaluate(env, policy, EVAL_EPISODES, np.random.default_rng(env_seed))
    return rets.mean(), succ.mean(), lens.mean()


def run_sarsa(epsilon):
    """Train + evaluate SARSA for every seed. Returns training curves, eval stats, Q tables."""
    curves, stats, Qs = [], [], []
    for seed in SEEDS:
        env = CourierGrid(seed=seed)
        Q, rets = train_sarsa(env, EPISODES, ALPHA, GAMMA, epsilon, np.random.default_rng(seed))
        curves.append(rets)
        stats.append(eval_stats(EVAL_SEED_OFFSET + seed, q_policy(Q)))
        Qs.append(Q)
    return np.array(curves), np.array(stats), Qs


def summarize(stats):
    """stats: (n_seeds, 3) of per-seed mean return / success rate / length -> mean and std over seeds."""
    m, s = stats.mean(axis=0), stats.std(axis=0)
    return {"return_mean": m[0], "return_std": s[0], "success_mean": m[1], "success_std": s[1],
            "length_mean": m[2], "length_std": s[2]}


def main():
    results = {"config": {"seeds": SEEDS, "episodes": EPISODES, "eval_episodes": EVAL_EPISODES,
                          "alpha": ALPHA, "gamma": GAMMA, "epsilon": EPSILON,
                          "epsilon_study": EPSILON_STUDY}}

    # --- baselines and model-based reference, evaluated on the same evaluation seeds ---
    env = CourierGrid()
    Q_opt = value_iteration(env, GAMMA)
    methods = {
        "Random baseline": random_policy,
        "Shortest-path rule": shortest_path_rule,
        "Value iteration (knows model)": q_policy(Q_opt),
    }
    table = {name: summarize(np.array([eval_stats(EVAL_SEED_OFFSET + s, pol) for s in SEEDS]))
             for name, pol in methods.items()}

    # --- SARSA, including the epsilon study ---
    sarsa = {eps: run_sarsa(eps) for eps in EPSILON_STUDY}
    for eps in EPSILON_STUDY:
        table[f"SARSA eps={eps}"] = summarize(sarsa[eps][1])
    for eps in EPSILON_STUDY:   # average return over the last 100 training episodes (with exploration)
        last = sarsa[eps][0][:, -100:].mean(axis=1)
        table[f"SARSA eps={eps}"]["train_last100_mean"] = last.mean()
        table[f"SARSA eps={eps}"]["train_last100_std"] = last.std()
    results["table"] = table

    # per-seed numbers and agreement of each learned greedy policy with the optimal one
    decision_states = [s for s in range(env.n_states) if s != env.goal and s not in env.hazards]
    for eps in EPSILON_STUDY:
        _, stats, Qs = sarsa[eps]
        agree = [np.mean([Q[s].argmax() == Q_opt[s].argmax() for s in decision_states]) for Q in Qs]
        table[f"SARSA eps={eps}"]["policy_agreement_mean"] = float(np.mean(agree))
        table[f"SARSA eps={eps}"]["per_seed_eval_return"] = stats[:, 0].tolist()
    results["optimal_V_start"] = float(Q_opt[env.start].max())

    with open(RES / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print_table(table)

    curves, _, Qs = sarsa[EPSILON]
    plot_grid_and_policy(env, Qs[0], Q_opt)
    plot_training(curves, table)
    plot_comparison(table)
    plot_epsilon_study(sarsa, table)
    print(f"\nFigures saved to {FIG}/, numbers saved to {RES}/summary.json")


# ---------------- output ----------------
def print_table(table):
    lines = ["| Method | Eval return (mean ± std over seeds) | Success rate | Episode length |",
             "|---|---|---|---|"]
    for name, r in table.items():
        lines.append(f"| {name} | {r['return_mean']:.2f} ± {r['return_std']:.2f} | "
                     f"{r['success_mean']:.3f} ± {r['success_std']:.3f} | "
                     f"{r['length_mean']:.1f} ± {r['length_std']:.1f} |")
    text = "\n".join(lines)
    print(text)
    (RES / "results_table.md").write_text(text + "\n")


ARROWS = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}   # (dx, dy) in plot coords


def draw_grid(ax, env, title):
    colors = {"S": "#cfe8ff", "G": "#c8f0c8", "H": "#f6c4c4", ".": "white"}
    for r, row in enumerate(LAYOUT):
        for c, ch in enumerate(row):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=colors[ch], edgecolor="gray"))
            if ch != ".":
                ax.text(c, r + 0.33, ch, ha="center", va="center", fontsize=9, color="dimgray")
    ax.set_xlim(-0.5, env.n_cols - 0.5); ax.set_ylim(env.n_rows - 0.5, -0.5)
    ax.set_xticks(range(env.n_cols)); ax.set_yticks(range(env.n_rows))
    ax.set_xlabel("column"); ax.set_ylabel("row"); ax.set_aspect("equal"); ax.set_title(title, fontsize=10)


def draw_policy(ax, env, Q):
    for s in range(env.n_states):
        if s == env.goal or s in env.hazards:
            continue
        r, c = env.to_pos(s)
        dx, dy = ARROWS[int(np.argmax(Q[s]))]
        ax.arrow(c - dx / 2, r - dy / 2, dx, dy, head_width=0.15, length_includes_head=True, color="black")


def plot_grid_and_policy(env, Q_sarsa, Q_opt):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    draw_grid(axes[0], env, "(a) Environment and one transition")
    axes[0].annotate("", xy=(1, 4), xytext=(0, 4),
                     arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    axes[0].text(2.6, 4.05, "s=(4,0), a=right\nr=-1, s'=(4,1)", ha="center", va="center", fontsize=8, color="blue")
    draw_grid(axes[1], env, f"(b) SARSA greedy policy (seed 0, eps={EPSILON})")
    draw_policy(axes[1], env, Q_sarsa)
    draw_grid(axes[2], env, "(c) Optimal policy (value iteration)")
    draw_policy(axes[2], env, Q_opt)
    fig.tight_layout(); fig.savefig(FIG / "fig1_environment_policy.png", dpi=160); plt.close(fig)


def plot_training(curves, table):
    sm = np.array([moving_average(c, SMOOTH) for c in curves])
    x = np.arange(SMOOTH, EPISODES + 1)
    m, s = sm.mean(axis=0), sm.std(axis=0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(x, m, label=f"SARSA training (eps={EPSILON}), mean of {len(SEEDS)} seeds")
    ax.fill_between(x, m - s, m + s, alpha=0.25, label="± 1 std over seeds")
    for name, style in [("Random baseline", ":"), ("Shortest-path rule", "--"),
                        ("Value iteration (knows model)", "-.")]:
        ax.axhline(table[name]["return_mean"], ls=style, color="gray", label=f"{name} (eval)")
    ax.set_xlabel("training episode"); ax.set_ylabel(f"return ({SMOOTH}-episode moving average)")
    ax.set_title("Figure 2. SARSA training performance"); ax.legend(fontsize=7.5, loc="lower right")
    ax.set_ylim(-200, 10)
    fig.tight_layout(); fig.savefig(FIG / "fig2_training_curve.png", dpi=160); plt.close(fig)


def plot_comparison(table):
    names = ["Random baseline", "Shortest-path rule", f"SARSA eps={EPSILON}", "Value iteration (knows model)"]
    labels = ["Random", "Shortest-path\nrule", f"SARSA\n(eps={EPSILON})", "Value iteration\n(reference)"]
    means = [table[n]["return_mean"] for n in names]
    stds = [table[n]["return_std"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=["#bbbbbb", "#999999", "#4c8bd6", "#7fbf7f"])
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, min(v, 0) - 6, f"{v:.1f}", ha="center", va="top", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("greedy evaluation return\n(mean ± std over 10 seeds)")
    ax.set_title(f"Figure 3. Final comparison ({EVAL_EPISODES} eval episodes per seed)")
    ax.set_ylim(min(means) - 30, 15)
    fig.tight_layout(); fig.savefig(FIG / "fig3_final_comparison.png", dpi=160); plt.close(fig)


def plot_epsilon_study(sarsa, table):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(SMOOTH, EPISODES + 1)
    for eps in EPSILON_STUDY:
        sm = np.array([moving_average(c, SMOOTH) for c in sarsa[eps][0]])
        m, s = sm.mean(axis=0), sm.std(axis=0)
        ax.plot(x, m, label=f"eps={eps}  (greedy eval: {table[f'SARSA eps={eps}']['return_mean']:.2f})")
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    ax.set_xlabel("training episode"); ax.set_ylabel(f"training return ({SMOOTH}-ep. moving avg)")
    ax.set_title("Figure 4. Effect of exploration rate epsilon (mean ± std, 10 seeds)")
    ax.set_ylim(-140, 10); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig4_epsilon_study.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
