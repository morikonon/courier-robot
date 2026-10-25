# CSCI3235 Midterm 1 — Hospital Courier Robot with SARSA

A delivery robot learns a safe route through a slippery 5×5 hospital floor with tabular SARSA.
It is compared with a random baseline, a fixed shortest-path rule, and value iteration (a model-based reference).

## Setup

Tested with Python 3.12.6, numpy 1.26.4, matplotlib 3.8.3 on macOS. No GPU, internet or datasets are needed.

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Run

```bash
python run_experiments.py
```

Expected runtime: about 10 seconds on a laptop CPU.
The script prints the results table and writes:

- `figures/fig1_environment_policy.png`: the map, a transition example, and the learned vs. optimal policy
- `figures/fig2_training_curve.png`: SARSA training return (mean ± std over 10 seeds)
- `figures/fig3_final_comparison.png`: greedy evaluation of SARSA vs. baselines
- `figures/fig4_epsilon_study.png`: effect of epsilon (0.01 / 0.1 / 0.3)
- `results/summary.json`, `results/results_table.md`: all numbers used in the report

## Random seeds

- Training run k (k = 0…9) uses environment seed `k` and agent seed `k`.
- Evaluation of run k uses a separate stream: environment and policy seed `10000 + k`.
- Baselines are evaluated on the same evaluation seeds, so all methods face the same randomness.

Running the script twice produces identical numbers.

## Hyperparameters

| Parameter | Value |
|---|---|
| alpha (step size) | 0.1 |
| gamma (discount) | 0.99 |
| epsilon (exploration) | 0.1; the study also uses 0.01 and 0.3 |
| training episodes per seed | 1000 |
| evaluation episodes per seed | 200 (greedy, no learning) |
| slip probability | 0.1 |
| time limit | 100 steps |

## Files

| File | Contents |
|---|---|
| `env.py` | `CourierGrid` environment: map, slip, rewards, time limit, and the model used by value iteration |
| `agents.py` | Random and shortest-path baselines, SARSA training, value iteration, evaluation |
| `run_experiments.py` | All experiments, the results table, and the figures |
| `report/report.pdf` | The report, with the learning record as the last page |
| `requirements.txt` | Dependencies |
