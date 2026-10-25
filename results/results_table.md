| Method | Eval return (mean ± std over seeds) | Success rate | Episode length |
|---|---|---|---|
| Random baseline | -185.34 ± 2.31 | 0.103 ± 0.013 | 95.5 ± 0.8 |
| Shortest-path rule | -48.90 ± 4.28 | 0.996 ± 0.003 | 21.8 ± 1.1 |
| Value iteration (knows model) | 0.63 ± 0.23 | 1.000 ± 0.000 | 9.4 ± 0.2 |
| SARSA eps=0.01 | 0.30 ± 0.36 | 1.000 ± 0.000 | 9.6 ± 0.2 |
| SARSA eps=0.1 | 0.38 ± 0.34 | 1.000 ± 0.000 | 9.6 ± 0.2 |
| SARSA eps=0.3 | 0.11 ± 0.98 | 0.998 ± 0.005 | 9.7 ± 0.8 |
