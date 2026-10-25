"""Hospital courier gridworld: a small stochastic, episodic MDP for tabular RL.

A delivery robot starts at the pharmacy (S) and must bring medicine to a ward (G).
Wet-floor cells (H) are dangerous: entering one costs -10 and the robot is sent
back to S. The floor is slippery: with probability `slip` the robot moves in a
random perpendicular direction instead of the chosen one.
"""
import numpy as np

LAYOUT = [
    "....G",
    ".H.H.",
    "..H..",
    ".H...",
    "S....",
]

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["up", "right", "down", "left"]
MOVES = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
PERPENDICULAR = {UP: (LEFT, RIGHT), DOWN: (LEFT, RIGHT), LEFT: (UP, DOWN), RIGHT: (UP, DOWN)}

STEP_REWARD = -1.0
GOAL_REWARD = 10.0
HAZARD_REWARD = -10.0


class CourierGrid:
    def __init__(self, slip=0.1, max_steps=100, seed=None):
        self.n_rows, self.n_cols = len(LAYOUT), len(LAYOUT[0])
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        self.slip = slip
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.start = self._find("S")
        self.goal = self._find("G")
        self.hazards = {self.to_state(r, c) for r in range(self.n_rows)
                        for c in range(self.n_cols) if LAYOUT[r][c] == "H"}

    # ---- state <-> (row, col) ----
    def to_state(self, r, c):
        return r * self.n_cols + c

    def to_pos(self, s):
        return divmod(s, self.n_cols)

    def _find(self, ch):
        for r, row in enumerate(LAYOUT):
            if ch in row:
                return self.to_state(r, row.index(ch))

    # ---- deterministic effect of actually moving in direction `a` ----
    def _move(self, s, a):
        """Return (next_state, reward, terminated) if the robot really moves in direction a."""
        r, c = self.to_pos(s)
        dr, dc = MOVES[a]
        r2, c2 = r + dr, c + dc
        if not (0 <= r2 < self.n_rows and 0 <= c2 < self.n_cols):
            r2, c2 = r, c                      # bumped into a wall: stay
        s2 = self.to_state(r2, c2)
        if s2 == self.goal:
            return s2, GOAL_REWARD, True
        if s2 in self.hazards:
            return self.start, HAZARD_REWARD, False   # slip on wet floor, back to start
        return s2, STEP_REWARD, False

    # ---- model, used ONLY by value iteration (the non-RL reference) ----
    def outcomes(self, s, a):
        """List of (probability, next_state, reward, terminated) for taking action a in s."""
        p1, p2 = PERPENDICULAR[a]
        return [(1 - self.slip, *self._move(s, a)),
                (self.slip / 2, *self._move(s, p1)),
                (self.slip / 2, *self._move(s, p2))]

    # ---- interaction interface, used by the RL agent and baselines ----
    def reset(self):
        self.s = self.start
        self.t = 0
        return self.s

    def step(self, a):
        """Returns next_state, reward, terminated (goal reached), truncated (time limit)."""
        if self.rng.random() < self.slip:
            a = PERPENDICULAR[a][self.rng.integers(2)]
        self.s, reward, terminated = self._move(self.s, a)
        self.t += 1
        truncated = (not terminated) and self.t >= self.max_steps
        return self.s, reward, terminated, truncated
