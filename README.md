# WALL-E and EVE's Signal

A small, simple autonomous agent that searches a grid world for EVE using noisy
signal readings, a few hidden clues, and a limited battery.
Built for the VinnovateIT AI/ML recruitment task (Task 2). Pure Python 3, **no libraries needed**.

## Run it

```bash
python walle.py configs/mission1.json
python walle.py configs/mission2.json
python test_many.py        # runs 200 random missions and prints accuracy
```

Outputs are written to `outputs/`: a step-by-step trace (`*_trace.jsonl`) and a final report (`*_report.txt`).

## How it works

**Environment** (`Environment` in `walle.py`)
- A 10x10 grid loaded from a JSON mission config (see `configs/`).
- EVE's true position is hidden. The agent can only learn about it by acting.
- Actions: `move` (cost 1), `scan` (cost 2), `recharge` (only on the charger tile, +20).
- A scan returns a noisy signal: the direction to EVE (like NE) and a strength that falls as distance grows (noise of +-1 cell).
- Scanning also reveals any clue within 1 cell. A clue says "EVE is within 2 cells of this point" (the point is slightly off on purpose).

**Agent** (`Agent` in `walle.py`)
- **Belief:** a probability for every grid cell, starting from the initial signal. Each scan or clue multiplies the cells that fit the evidence and shrinks the ones that don't, then renormalises (a simple Bayes-style update).
- **Decisions**, in priority order:
  1. Stop when one cell holds at least 60% of the probability.
  2. Recharge if at the charger and not full.
  3. Head back to the charger if battery only just covers the trip home.
  4. Scan if it hasn't scanned for 3 moves.
  5. Otherwise move toward the most likely cell.
- **Trace:** every step logs action, reasoning, observation, battery, and current best guess.
- **Report:** evidence found, inferred location, confidence, and recommended next step.

## Config format

```json
{
  "seed": 42, "grid": [10, 10], "start": [1, 8], "charger": [0, 9], "eve": [7, 2],
  "max_steps": 80,
  "battery": {"start": 40, "max": 50},
  "costs": {"move": 1, "scan": 2, "recharge_gain": 20},
  "initial_signal": {"direction": "NE", "strength": 40, "timestamp": "00:00"},
  "clues": [[4, 5], [6, 3]]
}
```
`eve` is used only by the simulator; the agent never reads it.

## Results

- `mission1`: located EVE correctly in 13 steps (see `outputs/mission1_report.txt`).
- `test_many.py`: on 200 random missions in this simulator, the agent's guess was correct every time, in about 10 steps on average (max 56).

## Limitations (honest list)

- It's a toy simulation: small open grid, no obstacles, EVE never moves.
- The noise model is simple and the agent is built around that same model, so the 200/200 result does **not** mean it would work in a messier, real setting.
- It is a classical rule-based planner. There is no LLM, no learning, no multi-agent coordination, no decoy signals, no physics simulation, and no graphical UI (only an ASCII map).
- It reports an *inferred* location with a confidence; it never actually "sees" EVE.

## Possible next steps

Obstacles/walls, a moving EVE, decoy signals, a matplotlib visualisation, comparing this planner against an LLM-driven one.
