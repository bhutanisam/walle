"""WALL-E and EVE's Signal: a small autonomous search agent.

Three parts in one file:
  1. Environment : the hidden world (EVE's true location, clues, battery rules)
  2. Agent       : keeps a belief (probability for every cell) and decides actions
  3. run()       : loops agent <-> environment, writes trace + report
"""
import argparse, json, os, random

STRENGTH_PER_CELL = 5   # signal strength drops by 5 for every cell of distance


def sign(v):
    return (v > 0) - (v < 0)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def parse_direction(text):
    """'NE' -> (1, -1).  x grows to the East, y grows to the South."""
    sx = (1 if "E" in text else 0) - (1 if "W" in text else 0)
    sy = (1 if "S" in text else 0) - (1 if "N" in text else 0)
    return sx, sy


# ----------------------------------------------------------------- environment
class Environment:
    def __init__(self, cfg):
        self.rng = random.Random(cfg["seed"])
        self.w, self.h = cfg["grid"]
        self.eve = tuple(cfg["eve"])              # hidden from the agent
        self.charger = tuple(cfg["charger"])
        self.pos = tuple(cfg["start"])
        self.battery = cfg["battery"]["start"]
        self.max_battery = cfg["battery"]["max"]
        self.costs = cfg["costs"]
        # Each clue says "EVE is within 2 cells of <hint_center>". The centre is
        # shifted by up to 1 cell so a clue is helpful but not exact.
        self.clues = []
        for c in cfg["clues"]:
            hint = (self.eve[0] + self.rng.randint(-1, 1),
                    self.eve[1] + self.rng.randint(-1, 1))
            self.clues.append({"pos": tuple(c), "hint_center": hint, "found": False})

    def _reading(self):
        dx, dy = self.eve[0] - self.pos[0], self.eve[1] - self.pos[1]
        noisy = max(0, abs(dx) + abs(dy) + self.rng.choice([-1, 0, 1]))  # noisy sensor
        return {"direction": (sign(dx), sign(dy)),
                "strength": max(0, 100 - STRENGTH_PER_CELL * noisy)}

    def act(self, action, arg=None):
        obs = {"ok": True, "msg": ""}
        cost = self.costs["move"] if action == "move" else self.costs["scan"] if action == "scan" else 0
        if cost > self.battery:
            obs.update(ok=False, msg="not enough battery")
        elif action == "move":
            dx, dy = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}[arg]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                self.pos = (nx, ny); self.battery -= cost
            else:
                obs.update(ok=False, msg="wall")
        elif action == "scan":
            self.battery -= cost
            obs["reading"] = self._reading()
            obs["clues"] = []
            for c in self.clues:   # scanning finds clues within 1 cell
                if not c["found"] and max(abs(c["pos"][0] - self.pos[0]), abs(c["pos"][1] - self.pos[1])) <= 1:
                    c["found"] = True
                    obs["clues"].append({"pos": c["pos"], "hint_center": c["hint_center"]})
        elif action == "recharge":
            if self.pos != self.charger:
                obs.update(ok=False, msg="not at charger")
            else:
                self.battery = min(self.max_battery, self.battery + self.costs["recharge_gain"])
        obs["pos"], obs["battery"] = self.pos, self.battery
        return obs


# ----------------------------------------------------------------------- agent
class Agent:
    CONFIDENCE = 0.6   # stop when one cell holds this much probability

    def __init__(self, cfg):
        self.w, self.h = cfg["grid"]
        self.charger = tuple(cfg["charger"])
        self.pos = tuple(cfg["start"])
        self.battery = cfg["battery"]["start"]
        self.max_battery = cfg["battery"]["max"]
        self.move_cost = cfg["costs"]["move"]
        self.scan_cost = cfg["costs"]["scan"]
        self.cells = [(x, y) for x in range(self.w) for y in range(self.h)]
        self.belief = {c: 1 / len(self.cells) for c in self.cells}   # start: no idea
        self.scanned, self.since_scan, self.evidence = set(), 3, []
        sig = cfg["initial_signal"]
        sx, sy = parse_direction(sig["direction"])
        self._update_direction(self.pos, sx, sy, (100 - sig["strength"]) / STRENGTH_PER_CELL)
        self.evidence.append(f"initial signal: direction {sig['direction']}, strength {sig['strength']}")

    # ---- belief updates (Bayes-style: multiply by how well each cell fits, renormalise)
    def _normalise(self):
        total = sum(self.belief.values())
        for c in self.belief:
            self.belief[c] /= total

    def _update_direction(self, origin, sx, sy, dist_est):
        for c in self.cells:
            dx, dy = c[0] - origin[0], c[1] - origin[1]
            fits = (sign(dx), sign(dy)) == (sx, sy) and abs(abs(dx) + abs(dy) - dist_est) <= 1
            self.belief[c] *= 1.0 if fits else 0.02
        self._normalise()

    def _update_clue(self, center):
        for c in self.cells:
            near = max(abs(c[0] - center[0]), abs(c[1] - center[1])) <= 2
            self.belief[c] *= 1.0 if near else 0.05
        self._normalise()

    def observe(self, action, obs):
        if not obs["ok"]:
            return
        self.pos, self.battery = obs["pos"], obs["battery"]
        self.since_scan += 1
        if action == "scan":
            self.scanned.add(self.pos); self.since_scan = 0
            r = obs["reading"]
            self._update_direction(self.pos, *r["direction"], (100 - r["strength"]) / STRENGTH_PER_CELL)
            self.evidence.append(f"scan at {self.pos}: direction {r['direction']}, strength {r['strength']}")
            for c in obs["clues"]:
                self._update_clue(c["hint_center"])
                self.evidence.append(f"clue at {c['pos']}: EVE within 2 cells of {c['hint_center']}")

    # ---- decision making
    def best(self, exclude=None):
        cand = [c for c in self.cells if c != exclude]
        return max(cand, key=lambda c: (self.belief[c], -manhattan(c, self.pos)))

    def _step_toward(self, target):
        dx, dy = target[0] - self.pos[0], target[1] - self.pos[1]
        if abs(dx) >= abs(dy):
            return "E" if dx > 0 else "W"
        return "S" if dy > 0 else "N"

    def decide(self):
        """Returns (action, argument, reasoning)."""
        best = self.best()
        p = self.belief[best]
        to_charger = manhattan(self.pos, self.charger) * self.move_cost
        if p >= self.CONFIDENCE and self.scanned:
            return "finish", None, f"belief in {best} is {p:.0%} (>= {self.CONFIDENCE:.0%}), confident enough"
        if self.pos == self.charger and self.battery < self.max_battery * 0.9:
            return "recharge", None, f"battery {self.battery} is not full and I'm at the charger"
        if self.battery <= to_charger + 5 and self.pos != self.charger:
            return "move", self._step_toward(self.charger), f"battery {self.battery} only just covers the trip back ({to_charger}); heading to charger"
        if self.pos not in self.scanned and self.since_scan >= 3:
            return "scan", None, "haven't scanned for a while; gather new evidence here"
        target = best if best != self.pos else self.best(exclude=self.pos)
        return "move", self._step_toward(target), f"moving toward most likely cell {target} ({self.belief[target]:.0%})"


# ------------------------------------------------------------------------- run
def ascii_map(env, path, guess, found_clues):
    rows = []
    for y in range(env.h):
        row = ""
        for x in range(env.w):
            c = (x, y)
            ch = "."
            if c in path: ch = "*"
            if c in found_clues: ch = "#"
            if c == env.charger: ch = "C"
            if c == guess: ch = "G"
            if c == env.eve: ch = "E" if c != guess else "!"   # "!" = guess exactly right
            row += ch + " "
        rows.append(row)
    return "\n".join(rows)


def run(cfg, outdir, quiet=False):
    env, agent = Environment(cfg), Agent(cfg)
    trace, path, stop, failures = [], [env.pos], "max_steps reached", 0
    for step in range(1, cfg["max_steps"] + 1):
        action, arg, why = agent.decide()
        if action == "finish":
            stop = "confident"; break
        obs = env.act(action, arg)
        agent.observe(action, obs)
        failures = 0 if obs["ok"] else failures + 1
        path.append(env.pos)
        top = agent.best()
        trace.append({"step": step, "action": action, "arg": arg, "reasoning": why,
                      "ok": obs["ok"], "pos": env.pos, "battery": env.battery,
                      "best_guess": top, "best_prob": round(agent.belief[top], 3),
                      "reading": obs.get("reading"), "clues_found": [c["pos"] for c in obs.get("clues", [])]})
        if failures >= 3:
            stop = "stuck (actions keep failing)"; break

    guess = agent.best()
    conf = agent.belief[guess]
    found = {c["pos"] for c in env.clues if c["found"]}
    correct = guess == env.eve
    report = [
        f"MISSION REPORT: {cfg['name']}",
        "=" * 40,
        f"Stopped because : {stop}",
        f"Steps used      : {len(trace)}   Battery left: {env.battery}",
        f"Evidence found  :",
        *[f"  - {e}" for e in agent.evidence],
        f"Inferred EVE location : {guess}",
        f"Confidence            : {conf:.0%}",
        f"Status                : EVE is most likely stationary at {guess} (inferred, not directly seen)",
        f"Recommended next step : WALL-E should travel to {guess} and visually confirm."
        + ("" if conf >= Agent.CONFIDENCE else "  (Low confidence: scan more around there first.)"),
        "",
        f"[Evaluation only] True EVE location: {env.eve} -> guess {'CORRECT' if correct else 'WRONG'}",
        "",
        "Map  (* path, # clue found, C charger, G guess, E true EVE, ! guess = EVE)",
        ascii_map(env, set(path), guess, found),
    ]
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{cfg['name']}_trace.jsonl"), "w") as f:
        for t in trace:
            f.write(json.dumps(t) + "\n")
    with open(os.path.join(outdir, f"{cfg['name']}_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    if not quiet:
        print("\n".join(report))
    return correct, len(trace), conf


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run WALL-E's search mission.")
    ap.add_argument("config", help="path to a mission JSON file, e.g. configs/mission1.json")
    ap.add_argument("--out", default="outputs", help="folder for trace and report")
    args = ap.parse_args()
    with open(args.config) as f:
        run(json.load(f), args.out)
