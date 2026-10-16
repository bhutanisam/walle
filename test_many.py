"""Runs the agent on 200 random missions and prints how often it finds EVE."""
import random
from walle import run, STRENGTH_PER_CELL

def make_cfg(seed):
    r = random.Random(seed)
    start, charger = (r.randrange(10), r.randrange(10)), (0, 9)
    eve = (r.randrange(10), r.randrange(10))
    while eve == start:
        eve = (r.randrange(10), r.randrange(10))
    dx, dy = eve[0]-start[0], eve[1]-start[1]
    d = ("S" if dy > 0 else "N" if dy < 0 else "") + ("E" if dx > 0 else "W" if dx < 0 else "")
    strength = max(0, 100 - STRENGTH_PER_CELL * (abs(dx)+abs(dy)))
    clues = [[r.randrange(10), r.randrange(10)] for _ in range(5)]
    return {"name": f"rand{seed}", "seed": seed, "grid": [10, 10], "start": list(start),
            "charger": list(charger), "eve": list(eve), "max_steps": 80,
            "battery": {"start": 40, "max": 50},
            "costs": {"move": 1, "scan": 2, "recharge_gain": 20},
            "initial_signal": {"direction": d, "strength": strength, "timestamp": "00:00"},
            "clues": clues}

if __name__ == "__main__":
    ok, steps, n = 0, [], 200
    for s in range(n):
        c, st, _ = run(make_cfg(s), "/tmp/test_out", quiet=True)
        ok += c; steps.append(st)
    print(f"Correct: {ok}/{n} ({ok/n:.0%}), average steps: {sum(steps)/n:.1f}, max steps: {max(steps)}")
