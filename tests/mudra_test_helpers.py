"""
Shared synthetic hand-shape generator for mudra tests (no real mudra
footage exists yet -- see methods.md). Uses real mudra names whose
reference rules (mudra_reference.MUDRA_RULES) only check extended/curled
finger state -- no thumb-touch or finger-spread distance requirements --
so a simple synthetic hand shape can satisfy or violate them precisely
without needing detailed finger-position geometry.
"""

import numpy as np

FINGER_INDICES = {
    "thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20),
}

# name -> mudra_reference.MUDRA_RULES-compatible finger states.
SHAPES = {
    "mushti": {  # all curled -- MUDRA_RULES["mushti"]: curled=[index,middle,ring,pinky]
        "extended": {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False},
        "x_offsets": {"thumb": -0.03, "index": -0.015, "middle": 0.0, "ring": 0.015, "pinky": 0.03},
    },
    "soochi": {  # only index extended -- MUDRA_RULES["soochi"]: extended=[index], curled=[middle,ring,pinky]
        "extended": {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False},
        "x_offsets": {"thumb": -0.03, "index": 0.0, "middle": 0.015, "ring": 0.03, "pinky": 0.045},
    },
    "alapadma": {  # all extended -- MUDRA_RULES["alapadma"]: extended=[index,middle,ring,pinky]
        "extended": {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True},
        "x_offsets": {"thumb": -0.15, "index": -0.075, "middle": 0.0, "ring": 0.075, "pinky": 0.15},
    },
}


def make_hand_frame(extended, x_offsets, wrist=(0.5, 0.9), rng=None):
    rng = rng or np.random.default_rng(0)
    coords = {0: np.array(wrist, dtype=float)}
    for finger, (mcp_i, pip_i, dip_i, tip_i) in FINGER_INDICES.items():
        x = wrist[0] + x_offsets.get(finger, 0.0)
        mcp = np.array([x, wrist[1] - 0.1])
        if extended.get(finger, True):
            pip, dip, tip = [x, wrist[1] - 0.2], [x, wrist[1] - 0.3], [x, wrist[1] - 0.4]
        else:
            pip, dip, tip = [x, wrist[1] - 0.2], [x, wrist[1] - 0.12], [x, wrist[1] - 0.05]
        coords[mcp_i], coords[pip_i], coords[dip_i], coords[tip_i] = mcp, np.array(pip), np.array(dip), np.array(tip)
    jitter = rng.normal(0, 0.005, size=(21, 2))
    return {i: coords[i] + jitter[i] for i in range(21)}


def make_synthetic_dataset(n_frames_per_class=40, side="left", seed=0, shapes=None):
    import pandas as pd
    shapes = shapes or SHAPES
    rng = np.random.default_rng(seed)
    rows = []
    for mudra_name, shape in shapes.items():
        for _ in range(n_frames_per_class):
            frame = make_hand_frame(shape["extended"], shape["x_offsets"], rng=rng)
            row = {}
            for i in range(21):
                row[f"hand_{side}_{i}_x"] = frame[i][0]
                row[f"hand_{side}_{i}_y"] = frame[i][1]
            row["mudra"] = mudra_name
            row["hand_side"] = side
            rows.append(row)
    df = pd.DataFrame(rows)
    df["timestamp_ms"] = np.arange(len(df)) * 33
    return df
