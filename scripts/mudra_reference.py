"""
Phase 2B/4: Mudra reference definitions and correctness checking.

Encodes each mudra as a rule over the features already computed by
extract_features.py (hand_<side>_<finger>_extended, hand_<side>_thumb_to_<finger>_tip,
hand_<side>_<finger1>_<finger2>_spread). Definitions transcribed directly from
the descriptions given for this project, then independently verified against
real recorded footage (mudra_01.mov's known mudra sequence) -- see the "Real
bug found and fixed" note below and methods.md step 4 for that verification
and the threshold fix it led to. (This docstring used to say identification
was still blocked on ground-truth timestamps; that was true when first
written, stale by the time the fix below happened, and never updated --
exactly the kind of stale-doc mismatch worth catching on a later pass.)

"touch" is checked as thumb-to-fingertip distance (normalized by hand_scale
in extract_features.py -- wrist-to-middle-MCP "palm length", so this is
scale-invariant across videos shot at different distances from the camera)
below a per-finger threshold in TOUCH_THRESHOLDS. "spread" is checked as
adjacent-fingertip distance above SPREAD_THRESHOLD, same normalization.

Real bug found and fixed: this used to be a single absolute-distance
TOUCH_THRESHOLD (0.08, unguessed pixels-ish scale), which broke completely
on real video shot at a different distance than the reference photos --
confirmed directly when a frame that MUST be pataka (guaranteed by
recording order) failed pataka's own thumb-to-index touch check. Two
problems, both fixed: (1) un-normalized distances aren't comparable across
different hand-to-camera distances -- fixed by normalizing by hand_scale in
extract_features.py; (2) "touching" isn't one universal distance -- thumb-
to-pinky touching measures meaningfully closer than thumb-to-index touching
even after normalizing, because of hand anatomy (the thumb travels a
shorter path to reach the pinky than the index). A single blanket threshold
can't fit both. Calibrated per finger instead, using real median/p75
touching-vs-not-touching distances measured across the full training
dataset (data/mudra_training/dataset_combined_full.csv, ~13,600 images),
not guessed.

Usage:
    from mudra_reference import check_mudra
    result = check_mudra(feature_row, "pataka", side="right")
"""

# Per-finger thumb-touch thresholds -- calibrated from real training-data
# distributions (see module docstring). Each value sits just above that
# finger's own touching-case p75, comfortably below any measured non-
# touching case where a same-mudra comparison existed (e.g. index: touching
# median 0.32, p75 0.91, vs. spread-away median 1.35).
TOUCH_THRESHOLDS = {
    "index": 1.05,   # bumped from the training-photo p75 (0.91) after checking against real video: a guaranteed-correct real frame measured 0.97, just over the original cutoff -- real footage has more natural variation than posed photos
    "middle": 0.55,
    "ring": 0.70,
    "pinky": 0.55,
}
SPREAD_THRESHOLD = 0.28  # normalized adjacent-fingertip distance; fingers "spread apart" (calibrated from pataka-together vs. alapadma-spread)

# Each rule: which fingers must be extended/curled, and any thumb-touch or
# adjacent-finger-spread requirements. Fingers not mentioned are unconstrained.
MUDRA_RULES = {
    "pataka": {
        "extended": ["index", "middle", "ring", "pinky"],
        "curled": [],
        "thumb_touches": "index",
        "description": "All fingers straight and together, thumb bent to touch index finger base.",
    },
    "tripataka": {
        "extended": ["index", "middle", "pinky"],
        "curled": ["ring"],
        "thumb_touches": "index",
        "description": "Pataka with the ring finger bent.",
    },
    "ardhapataka": {
        "extended": ["index", "middle"],
        "curled": ["ring", "pinky"],
        "thumb_touches": "index",
        "description": "Tripataka with the little finger also bent.",
    },
    "kartarimukh": {
        "extended": ["index", "middle"],
        "curled": ["ring", "pinky"],
        "spread_pairs": [("index", "middle")],
        "thumb_touches": ["ring", "pinky"],
        "description": "Thumb rests on the tips of the ring and pinky fingers; index and middle spread into a V shape.",
    },
    "mayur": {
        "extended": ["index", "middle", "pinky"],
        "curled": ["ring"],
        "thumb_touches": "ring",
        "description": "Like Tripataka, except the thumb touches the tip of the ring finger.",
    },
    "ardhachandra": {
        "extended": ["index", "middle", "ring", "pinky"],
        "curled": [],
        "thumb_spread_from": "index",
        "description": "Pataka with the thumb stretched outward, separated from the fingers.",
    },
    "aral": {
        "extended": ["middle", "ring", "pinky"],
        "curled": ["index"],
        "description": "Pataka with the index finger bent.",
    },
    "shuktund": {
        "extended": ["middle", "pinky"],
        "curled": ["index", "ring"],
        "description": "Aral with the ring finger also bent.",
    },
    "mushti": {
        "extended": [],
        "curled": ["index", "middle", "ring", "pinky"],
        "description": "All four fingers folded into a fist, thumb on top.",
    },
    "shikhar": {
        "extended": ["thumb"],
        "curled": ["index", "middle", "ring", "pinky"],
        "description": "Mushti with the thumb raised upward.",
    },
    "kapitth": {
        "extended": [],
        "curled": ["middle", "ring", "pinky"],
        "thumb_touches": "index",
        "description": "Shikhar with the index finger bent and joined with the thumb on top.",
    },
    # Katakamukh has 3 distinct valid hand shapes (confirmed against the
    # recording, where the name is spoken 3 times, and the reference chart,
    # which shows 3 images under one label). No single rule covers all three.
    "katakamukh_1": {
        "curled": ["index", "middle", "ring", "pinky"],
        "thumb_touches": ["index", "middle"],
        "description": "Like Kapitth's thumb-to-index touch, but middle finger also bent to touch the thumb.",
    },
    "katakamukh_2": {
        "extended": ["index", "middle", "ring", "pinky"],
        "thumb_touches": ["index", "middle"],
        "spread_pairs": [("ring", "pinky")],
        "description": "Index and middle held flat/straight, touching the flat thumb; ring and pinky spread into a V.",
    },
    "katakamukh_3": {
        "curled": ["index", "middle"],
        "extended": ["ring", "pinky"],
        "thumb_touches": ["index"],
        "spread_pairs": [("ring", "pinky")],
        "description": "Kapitth's index-to-thumb touch, with ring and pinky lifted into a V shape.",
    },
    "soochi": {
        "extended": ["index"],
        "curled": ["middle", "ring", "pinky"],
        "description": "Katakamukh with the index finger raised upright.",
    },
    "chandrakala": {
        "extended": ["index"],
        "curled": ["middle", "ring", "pinky"],
        "thumb_spread_from": "index",
        "description": "Soochi with the thumb stretched outward, separated from the fingers.",
    },
    "padmakosh": {
        "extended": ["index", "middle", "ring", "pinky"],
        "curled": [],
        "description": "All fingers spread slightly apart while remaining curved inward.",
    },
    "sarpasheesh": {
        "extended": ["index", "middle", "ring", "pinky"],
        "curled": [],
        "description": "Pataka with the fingers bent slightly (less straight than Pataka).",
    },
    "mrigasheesh": {
        "extended": ["pinky", "thumb"],
        "curled": ["index", "middle", "ring"],
        "description": "Sarpasheesh with the little finger and thumb stretched upright, other fingers bent.",
    },
    "singhamukh": {
        "curled": ["middle", "ring"],
        "extended": ["index", "pinky"],
        "thumb_touches": ["middle", "ring"],
        "description": "Middle and ring finger folded in to join the thumb; index and pinky spread apart.",
    },
    "kangul": {
        "extended": ["index", "middle", "pinky"],
        "curled": ["ring"],
        "description": "Padmakosh with the ring finger bent completely inward.",
    },
    "alapadma": {
        "extended": ["index", "middle", "ring", "pinky"],
        "curled": [],
        "description": "All fingers spread apart, ending slightly curved (fully bloomed flower).",
    },
    "chatur": {
        "extended": ["pinky"],
        "curled": ["index", "middle"],
        "thumb_touches": "ring",
        "description": "Little finger stretched straight, thumb touches the base of the ring finger.",
    },
    "bhramara": {
        "curled": ["index"],
        "extended": ["ring", "pinky"],
        "thumb_touches": "middle",
        "description": "Index folded, middle finger and thumb touch, ring and pinky spread.",
    },
    "hansasya": {
        "extended": ["middle", "ring", "pinky"],
        "curled": [],
        "thumb_touches": "index",
        "description": "Aral hasta with the index finger placed on the thumb.",
    },
    "hanspaksha": {
        "extended": ["index", "middle", "ring", "pinky"],
        "curled": [],
        "description": "Sarpasheesh with the little finger stretched straight.",
    },
    "sandansh": {
        "extended": ["index", "middle", "ring", "pinky"],
        "curled": [],
        "description": "Padmakosh fingers repeatedly opened and shut (a motion, not a static shape).",
    },
    "mukul": {
        "extended": [],
        "curled": ["index", "middle", "ring", "pinky"],
        "thumb_touches": ["index", "middle", "ring", "pinky"],
        "description": "All fingertips joined together into a point.",
    },
    "tamrachud": {
        "extended": ["index"],
        "curled": ["middle", "ring", "pinky"],
        "description": "Mukul with the index finger raised and kept crooked.",
    },
    "trishool": {
        "extended": ["index", "middle", "ring"],
        "curled": [],
        "thumb_touches": "pinky",
        "description": "Little finger and thumb tips joined, other three fingers spread.",
    },
}


def check_mudra(feature_row, mudra_name, side="right"):
    """
    Compare one frame's computed hand features against a mudra's reference
    rule. Returns a list of mismatch descriptions (empty list = matches).
    `feature_row` is a dict-like row from an extract_features.py output CSV.
    """
    rule = MUDRA_RULES.get(mudra_name.lower())
    if rule is None:
        raise ValueError(f"No reference definition for mudra '{mudra_name}'")

    mismatches = []
    prefix = f"hand_{side}"

    for finger in rule.get("extended", []):
        col = f"{prefix}_{finger}_extended"
        if col in feature_row and feature_row[col] is not None and not bool(feature_row[col]):
            mismatches.append(f"{finger} should be extended but was curled")

    for finger in rule.get("curled", []):
        col = f"{prefix}_{finger}_extended"
        if col in feature_row and feature_row[col] is not None and bool(feature_row[col]):
            mismatches.append(f"{finger} should be curled but was extended")

    thumb_touches = rule.get("thumb_touches", [])
    if isinstance(thumb_touches, str):
        thumb_touches = [thumb_touches]
    for finger in thumb_touches:
        col = f"{prefix}_thumb_to_{finger}_tip"
        if col in feature_row and feature_row[col] is not None:
            if feature_row[col] > TOUCH_THRESHOLDS[finger]:
                mismatches.append(f"thumb should touch {finger} but was too far away")

    for f1, f2 in rule.get("spread_pairs", []):
        pair_cols = [f"{prefix}_{f1}_{f2}_spread", f"{prefix}_{f2}_{f1}_spread"]
        col = next((c for c in pair_cols if c in feature_row), None)
        if col and feature_row[col] is not None:
            if feature_row[col] < SPREAD_THRESHOLD:
                mismatches.append(f"{f1} and {f2} should be spread apart but were close together")

    if "thumb_spread_from" in rule:
        finger = rule["thumb_spread_from"]
        col = f"{prefix}_thumb_to_{finger}_tip"
        if col in feature_row and feature_row[col] is not None:
            if feature_row[col] < TOUCH_THRESHOLDS[finger]:
                mismatches.append(f"thumb should be stretched away from {finger} but was too close")

    return mismatches


# Mudras with more than one valid hand shape. check_mudra_variants() picks
# whichever variant the measured hand is closest to, rather than forcing a
# match against just one.
MUDRA_VARIANTS = {
    "katakamukh": ["katakamukh_1", "katakamukh_2", "katakamukh_3"],
}


def check_mudra_variants(feature_row, mudra_name, side="right"):
    """
    Like check_mudra(), but for mudras with multiple valid forms. Tries every
    known variant and returns (best_variant_name, mismatches_for_that_variant)
    -- "best" meaning fewest mismatches, i.e. the variant the hand shape is
    closest to. For mudras with only one form, behaves like check_mudra().
    """
    name = mudra_name.lower()
    variant_names = MUDRA_VARIANTS.get(name, [name])

    best_variant, best_mismatches = None, None
    for variant in variant_names:
        mismatches = check_mudra(feature_row, variant, side=side)
        if best_mismatches is None or len(mismatches) < len(best_mismatches):
            best_variant, best_mismatches = variant, mismatches

    return best_variant, best_mismatches
