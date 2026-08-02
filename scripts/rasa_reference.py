"""
Navarasa (nine rasas) reference: maps each rasa to a characteristic pattern
of MediaPipe FaceLandmarker blendshape scores (mouthSmileLeft, browDownLeft,
eyeWideRight, jawOpen, noseSneerLeft, etc. -- 52 named, semantically
meaningful facial-muscle-activation scores per frame, extracted by
extract_landmarks.py's face_landmarker.task pass).

Same architecture as mudra_reference.py (check_X() / classify_X() over a
dict of measured features against hand-written criteria), applied to faces
instead of hands. Chose this over training a classifier from scratch --
unlike mudras, no rasa-specific labeled dataset was searched for yet, and
blendshapes are already close enough to the rasa table's own descriptions
("soft smile," "eyebrows drawn together," "wrinkled nose") that direct
rules are a reasonable first attempt, same as mudra_reference.py was before
any training data existed for it.

HIGH_THRESHOLD/LOW_THRESHOLD are reasoned defaults, NOT calibrated against
real abhinaya footage -- none exists in this project yet. Same honesty
caveat as taal_reference.py's sam-detection gap.

Confidence genuinely varies by rasa, not uniformly "these are all guesses":
some map to specific, well-matched blendshapes (Raudra's browDown+eyeWide,
Bibhatsa's noseSneer, Adbhuta's browUp+eyeWide+jawOpen). Others are much
weaker fits -- Vira (heroism) is mostly about a STEADY, composed quality
sustained over time and posture, which a single frame's blendshape snapshot
fundamentally can't capture; its definition here is closer to "no other
rasa is showing" than a positive signature. Each rasa's "confidence" field
says this plainly rather than presenting all nine as equally reliable.
"""

HIGH_THRESHOLD = 0.4  # a blendshape at or above this reads as meaningfully activated
LOW_THRESHOLD = 0.15  # at or below this reads as inactive/relaxed

RASA_DEFINITIONS = {
    "shringara": {
        "display_name": "Shringara (Love)",
        "criteria": [
            ("mouthSmileLeft", "high"), ("mouthSmileRight", "high"),
            ("browDownLeft", "low"), ("browDownRight", "low"),
            ("eyeSquintLeft", "high"), ("eyeSquintRight", "high"),
        ],
        "confidence": "medium",
        "description": "Soft smile, relaxed eyebrows, gentle gaze, slightly lowered eyelids.",
    },
    "hasya": {
        "display_name": "Hasya (Joy/Laughter)",
        "criteria": [
            ("mouthSmileLeft", "high"), ("mouthSmileRight", "high"),
            ("cheekSquintLeft", "high"), ("cheekSquintRight", "high"),
        ],
        "confidence": "high",
        "description": "Bright smile, raised cheeks, sparkling eyes.",
    },
    "karuna": {
        "display_name": "Karuna (Compassion/Sadness)",
        "criteria": [
            ("mouthFrownLeft", "high"), ("mouthFrownRight", "high"),
            ("browInnerUp", "high"),
            ("mouthSmileLeft", "low"), ("mouthSmileRight", "low"),
        ],
        "confidence": "high",
        "description": "Downturned lips, moist-looking eyes, lowered eyebrows.",
    },
    "raudra": {
        "display_name": "Raudra (Anger)",
        "criteria": [
            ("browDownLeft", "high"), ("browDownRight", "high"),
            ("eyeWideLeft", "high"), ("eyeWideRight", "high"),
            ("mouthPressLeft", "high"), ("mouthPressRight", "high"),
        ],
        "confidence": "high",
        "description": "Eyes wide, eyebrows drawn together, nostrils flared, tense jaw. "
                        "(No blendshape directly measures nostril flare -- not checked.)",
    },
    "vira": {
        "display_name": "Vira (Heroism)",
        "criteria": [
            ("browDownLeft", "low"), ("browDownRight", "low"),
            ("mouthSmileLeft", "low"), ("mouthSmileRight", "low"),
            ("jawOpen", "low"),
            ("eyeWideLeft", "low"), ("eyeWideRight", "low"),
        ],
        "confidence": "low",
        "description": "Steady gaze, lifted chin, composed face -- genuinely weak fit: 'steadiness' "
                        "and 'confidence' aren't single-frame facial-muscle signatures. This is "
                        "closer to 'no other rasa is showing' than a positive signature.",
    },
    "bhayanaka": {
        "display_name": "Bhayanaka (Fear)",
        "criteria": [
            ("eyeWideLeft", "high"), ("eyeWideRight", "high"),
            ("browInnerUp", "high"),
            ("browOuterUpLeft", "high"), ("browOuterUpRight", "high"),
            ("jawOpen", "mid"),
        ],
        "confidence": "medium",
        "description": "Wide eyes, raised eyebrows, slightly open mouth.",
    },
    "bibhatsa": {
        "display_name": "Bibhatsa (Disgust)",
        "criteria": [
            ("noseSneerLeft", "high"), ("noseSneerRight", "high"),
            ("mouthUpperUpLeft", "high"), ("mouthUpperUpRight", "high"),
            ("eyeSquintLeft", "high"), ("eyeSquintRight", "high"),
        ],
        "confidence": "high",
        "description": "Wrinkled nose, curled lip, narrowed eyes.",
    },
    "adbhuta": {
        "display_name": "Adbhuta (Wonder)",
        "criteria": [
            ("browInnerUp", "high"),
            ("browOuterUpLeft", "high"), ("browOuterUpRight", "high"),
            ("eyeWideLeft", "high"), ("eyeWideRight", "high"),
            ("jawOpen", "high"),
        ],
        "confidence": "high",
        "description": "Raised eyebrows, open eyes, relaxed open mouth.",
    },
    "shanta": {
        "display_name": "Shanta (Peace)",
        "criteria": [
            ("mouthSmileLeft", "mid"), ("mouthSmileRight", "mid"),
            ("browDownLeft", "low"), ("browDownRight", "low"),
            ("browInnerUp", "low"),
            ("eyeWideLeft", "low"), ("eyeWideRight", "low"),
            ("eyeSquintLeft", "low"), ("eyeSquintRight", "low"),
        ],
        "confidence": "medium",
        "description": "Calm eyes, slight smile, relaxed forehead.",
    },
}


def _check_criterion(blendshapes, name, direction):
    score = blendshapes.get(name)
    if score is None:
        return None  # missing data -- not evaluable, not a mismatch
    if direction == "high":
        return score >= HIGH_THRESHOLD
    if direction == "low":
        return score <= LOW_THRESHOLD
    if direction == "mid":
        return LOW_THRESHOLD < score < HIGH_THRESHOLD
    raise ValueError(f"unknown direction '{direction}'")


def check_rasa(blendshapes, rasa_name):
    """
    Compare one frame's (or one averaged window's) face blendshape scores
    against a rasa's reference criteria. Returns a list of mismatch
    descriptions (empty list = matches). `blendshapes` is a dict-like of
    blendshape name -> score -- accepts either the raw "face_mouthSmileLeft"
    column-name form or the bare "mouthSmileLeft" form.
    """
    if rasa_name not in RASA_DEFINITIONS:
        raise ValueError(f"No reference definition for rasa '{rasa_name}'. Known: {sorted(RASA_DEFINITIONS)}")
    rasa = RASA_DEFINITIONS[rasa_name]

    normalized = {}
    for k, v in blendshapes.items():
        key = k[len("face_"):] if k.startswith("face_") else k
        normalized[key] = v

    mismatches = []
    for name, direction in rasa["criteria"]:
        result = _check_criterion(normalized, name, direction)
        if result is False:
            verb = {"high": "high", "low": "low", "mid": "moderate"}[direction]
            mismatches.append(f"{name} should be {verb} for {rasa['display_name']} but wasn't")
    return mismatches


def classify_rasa(blendshapes):
    """
    Tries every rasa against the same blendshape data. Returns
    (best_rasa_name, mismatches_for_best, all_mismatch_counts) -- the last
    one maps every rasa name to its mismatch count, so a caller can see how
    close the runner-ups were, not just which one "won."
    """
    scores = {name: len(check_rasa(blendshapes, name)) for name in RASA_DEFINITIONS}
    best = min(scores, key=scores.get)
    return best, check_rasa(blendshapes, best), scores
