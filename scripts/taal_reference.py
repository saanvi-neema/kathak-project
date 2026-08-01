"""
Taal reference data + cycle-aware beat math.

Real gap this fills: beat_sync_check.py's beat grid (from librosa.beat.
beat_track) is just a flat, evenly-spaced sequence of beat timestamps --
it has no concept of taal (the repeating rhythmic cycle), vibhag
(subdivisions within a cycle), or sam (the "one" the cycle resolves back
to). Without that structure, feedback can only ever say "off the beat
grid," never "your chakkar ended 1 beat before sam" -- which is what
actually matters for Kathak.

TAAL_DEFINITIONS' matra counts, vibhag divisions, and tali/khali positions
are sourced from Wikipedia's Vibhag article (cross-checked against several
Hindustani-music references, not transcribed from memory -- matra/vibhag
counts are exactly the kind of specific fact worth getting from a citable
source rather than guessing). Only taals actually relevant to Hindustani
classical music / Kathak are included, not every taal that exists.

Explicitly NOT built here (see methods.md): automatic detection of *where*
the first sam falls in a real recording, or of which taal is being used.
Both are genuinely hard audio-analysis problems (meter/downbeat tracking),
and there's no labeled real audio in this project to validate a detector
against -- building one now and not testing it would repeat the same
mistake made earlier with the mudra classifier. For now, both are meant to
be supplied (taal name + one known sam timestamp), not guessed.
"""

import numpy as np

# name -> {matras, vibhags (list of vibhag lengths, sums to matras),
# khali_vibhags (0-indexed vibhag positions that are khali/waved -- every
# other vibhag, including the first, is tali/clapped), description}.
TAAL_DEFINITIONS = {
    "teentaal": {
        "matras": 16, "vibhags": [4, 4, 4, 4], "khali_vibhags": [2],
        "description": "The most common taal in Kathak and Hindustani music generally. Tali on matras 1, 5, 13; khali on matra 9.",
    },
    "tilwada": {
        "matras": 16, "vibhags": [4, 4, 4, 4], "khali_vibhags": [2],
        "description": "Same matra/vibhag structure as Teentaal, different theka (played bol pattern).",
    },
    "ektaal": {
        "matras": 12, "vibhags": [2, 2, 2, 2, 2, 2], "khali_vibhags": [1, 3],
        "description": "Common in Kathak for slower (vilambit) compositions.",
    },
    "chautaal": {
        "matras": 12, "vibhags": [2, 2, 2, 2, 2, 2], "khali_vibhags": [1, 3],
        "description": "Same matra/vibhag structure as Ektaal; associated more with Dhrupad/pakhawaj than Kathak specifically.",
    },
    "jhaptaal": {
        "matras": 10, "vibhags": [2, 3, 2, 3], "khali_vibhags": [2],
        "description": "Common in Kathak; asymmetric 2+3+2+3 grouping.",
    },
    "sultaal": {
        "matras": 10, "vibhags": [2, 2, 2, 2, 2], "khali_vibhags": [1, 4],
        "description": "10 matras, symmetric 2x5 grouping (distinct from Jhaptaal's 2+3+2+3 despite the same matra count).",
    },
    "rupak": {
        "matras": 7, "vibhags": [3, 2, 2], "khali_vibhags": [0],
        "description": "Unusual: sam falls on the khali (waved) vibhag rather than a tali (clapped) one -- the only common taal that starts this way.",
    },
    "dadra": {
        "matras": 6, "vibhags": [3, 3], "khali_vibhags": [1],
        "description": "Light, often folk-influenced.",
    },
    "keherwa": {
        "matras": 8, "vibhags": [4, 4], "khali_vibhags": [1],
        "description": "Common in lighter compositions (thumri, etc.).",
    },
    "dhamar": {
        "matras": 14, "vibhags": [5, 2, 3, 4], "khali_vibhags": [2],
        "description": "Used for Dhamar/Hori compositions; asymmetric 5+2+3+4 grouping.",
    },
    "jhoomra": {
        "matras": 14, "vibhags": [3, 4, 3, 4], "khali_vibhags": [2],
        "description": "Slow tempo, 3+4+3+4 grouping.",
    },
    "deepchandi": {
        "matras": 14, "vibhags": [3, 4, 3, 4], "khali_vibhags": [2],
        "description": "Same matra/vibhag structure as Jhoomra; used for thumri. Also spelled Dipchandi.",
    },
}


def _vibhag_for_matra(taal, matra_in_cycle):
    """matra_in_cycle is 0-indexed (0 = sam). Returns (vibhag_index, is_khali)."""
    pos = 0
    for i, length in enumerate(taal["vibhags"]):
        if pos <= matra_in_cycle < pos + length:
            return i, i in taal.get("khali_vibhags", [])
        pos += length
    raise ValueError(f"matra_in_cycle={matra_in_cycle} out of range for a {taal['matras']}-matra taal")


def _avg_matra_interval(beat_times):
    beat_times = np.asarray(sorted(beat_times), dtype=float)
    if len(beat_times) < 2:
        return None
    return float(np.median(np.diff(beat_times)))


def build_cycle_map(beat_times, taal_name, sam_time):
    """
    Aligns a flat beat grid (e.g. from beat_sync_check.py) to a taal cycle,
    anchored at one known sam timestamp. Assumes the beat grid is already
    roughly evenly spaced (true for a stable taal performance -- laya/tempo
    changes mid-performance aren't handled here).

    Returns a list of dicts, one per input beat: {time, cycle, matra
    (1-indexed within its cycle), vibhag_index, is_khali}.
    """
    if taal_name not in TAAL_DEFINITIONS:
        raise ValueError(f"Unknown taal '{taal_name}'. Known: {sorted(TAAL_DEFINITIONS)}")
    taal = TAAL_DEFINITIONS[taal_name]
    matras = taal["matras"]

    beat_times = np.asarray(sorted(beat_times), dtype=float)
    avg_interval = _avg_matra_interval(beat_times)
    if avg_interval is None or avg_interval <= 0:
        return []

    cycle_map = []
    for t in beat_times:
        matra_offset = int(round((t - sam_time) / avg_interval))
        matra_in_cycle = matra_offset % matras
        cycle_number = matra_offset // matras
        vibhag_index, is_khali = _vibhag_for_matra(taal, matra_in_cycle)
        cycle_map.append({
            "time": float(t), "cycle": int(cycle_number), "matra": int(matra_in_cycle) + 1,
            "vibhag_index": vibhag_index, "is_khali": is_khali,
        })
    return cycle_map


def beats_from_sam(query_time, beat_times, taal_name, sam_time):
    """
    Signed distance, in matras (beats), from query_time to the nearest sam
    -- the actual number needed for "your chakkar ended 1.3 beats off sam."
    Positive = after the nearest sam, negative = before it. Returns None if
    the beat grid doesn't have enough beats to establish a tempo.
    """
    if taal_name not in TAAL_DEFINITIONS:
        raise ValueError(f"Unknown taal '{taal_name}'. Known: {sorted(TAAL_DEFINITIONS)}")
    matras = TAAL_DEFINITIONS[taal_name]["matras"]

    avg_interval = _avg_matra_interval(beat_times)
    if avg_interval is None or avg_interval <= 0:
        return None

    matra_offset = (query_time - sam_time) / avg_interval
    nearest_sam_offset = round(matra_offset / matras) * matras
    return matra_offset - nearest_sam_offset
