# Kathak Analysis Dashboard

Upload a Kathak performance video, or perform live in front of a webcam, and get
a timestamped breakdown across five checks. See `project_description.md` for
the original scoping, `methods.md` for how each check actually works and what's
been validated, and `pipeline_flow.html` (open it directly in a browser) for a
diagram of how a video or live session actually flows through the pipeline,
including real examples pulled from an actual performance and a real live
session.

## What it checks

| Check | What it does |
|---|---|
| **Chakkar** | Counts spins from shoulder rotation and scores landing quality — count accuracy, ending orientation, controlled vs. abrupt stop. |
| **Mudra** | Identifies held hand gestures with a trained classifier, then checks the shape against reference rules. |
| **Rasa** *(unvalidated)* | Reads facial expression against the navarasa (nine classical emotions) on a fixed 1-second clock. |
| **Timing & Taal** | Checks wrist-movement peaks against the music's beat grid, and chakkar landings against the taal cycle's sam. |
| **Tatkaar** *(unvalidated)* | Detects rhythmic footwork as a sustained run of closely-spaced audio strikes — no camera view of the feet required. |

*Unvalidated = built from domain reasoning, not yet checked against real
labeled footage. See `methods.md` for specifics.*

## Comparison mode

Separate from the five checks above: upload a teacher's performance plus
your own take on the same piece, and the dashboard aligns the two (via
audio-onset DTW, since they won't be frame-synced) and returns an
action-by-action breakdown — which movement sounds (claps, stomps, bells)
matched, which didn't, and dense rhythmic stretches (tatkaar-style
passages) compared by strike count rather than matched bol-by-bol.

## Scores

The Overview tab's **Overall Score** is an equal-weighted average of
whichever category scores actually have data — chakkar quality, mudra
accuracy, and timing accuracy. Rasa and tatkaar are excluded from it
since they're unvalidated; a category with nothing to report (e.g. no
mudra holds detected) is left out of the average entirely, not counted
as a zero.

## Upload vs. live

- **Upload**: one finished video, analyzed once, top to bottom. Simple, but
  you only see results after the whole file is processed.
- **Live**: the browser records in ~1.5s chunks; each one is decoded,
  appended to a rolling buffer, and the full buffer is re-analyzed — so
  feedback updates as you dance, until you click Stop. Both paths run the
  exact same analysis code; nothing about the checks themselves changes
  between the two.

## Setup

```
pip install -r requirements.txt
```

Two one-time steps after cloning:

**1. Mudra model.** The trained classifier ships compressed to stay under
GitHub's file size limits. The app expects it at
`data/mudra_training/model.joblib`, so copy it into place:

```
cp data/mudra_training/model_compressed.joblib data/mudra_training/model.joblib
```

(Same model, same predictions — just a smaller file on disk. Without this
step the app still runs, but mudra identification returns nothing instead
of an error.)

**2. Bharatanatyam reference photos (optional).** Only needed if you want to
re-run mudra feature extraction from the original photos yourself — the
already-extracted training data (`data/mudra_training/*.csv`) is committed
directly and used by default, so this isn't required to run the app or
retrain the model.

```
git submodule update --init --recursive
```

## Running it

```
python app/server.py
```

Then open `http://localhost:5000`. Camera-based live capture works over
plain HTTP on `localhost` (Chrome treats it as a secure context); testing
from another device on the same network needs HTTPS instead.

## Tests

```
pytest
```
