# Project Description (temp notes, not a polished README)

## Full planned project: Pose-Based Computational Analysis of Kathak

7-phase pipeline, long-term vision — this doc is the full picture for reference so later phases don't get designed from scratch. **The feasibility pilot below was never formally completed** (its own go/no-go criteria called for a 500-1000 frame manual QA pass, a 10-clip + harder-conditions dataset, and a tatkaar audio-onset experiment — none of that full scope happened) **but its early small-scale results were promising enough that development moved directly into building the real pipeline anyway**, described in methods.md's "Vertical Stack" section. That section is the up-to-date source of truth for what's built vs. not; this doc's pilot section below is kept as a historical record of why that jump felt justified, not a live status report or a claim that a formal go decision was made.

**Deliverable format: decided.** It's a usable app — upload a video, get back a dashboard with a detailed, timestamped critique (see methods.md's "Vertical Stack" intro for the exact target example). Settled after the pilot succeeded; a dashboard mockup is the target reference for the app's shape.

### Phase 1: Pose and Landmark Extraction
Goal: convert Kathak videos into structured motion data.

- Body pose: MediaPipe Pose (chosen — passed pilot; MoveNet and YOLO-Pose dropped, see pilot notes below)
- Hand landmarks: MediaPipe Hands (21 landmarks/hand: wrist, thumb/index/middle/ring/pinky joints)
- Facial landmarks/expression: done — MediaPipe FaceLandmarker with blendshapes, added for navarasa/abhinaya analysis (see methods.md step 9). Not head-orientation tracking specifically (that's still hypothetical), but the expression half of this line is built.

Input: Kathak performance video.
Output: time-series of body keypoints `(x, y)` per joint + hand landmarks per frame.

### Phase 2: Feature Extraction

**2A. Body features**
- Joint angles (elbow, shoulder, knee, hip) via vector geometry
- Posture: torso tilt, head alignment, body symmetry, center of mass
- Motion: joint velocity/acceleration, trajectory smoothness, movement range/consistency

**2B. Hand features (mudra analysis)**
- Finger joint angles, extension, thumb position, relative finger distances, hand orientation
- Possible tasks: mudra classification (**done** — see methods.md step 4), hand trajectory analysis, transition detection, frequency stats

**2C. Footwork features (tatkaar analysis)**
- From video: foot position/velocity/trajectory, left/right coordination
- From audio: foot strike timestamps, strike intensity, rhythm consistency, beat spacing, footwork clarity

### Phase 3: Chakkar Detection and Analysis
Goal: automatically identify and analyze spins.

- Body orientation from shoulder positions: `orientation = atan2(shoulder_y_diff, shoulder_x_diff)`
- Track orientation through time; detect complete/partial rotations, spin direction
- Metrics per chakkar: duration, angular velocity/acceleration, consistency, recovery stability
- Output: total spin count, avg spin speed, fastest spin, spin variability/consistency

### Phase 4: Movement Metric Development
Interpretable computational measures: stability, symmetry, smoothness, timing consistency, chakkar consistency, postural control, mudra consistency, footwork consistency. Output is a collection of quantitative metrics per performance (not a single score).

### Phase 5: Comparative Analysis
Collect videos across beginner/intermediate/advanced dancers; compare posture/smoothness/spin/mudra/footwork/timing metrics. Research questions: which metrics vary most with skill level, best distinguish experienced dancers, are most predictive of proficiency, show greatest variability.

### Phase 6: Machine Learning Modeling
Inputs: all features from prior phases. Models: Random Forest, XGBoost, SVM, Neural Network. Tasks: skill classification, feature importance, clustering of movement styles.

### Phase 7: Hardware Acceleration and Benchmarking
Pipeline: Video → Pose+Hand Extraction → Feature Extraction → Movement Analysis → ML.
Compare CPU / GPU / FPGA (KV260 DPU) on FPS, latency, throughput, power, energy efficiency.

---

## Historical: the feasibility pilot (superseded — kept as a record of why the project moved forward)

Originally decided (based on a prior ChatGPT conversation, full advice captured below) not to build the full 7-phase system until a narrow go/no-go pilot passed, since body pose, hand tracking, chakkar counting, and tatkaar detection are independent technical risks that could each fail without sinking the others. **The pilot itself was never finished to its own criteria** (see Experiment 1/2/3 progress below — no formal go decision was made), but Experiments 1 and 2's early results (100% tracking detection, exact chakkar counts on 3 test clips) were promising enough that work moved directly into the real pipeline instead of waiting on the full planned dataset. That means the original "do not build yet" list below is now stale in places — mudra classification and overall dance scores are both built (methods.md step 4 and the Overview section) despite never being pilot-gated. Skill-level prediction, beginner/intermediate/advanced datasets, the full 15-movement-metric suite, ML modeling (Phase 6), and FPGA work (Phase 7) genuinely still haven't been started — methods.md's "Vertical Stack" section is the authoritative list of what's built vs. pending day to day.

**Location**: `C:\Users\Saanvi\comp-sci\dance-project` (moved from an old-computer backup dir; now a git repo, pushed to `https://github.com/saanvi-neema/kathak-project.git`).

**Core question**: Can ordinary video contain enough reliable signal to measure a few Kathak techniques accurately enough to justify the full project?

**Two-stage validation concept (for later, not this pilot)**: Technical validation ("does the algorithm measure what physically happened?") is a completely different question from construct validation ("does the measurement correspond to meaningful Kathak technique, per knowledgeable dancers?"). This pilot only tests the first. A 100%-accurate chakkar counter still says nothing about whether "Chakkar Consistency: 93" means anything to an actual Kathak teacher — that's future work.

**Planned dataset** (~12-20 min total, one dancer — ideally the user — controlled conditions, no skill-level comparison yet):

| Test | Clips | Length |
|---|---|---|
| Basic body movements | 5 | 15-30s |
| Chakkars | 10 | 10-20s |
| Tatkaar | 10 | 15-30s |

First recordings: fixed camera, full body visible, decent lighting, plain background, 60fps if possible, clear audio. Then deliberately record harder versions: loose clothing, faster spins, different camera angle, normal room background — gives an easy test and a realism test.

**Three pilot experiments:**

1. **Pose tracking survival** — MediaPipe Pose vs MoveNet only (not three models). Draw skeleton overlays, manually inspect ~500-1000 sampled frames, label each Good / Usable / Failed, log confidence scores. Output a table of good/usable-frame % and major-failure count per movement type (basic stance, arm movement, fast tatkaar, chakkar, etc). **Decision rule**: define the threshold before looking at results — proceeding needs ≥90% usable frames on non-spin body movement. If tracking is good on normal movement but poor during chakkars, project isn't dead, chakkars just need a different detection method. If tracking collapses on almost all realistic movement, the pose-based plan needs a major redesign. *Left incomplete at a small-sample stage, see progress below — not resumed since, work moved to the real pipeline instead.*

2. **Chakkar counting** — the highest-priority experiment; likely flaw in the naive plan. `atan2(shoulder_y_diff, shoulder_x_diff)` is not guaranteed to give a clean monotonic 0°→360° signal from a frontal 2D video — side views can cause shoulder overlap, and front/back can look ambiguous, because a monocular camera is projecting a 3D rotation onto 2D (known limitation: occlusion + no depth). So test multiple candidate signals, not just shoulder-line geometry: left/right shoulder ordering, hip geometry, nose position relative to shoulders, ear visibility/confidence, periodicity of pose features. First goal is counting only — no angular acceleration/recovery-stability/spin-quality metrics yet. **Go/no-go**: ≥90% exact-count accuracy on controlled clips, ≥80% on harder clips. If the simple shoulder method fails but a signal combination works, that's still a good result. *Left incomplete at a small-sample stage (n=3 clips), see progress below — real chakkar work then continued directly in methods.md step 3 rather than expanding this pilot dataset.*

3. **Tatkaar audio onset detection** — fully separate from foot pose tracking for this pilot. Manually annotate exact strike timestamps in an audio editor as ground truth (e.g. 0.52s, 0.91s, 1.31s...). Pipeline: audio → onset strength → peak detection → predicted timestamps. Match predicted-to-real within ±50ms. Measure precision, recall, F1, mean timing error. Test slow/medium/fast tatkaar, with and without music. **Go/no-go**: F1 ≥ 0.90 without music, F1 ≥ 0.80 with realistic accompaniment. If audio works but visual foot tracking doesn't, the project can still succeed as a multimodal system — possibly stronger than forcing pose estimation to do everything. *Not started.*

**Most valuable possible outcomes** (pilot is useful even if parts fail): pose might work for posture but not chakkars; visual tracking might work for chakkars but not feet; audio might work great for tatkaar even if video doesn't; hand tracking might only work in close-up video; controlled recordings might work while YouTube performance videos don't. Any of these tells you what the real research project should actually be.

**Experiment 1 progress (pose tracking survival, non-spin movement):**
- `scripts/movement_pilot.py`: runs MediaPipe Pose (Tasks API) and MoveNet Thunder (TFLite, `models/movenet_thunder.tflite`, downloaded from tfhub.dev) side by side per frame, writes an overlay video per model, and samples a side-by-side still every 15 frames into `review_stills/` for manual Good/Usable/Failed QA. Added `tensorflow` to `requirements.txt` for `tf.lite.Interpreter`.
- First clip: `data/raw/movement/movement_01.mov` (arm movements + some walking, one continuous take, ~48MB, recorded 2026-07-11) — not yet split into the 5 separate movement-type clips from the original plan (stance / arms / torso / walking / combined); this one clip mixes several.
- **Frame-level detection stats:** both models detected a person on 100% of frames. MediaPipe mean visibility confidence 0.89; MoveNet mean keypoint-score confidence 0.60 (lower headline number, but still well above the 0.3 threshold used for "detected").
- **Manual visual QA (4 sampled frames spread across the clip):** static arms-out stance, crossed-arm gesture, side-profile raised-hand pose, and a fast/motion-blurred arm-raise — all four rated **Good** on both models. Skeletons stayed accurate through the blurry fast-arm frame; MediaPipe appropriately flagged genuinely ambiguous points (e.g. near-hand foreshortening) as low-confidence (red) rather than confidently wrong. No failures spotted yet in this small manual sample.
- Not yet done: the fuller 500-1000 sampled-frame labeling pass the original plan calls for (only ~4 of 52 sampled stills manually checked so far); the walking-specific segment hasn't been specifically isolated and checked; no "harder" conditions (loose clothing, off-angle camera, busier background) tested yet.

**Experiment 2 progress (chakkar counting):**
- `scripts/chakkar_pilot.py`: runs MediaPipe Pose per-frame using the Tasks API (mediapipe 0.10.35 on Windows dropped the old `mp.solutions` API; needs a downloaded model bundle at `models/pose_landmarker_full.task`). Tracks shoulder-line angle via `atan2`, unwraps across frames, estimates rotation count as total unwrapped degrees ÷ 360. Reports both the raw (unrounded) estimate and a rounded count, so drift is visible even without fixing the half-rotation rounding issue.
- Ground truth clips in `data/raw/`: `chakkar_01.mov` (1 slow spin), `chakkar_02.mov` (3 fast spins), `chakkar_03.mov` (10 continuous fast spins, recorded 2026-07-11).
- **Results (all 3 clips, run 2026-07-11):**

  | clip | true | raw_est | rounded | error | pose% | shoulders% |
  |---|---|---|---|---|---|---|
  | chakkar_01 | 1 | 0.99 | 1 | 0 | 100% | 100% |
  | chakkar_02 | 3 | 3.00 | 3 | 0 | 100% | 100% |
  | chakkar_03 | 10 | 10.02 | 10 | 0 | 100% | 100% |

  Pose + both-shoulder detection held at 100% on all three clips, including the longer/faster 10-spin one. Critically, the 10-spin clip shows **no meaningful drift**: raw estimate 10.02 vs true 10, essentially the same relative error as the 1- and 3-spin clips. Visual inspection of `chakkar_03_angle_plot.png` confirms the unwrapped angle climbs monotonically through all ~10 rotations with no reversals/discontinuities, and the wrapped angle shows a clean repeating sawtooth — i.e., this isn't a lucky rounding coincidence, the underlying signal is genuinely clean.
  - This directly answers the concern ChatGPT raised about monocular 2D shoulder-angle ambiguity (side-view overlap, front/back confusion) — at least for this dancer/framing/spin style, the naive `atan2` signal is not exhibiting that failure mode.
- Still n=3 clips total, all one dancer/one setup — nowhere near the 10-clip controlled + harder-clip dataset in the plan, and no "harder" conditions (loose clothing, off-angle camera, faster spins, plain-vs-normal background) tested yet.
- `methods.md` (project root) documents the testing methodology in plain language.

**Next steps (as originally planned for the pilot itself — mostly bypassed in practice, see below):**
- Expand toward the full planned chakkar dataset (10 controlled clips + harder variants: loose clothing, faster spins, different camera angle, normal background) to actually test the 90%/80% go/no-go threshold rather than 3 hand-picked clips. **Never done** — chakkar work instead moved straight into real-footage multi-event segmentation and scoring (methods.md step 3), tested against an actual excerpted performance clip rather than more controlled single-spin clips.
- ~~Fix rounding: the script currently rounds to nearest whole spin...~~ **Done, but as part of the real pipeline, not the pilot script**: `chakkar_scoring.py` rounds to the nearest clean landing and reports the gap (methods.md step 3), superseding this note.
- Experiment 3 (tatkaar audio-onset detection) hasn't been started — still need the 10 tatkaar clips per the dataset plan. Still true; tatkaar remains the one pipeline step (methods.md step 7) blocked purely on missing footage.
- Experiment 1: keep recording more movement clips (ideally the separate stance/arms/torso/walking/combined clips per the plan, plus harder variants) while doing a fuller manual QA pass on the frames already captured. Still true — the fuller QA pass never happened; `movement_01.mov` ended up reused directly as real-footage validation data for chakkar segmentation and drift instead (methods.md step 3).
