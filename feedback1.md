# Review: Kathak Engineering Project Submission

## Issues to Fix

1. **"readable haptic signals"** in the Goal — should be "wearable haptic signals." Haptic signals are felt, not read.

2. **Latency criterion: "under 1 second"** — current implementation takes 7–9 seconds per 1.5s chunk. This criterion will fail the stated test plan. Either adjust to reflect reality (e.g., "under 10 seconds") or note it as a target pending browser-side MediaPipe. Examiners will ask why 1 second was written if the test shows 8.

3. **Mudra accuracy criterion: "at least 80%"** — the current classifier always predicts "pataka" regardless of the actual mudra and needs retraining before this criterion can be tested honestly. Either note it as a target, or frame it as "upon retraining on Kathak-specific footage."

4. **Chakkar criterion: "within 0.15 rotations"** — validated on only 3 clips. Fine to state, but be prepared for the examiner to ask how 0.15 was arrived at specifically.

## What Is Working Well

- Goal framing is exactly right — "supplements," "disciple's independent practice," no overreach
- Software components section is detailed, accurate, and matches the actual implementation
- Libraries list is complete and correct
- Test plan is specific and measurable — each criterion has a concrete test method
- Bibliography is strong, especially the **Abhinaya Darpana** reference — citing the classical Sanskrit source on mudras shows genuine domain depth and will stand out to any examiner who notices it
