# Methods

This project has multiple separate pilot experiments (see project_description.md for the full list). Each section below documents what was actually done for one experiment, in plain language, so it's clear later what was tested and how.

## Chakkar Pilot Test (Experiment 2 — can we count spins?)

1. Got the actual video files onto disk. You'd said "downloaded videos" but that's a browser action — nothing lands in a project folder automatically. I found them in your Downloads folder and copied them into dance-project/data/raw/ so there'd be stable file paths to work with.

2. Looked at the raw video first, before running any algorithm. I pulled a few still frames out of each clip (first/middle/last) and looked at them directly — checking framing (is the whole body visible? is the camera angle sane? is it too dark/blurry?). This matters because if the video itself is bad (e.g., feet cut off, motion blur), any failure later could be a recording problem, not an algorithm problem. Found: camera was angled up and cropped your feet out — fine for spin-counting (which only needs shoulders), but would sink a footwork test later.

3. Ran a pose-detection model on every single frame of both videos. This is MediaPipe Pose — it's a pretrained model that looks at each video frame and tries to output the pixel location of ~33 body joints (shoulders, hips, knees, etc.), each with a confidence/"visibility" score. This is the same kind of model used for things like Snapchat filters or fitness apps — it doesn't know anything about Kathak, it just finds body joints.

4. Test A — "does tracking even survive the movement?" For every frame, I checked: did the model find a person at all, and specifically did it find both shoulders with high confidence? I counted what fraction of frames passed. Result: 100% of frames in both clips had a person detected, and both shoulders visible. That's the first pass/fail gate — if this were low (say 60%), there'd be no point trying to count anything, because the raw tracking would already be unreliable.

5. Test B — "can we count the spins from that tracking data?" For each frame, I drew a line between your left and right shoulder and computed the angle that line makes (basic trig — atan2, the same math as finding the angle of a slope). As you spin, that angle changes frame by frame. Problem: raw angle math wraps around at 180°/-180° (like a compass, going from 179° to -179° is actually just a 2° step, not a 358° jump) — so I "unwrapped" it, meaning I let the angle keep climbing past 180°, 270°, 360°, etc. instead of snapping back. Then: total degrees rotated ÷ 360 = number of full spins.

6. Checked the algorithm's count against what you actually did. You told me video 1 = 1 spin, video 2 = 3 spins — that's the ground truth, the "correct answer" a human knows and the algorithm doesn't. Algorithm output: video 1 ≈ 0.99 spins (rounds to 1), video 2 ≈ 3.00 spins (rounds to 3). Both matched exactly.

7. Tested a longer clip to check for "drift." Two clips (1 spin, 3 spins) is a very small test — if the angle math has a small per-frame error, it might not show up over 1-3 spins but could snowball into a visibly wrong count over more rotations. So you recorded a third clip with 10 continuous fast spins, and I ran the same process on it. Result: 10.02 estimated vs. 10 true — still rounds to exactly 10, and critically, the error didn't grow with more rotations. I also looked directly at the plotted angle data (a chart of the raw angle and the "unwrapped" angle over time) to confirm the climb was smooth and continuous the whole way through, not just a lucky final number.

## Movement Pilot Test (Experiment 1 — does pose tracking survive ordinary movement?)

This is a different question from the chakkar test. Chakkars are the hardest case (fast spinning). This experiment asks a more basic question first: does pose tracking even work reasonably well on *normal* Kathak-ish movement — arm gestures, walking, posture — not just spins? And does it hold up the same way across two different, independently-built tracking models, or does it just look good because of a quirk in one specific model?

1. Got the video onto disk, same as before — found it in Downloads, copied it into dance-project/data/raw/movement/.

2. Ran two different pose-tracking models on every frame, side by side, instead of just one:
   - **MediaPipe Pose** — the same model used for the chakkar test. Finds ~33 body points, including some face detail.
   - **MoveNet** (specifically the "Thunder" version, from Google) — a different, independently-trained model. Finds 17 body points (no face detail, just the main joints: shoulders, elbows, wrists, hips, knees, ankles).

   The reason to run two models instead of one: MediaPipe and MoveNet were built by different teams using different training data and different internal designs. If a movement trips up *one* model, that could just be a quirk of that specific model. If a movement trips up *both* models, that's much stronger evidence the movement itself is genuinely hard to track (motion blur, limbs overlapping, etc. — not just one model's weakness). Agreement between two independent models is a much better signal than one model alone.

3. For each frame, recorded whether each model found a person, and how confident it was. MediaPipe reports a 0-1 "visibility" score per joint (I averaged across joints for a per-frame number). MoveNet reports its own 0-1 "score" per joint the same way. These two confidence numbers aren't on directly comparable scales (different models calibrate confidence differently), so they're read separately, not compared like-for-like.

4. Drew the skeleton (all the tracked joints, connected with lines) directly on top of the video for both models, and saved those as separate "overlay" videos — this is the same idea as the guide-lines apps like Snapchat draw on your face, just for the whole body. Having the skeleton drawn on the actual footage is what makes it possible to visually check "does this look right?" rather than just trusting a number.

5. Since manually watching every single frame of every video isn't practical, saved one still image every 15 frames (about 2 per second), with both models' overlays side by side in the same image, into a `review_stills` folder. This gives a manageable, evenly-spread sample to look through by eye.

6. Manually looked at a handful of those saved stills, spread across the whole clip, and rated each one:
   - **Good** — skeleton position looks correct.
   - **Usable** — small error, but you could still measure things from it.
   - **Failed** — wrong limb, lost the person, or a major jump/glitch.

   Checked: a plain standing pose, a crossed-arm gesture, a side-profile pose with a raised hand, and — deliberately picked because it looked like the blurriest one — a fast arm-raise with visible motion blur. All four were rated Good on both models. Even the blurry one tracked correctly, and MediaPipe's low-confidence (red) markers correctly lined up with the genuinely ambiguous points (like a hand foreshortened toward the camera) rather than being wrong everywhere.

7. Result so far: both models detected a person on 100% of frames in this clip, and the small manual spot-check found no failures. This is an early good sign for the "does tracking survive ordinary movement" question, but it's not the full test yet — only about 4 stills out of 52 saved ones have actually been eyeballed so far (the original plan calls for checking hundreds), and this was one mixed clip (arm movements + some walking together) rather than the separate clean clips (stance / arms / torso / walking / combined) the plan called for. No "harder" conditions (looser clothing, different camera angle, busier background) have been tried yet either.
