# Codex Hardware and Sensor Review

## Environment and test baseline

Pytest and the complete project dependency set were installed in a Python 3.11 virtual environment. The system default Python was 3.6, which is too old for the project's current computer-vision stack.

The complete test suite passes:

```text
256 passed in 36.24s
```

Activate the environment with:

```bash
source .venv/bin/activate
python -m pytest -q
```

Because the workspace storage quota rejected the approximately 700 MB environment, `.venv` currently points to:

```text
/tmp/kathak-project-venv
```

This works in the current environment, but `/tmp` may be cleared after a reboot. A permanent setup should place the environment on storage with adequate quota or use a reproducible container.

---

## Primary sensor recommendation

For the science-fair project, prioritize a **wearable IMU for chakkar validation** over the haptic glove.

The strongest scientific question is:

> **How accurately can camera-based pose estimation measure Kathak chakkars compared with an independent wearable motion sensor?**

That creates two independent measurement systems:

```text
Camera -> MediaPipe -> shoulder angle -> estimated rotations
                                compared with
Waist IMU -> gyroscope -> angular velocity -> integrated rotations
```

This is scientifically stronger because the wearable sensor provides an independent reference for the camera result. It directly addresses the project's largest weakness: limited real-world ground truth.

### Suggested hardware

Use a small microcontroller and IMU worn on a waist belt or upper torso:

- ESP32 or Arduino Nano 33 BLE Sense.
- BNO085/BNO086 IMU, or an ICM-42688-class gyroscope.
- Small protected LiPo battery with a proper charger.
- Velcro waist pouch or elastic belt.
- Bluetooth or USB serial output.

The torso is a better location than the wrist because wrist rotations and arm gestures are not equivalent to whole-body chakkars.

### Data to collect

- Gyroscope Z-axis angular velocity.
- Integrated rotation count.
- Peak angular speed.
- Spin duration.
- Rotation direction.
- Stop/deceleration profile.
- Timestamped raw sensor samples.

### Experimental conditions

Compare the wearable and camera measurements across:

- Camera angle.
- Lighting.
- Clothing.
- Spin speed.
- Distance from camera.
- Multiple dancers.

This would produce meaningful comparison graphs and a defensible sensor-fusion experiment.

---

## Tatkaar sensor extension

The next-best sensor idea is an **ankle IMU**, ideally one on each ankle.

Ankle acceleration and impact peaks can provide candidate foot-strike timestamps. This would help distinguish actual foot movement from the current audio system's generic percussive onsets.

The experiment could compare:

```text
Audio onset detector
        versus
Left/right ankle IMU impacts
        versus
Human-annotated foot strikes
```

Measure:

- Strike-detection precision and recall.
- F1 score.
- Timing error in milliseconds.
- Left/right alternation.
- Inter-strike consistency.
- Performance with music and ghungroo.
- Performance without music.

An ankle IMU is safer and less intrusive than installing pressure sensors beneath the feet of a spinning dancer.

---

## Assessment of the haptic glove

The closed-loop concept is visually compelling:

```text
Camera -> hand tracking -> mudra check -> finger-specific vibration
```

It could become an excellent second-phase demonstration. It is not currently the strongest main experiment because mudra recognition is only approximately 50–60% accurate in the limited real-video evaluation, and the retrained model has not yet had that full per-window evaluation repeated.

If the input prediction is wrong, the glove may confidently buzz the wrong finger.

### Experimental circularity

There is also a circular evaluation risk:

1. The vision system decides which correction to provide.
2. The same vision system measures whether the correction succeeded.

The system could appear to improve its own score without the mudra becoming more correct according to a teacher.

For a proper haptic-learning study, use an independent outcome measure:

- Blinded Kathak teacher ratings.
- Manually annotated landmarks.
- A different held-out evaluation method.
- Ideally, a combination of expert ratings and geometric measurements.

Use a randomized crossover design so each participant tries both visual and haptic feedback, with different mudras and a counterbalanced order.

---

## Electrical corrections needed before building the glove

Do not build directly from the existing circuit description without revising it.

### 1. The 100-ohm transistor base resistor is too aggressive

With a 5 V Arduino output, a 100-ohm resistor could theoretically demand approximately:

```text
(5 V - 0.7 V) / 100 ohms = 43 mA
```

That is too high for a GPIO pin.

A properly calculated resistor and transistor—or preferably a suitable logic-level MOSFET motor driver—should be used.

A safer prototype architecture is:

- One suitable logic-level MOSFET per motor.
- Gate resistor.
- Gate pull-down resistor.
- Flyback diode across each ERM motor.
- Separate regulated motor supply.
- Common ground between the motor supply and microcontroller.

The final parts must be selected from the motors' rated voltage and stall current, not from generic assumptions.

### 2. Do not use a rectangular 9 V battery

Small rectangular 9 V batteries are poor at powering several vibration motors. Their internal resistance causes voltage sag, short runtime, and possible heating.

Use:

- A protected single-cell LiPo.
- A proper charger/protection board.
- A regulated output matching the motor voltage.
- Current capacity greater than the combined motor stall current.

Do not connect a LiPo directly without appropriate charge and protection electronics.

### 3. Verify each motor's rated voltage

Many coin vibration motors are rated at approximately 3 V. Driving one continuously at 5 V can shorten its life or cause overheating.

Select the supply voltage and PWM duty cycle from the actual motor datasheet.

### 4. Budget USB current carefully

Five motors may exceed what is safe or reliable to draw through a small development board or laptop USB port.

Power the motors separately through the motor drivers while sharing ground with the controller. Do not route all motor current through the microcontroller's regulator or GPIO pins.

### 5. Use appropriate flyback protection

Each ERM motor is an inductive load and needs flyback protection. A diode appropriate for the motor current and switching behavior should be used. A Schottky diode is commonly preferable for this type of low-voltage switching, subject to the selected motor and driver design.

### 6. Add a physical disable control

The wearable should include:

- A physical power switch or motor-disable switch.
- Firmware timeouts so a lost serial connection cannot leave a motor running continuously.
- Maximum pulse duration.
- Conservative vibration intensity.
- Insulated wiring and strain relief.

The device should fail silent, not fail with motors continuously energized.

---

## The glove may change the thing being measured

The glove, motors, wires, LEDs, and wrist electronics may:

- Alter finger posture.
- Add resistance or weight to movement.
- Cover hand contours or landmarks.
- Reduce MediaPipe tracking accuracy.
- Change hand appearance relative to classifier training images.
- Distract the dancer.
- Cause discomfort at the fingertips.

Before using it in an experiment, test mudra tracking and identification under at least these conditions:

1. Bare hand.
2. Plain glove with no electronics.
3. Complete glove with electronics turned off.
4. Complete glove with LEDs on.
5. Complete glove with haptics active.

This reveals whether the intervention hardware itself changes the measurement.

---

## A mismatch does not always identify one wrong finger

Some mudra constraints describe a relationship between two fingers:

- Thumb touching another finger.
- Spread between adjacent fingers.
- Distance between fingertips.
- Relative orientation of multiple fingers.

If thumb–index contact is missing, the software may detect the failed relationship without knowing whether the thumb, index finger, or both should move.

Therefore:

- Do not derive hardware commands by parsing English mismatch strings.
- Introduce structured correction objects in the software.
- Represent involved fingers separately from the recommended action.
- Allow feedback such as `thumb-index contact missing` instead of automatically declaring one finger wrong.
- Have a Kathak teacher review the correction mapping.
- Begin with mudras whose corrections are geometrically unambiguous.

An example structured correction could be:

```json
{
  "constraint": "thumb_index_contact",
  "status": "too_far_apart",
  "fingers": ["thumb", "index"],
  "feedback_target": ["thumb", "index"],
  "severity": 0.62
}
```

That is safer and more extensible than extracting finger names from a sentence.

---

## Recommended hardware roadmap

### Phase 1: strongest scientific contribution

Build a torso IMU logger for chakkar validation.

Deliverables:

- Timestamped sensor CSV.
- Camera/IMU synchronization.
- Gyroscope integration.
- Camera-versus-sensor comparison.
- Accuracy-by-condition plots.
- Repeatability and uncertainty estimates.

### Phase 2: validate tatkaar

Add one or two ankle IMUs.

Deliverables:

- Left/right strike candidates.
- Comparison with audio onsets.
- Human ground-truth annotations.
- Precision, recall, F1, and timing error.

### Phase 3: wow-factor feedback

Build a simplified haptic/LED glove only after the mudra correction signal is trustworthy.

Start with:

- One hand.
- Three carefully selected mudras.
- Three or five actuators.
- USB connection.
- No battery or Bluetooth initially.
- A physical emergency/off switch.
- Short vibration pulses instead of continuous buzzing.

### Phase 4: optional advanced work

Only after the basic experiment works:

- BLE wireless connection.
- Rechargeable wearable enclosure.
- Multiple haptic intensity patterns.
- KV260 benchmarking.
- Larger mudra set.
- Haptic-versus-visual learning study.

---

## Suggested chakkar sensor experiment

### Research question

> How closely does camera-based shoulder-angle rotation measurement agree with a torso-mounted gyroscope across different recording conditions?

### Hypothesis

> Camera and IMU rotation counts will agree under frontal, well-lit conditions, while camera error will increase with side angles, occlusion, loose clothing, and motion blur; IMU measurements will be less affected by visual conditions.

### Independent variables

- Camera angle.
- Lighting.
- Clothing.
- Spin speed.
- Camera distance.
- Dancer.

### Dependent variables

- Exact-count agreement.
- Absolute rotation-count difference.
- Spin start/end timing difference.
- Peak angular-speed difference.
- Direction agreement.
- Camera tracking coverage.

### Ground truth and synchronization

The IMU should not automatically be treated as perfect ground truth. Gyroscope bias and integration drift must be measured.

Use:

- A stationary calibration period before each trial.
- Known-turn bench tests or manually counted rotations.
- A visible and sensor-detectable synchronization event, such as a deliberate sharp movement at the beginning.
- Raw data retention so filtering choices can be audited.
- Human-reviewed video counts as an additional reference.

The strongest design compares all three:

```text
Human annotation <-> camera estimate <-> IMU estimate
```

---

## Final recommendation

Prioritize **sensor fusion first and haptic actuation second**.

A torso IMU makes the project's strongest existing component—chakkar measurement—more rigorous while also adding visible electronics, live plots, and an independent comparison signal. It strengthens both the science and the booth demonstration.

The haptic glove is more theatrical, but it currently depends on mudra identification and correction logic that is not reliable enough to serve as the project's principal experimental instrument. Once that signal has been validated and corrections are represented structurally, the glove can become an excellent closed-loop extension.
