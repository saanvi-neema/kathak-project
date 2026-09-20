# Hardware Glove — Haptic & LED Deep Dive

---

## Haptic Feedback Glove

### How it works end-to-end

```
Webcam → MediaPipe (hand landmarks) → mudra_reference.py (which fingers are wrong)
       → Python serial message → Arduino → vibration motor on that finger buzzes
```

The existing pipeline already knows *which specific finger* is violating a mudra rule. You just need to route that signal to hardware.

---

### Components

| Part | Purpose | Cost |
|------|---------|------|
| Arduino Nano | Receives signals from Python, drives motors | ~$5 |
| 5x vibration disc motors (coin type) | One per fingertip | ~$5 total |
| 5x NPN transistors (2N2222) | Arduino GPIO can't drive motors directly | ~$1 |
| 5x 1N4007 diodes | Protect against motor back-EMF | ~$1 |
| 5x 100Ω resistors | Base resistors for transistors | <$1 |
| Thin flexible wire | Runs along back of glove | ~$2 |
| A cheap cotton glove | Base to sew everything onto | ~$2 |
| USB cable | Python ↔ Arduino serial link | already have one |

**Total: ~$15-20**

---

### Wiring concept

```
Arduino GPIO pin → 100Ω resistor → transistor base
                                    transistor collector → motor (+)
                                    transistor emitter  → GND
                   diode across motor terminals (flyback protection)
```

One GPIO pin per finger (pins D3–D7). Python sends a byte over serial like `"10100\n"` meaning "buzz index and ring fingers." Arduino parses it and toggles the right pins.

The motors sit in small fabric pockets sewn at each fingertip. Wires run along the back of the hand to a small velcro pouch at the wrist holding the Arduino + a 9V battery.

---

### Python side (very small addition)

```python
import serial

port = serial.Serial('COM3', 9600)  # whichever port Arduino is on

def send_haptic_feedback(finger_flags):
    # finger_flags: dict like {'index': True, 'middle': False, ...}
    msg = ''.join(['1' if finger_flags[f] else '0'
                   for f in ['thumb','index','middle','ring','pinky']])
    port.write((msg + '\n').encode())
```

`mudra_reference.py` already returns which geometric constraints failed — each constraint maps to a finger. You extract the finger names and call `send_haptic_feedback()`. Probably 20-30 lines of new code total.

---

### Arduino side (~15 lines)

```cpp
const int pins[] = {3, 4, 5, 6, 7};  // thumb, index, middle, ring, pinky

void loop() {
  if (Serial.available() >= 5) {
    for (int i = 0; i < 5; i++) {
      char c = Serial.read();
      digitalWrite(pins[i], c == '1' ? HIGH : LOW);
    }
    while (Serial.available()) Serial.read();  // flush newline
  }
}
```

---

### What the dancer feels

- Hold up hand in front of webcam
- Attempt Pataka mudra (all fingers extended, together)
- If ring finger is slightly curled → ring fingertip buzzes
- Straighten ring finger → buzz stops
- All fingers correct → all quiet

Immediate, eyes-free feedback. The dancer doesn't need to look at a screen at all.

---

---

## LED Correction Glove

### How it works

Same pipeline, same Arduino, same serial protocol. Swap vibration motors for RGB LEDs. Each fingertip glows:
- **Green** = finger correct
- **Red** = finger wrong
- **Off** = finger not relevant to current mudra

---

### Components

| Part | Purpose | Cost |
|------|---------|------|
| Arduino Nano | Same as above | ~$5 |
| 5x RGB LEDs (common cathode) | One per fingertip | ~$3 |
| 15x 220Ω resistors | One per R/G/B channel | ~$1 |
| Thin wire + glove | Same as above | ~$4 |

**Total: ~$13-15**

RGB LEDs need 3 pins each (R, G, B) = 15 GPIO pins total. Arduino Nano has exactly enough. Alternatively use NeoPixels (WS2812B) — each addressable over a single data line, much cleaner wiring, slightly more expensive (~$6 for a strip of 5).

---

### NeoPixel version (cleaner)

```python
# Python sends finger states as hex color codes
# "FF000000FF00..." = red, green, ... per finger
```

```cpp
#include <Adafruit_NeoPixel.h>
Adafruit_NeoPixel strip(5, 6, NEO_GRB);  // 5 LEDs on pin 6

// Read 5 color values from serial, set each LED
```

Single wire from Arduino to all 5 LEDs in a chain. Much simpler physically.

---

## Haptic vs LED — honest comparison

| | Haptic | LED |
|--|--------|-----|
| **Learning value** | Higher — no visual distraction | Lower — dancer looks at hand |
| **Booth wow factor** | Medium (invisible, has to be explained) | High (instantly visible across room) |
| **Build difficulty** | Slightly harder (transistor circuit) | Easier (direct from Arduino) |
| **Wearability** | Comfortable, invisible under performance | Visible, colorful |
| **Science experiment** | Better — haptic vs. visual is a known research question | Weaker — just another visual |
| **Cost** | ~$20 | ~$15 |

---

## Recommendation: Build both on the same glove

Put NeoPixel LEDs on the fingernails (visible) and vibration motors in the fingertip pads (felt). Same Arduino, same serial signal. Switch between modes:
- **Demo mode** at the booth → LEDs on, instantly visible across the room
- **Learning mode** for the experiment → LEDs off, haptic only

### The science fair question becomes:

> **"Does embodied haptic feedback teach mudra hand position faster than visual feedback — and can a computer measure the difference?"**

That's genuinely original, testable, and publishable at a high school level.

---

---

## Complete System Architecture

### Full pipeline — how everything connects

```
┌─────────────────────────────────────────────────────┐
│                    LAPTOP / PC                       │
│                                                     │
│  Webcam feed                                        │
│      ↓                                              │
│  MediaPipe  →  mudra_reference.py                   │
│                      ↓                              │
│           "ring finger wrong, index wrong"          │
│                      ↓                              │
│           send_haptic_feedback()                    │
│                      ↓                              │
│           Serial port  (pyserial)                   │
│                                                     │
└──────────────────────┬──────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │   Option A: USB cable  │   Option B: Bluetooth
           │   (simplest)          │   (wireless)
           └───────────┬───────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              ARDUINO NANO                            │
│         (velcro pouch on wrist)                     │
│                                                     │
│   Receives "10100" → buzz thumb + middle            │
│                                                     │
│   D3 ──→ motor (thumb)                              │
│   D4 ──→ motor (index)                              │
│   D5 ──→ motor (middle)                             │
│   D6 ──→ motor (ring)                               │
│   D7 ──→ motor (pinky)                              │
│                                                     │
└──────────────────────────────────────────────────────┘
                       │
              thin wires along
              back of hand
                       │
        ┌──────────────┼──────────────┐
        ↓      ↓       ↓      ↓       ↓
      thumb  index   middle  ring   pinky
      motor  motor   motor  motor   motor
```

---

### Connection options

#### Option A — USB cable (use this for the science fair)

- Standard USB-A to USB-Mini/Micro from laptop to Arduino on dancer's wrist
- The dancer stands 1-2 feet from the webcam anyway so it can see their hand
- Cable just drapes to the floor — fine for a booth demo
- Dead simple, zero configuration, works immediately
- Python talks to it as `COM3` (Windows) or `/dev/ttyUSB0` (Linux/Mac)

#### Option B — Bluetooth HC-05 module (~$4 extra)

```
Laptop (Python)  →  Bluetooth  →  HC-05 module  →  Arduino Nano
```

- Plug an HC-05 Bluetooth module into the Arduino's RX/TX pins
- Python uses `pyserial` exactly the same way — Bluetooth creates a virtual COM port
- Dancer is completely wireless, can move freely
- Takes ~30 mins extra to configure the HC-05 baud rate via AT commands
- Better for a real performance demo; unnecessary for a science fair booth

**For the science fair: USB cable is fine.** The dancer stands still in front of the webcam. Add Bluetooth later if you want to demo it during movement.

---

### Physical layout on the dancer

```
Fingertip motors  ←── thin wires along back of hand
                                    ↓
                        small velcro wrist pouch
                        containing:
                          - Arduino Nano
                          - 9V battery (if wireless/Bluetooth)
                          - OR just USB cable out (if wired)
```

The Nano is about the size of a USB stick. The wrist pouch is barely noticeable. Total weight under 30g.

---

### Latency — does it feel instant?

| Step | Time |
|------|------|
| MediaPipe processes a frame | ~33ms (at 30fps) |
| Python detects wrong finger | ~1ms |
| Serial send to Arduino | ~5ms |
| Motor spins up | ~10ms |
| **Total** | **~50ms** |

50ms is imperceptible as a delay. The buzz feels immediate to the dancer.

---

### Off-the-shelf base glove

Thin nylon mobile gaming gloves (available on Amazon/Walmart, ~$5-8) make a good base:
- Lightweight, breathable, fingers stay separated
- Thin enough that coin motors sit flush at the fingertips
- Easy to sew or hot-glue components onto
- They have **no haptic capability on their own** — they are just the substrate

**Full bill of materials using a gaming glove as base:**

| Part | Cost |
|------|------|
| Nylon gaming glove (2 pack) | ~$8 |
| Arduino Nano | ~$5 |
| 5x coin vibration motors (Adafruit, pre-wired) | ~$10 |
| 5x NPN transistors + diodes + resistors | ~$3 |
| HC-05 Bluetooth module (optional) | ~$4 |
| Velcro wrist pouch | ~$2 |
| **Total (wired)** | **~$28** |
| **Total (wireless)** | **~$32** |

---

---

## KV260 as the Compute Platform + LiPo Power

---

### KV260 for the laptop/PC role

The KV260 was already in the project's Phase 7 roadmap (FPGA benchmarking) — so this is a natural fit, not a detour.

#### What's in the KV260

- **Quad-core ARM Cortex-A53 @ 1.333 GHz** — runs Linux, Python, OpenCV, Flask, all fine
- **4GB LPDDR4 RAM** — plenty
- **FPGA fabric (Zynq UltraScale+)** — programmable logic, the key differentiator
- **DPU (Deep Learning Processing Unit)** — AMD's on-chip neural network accelerator, accessed via Vitis AI

---

#### The bottleneck: MediaPipe

Everything in the pipeline is light except MediaPipe. The underlying models (BlazePose for body, Palm + Hand Landmark for hands) are TFLite models running inference per frame.

**ARM-only (A53 cores, no FPGA):**

| Board | CPU | MediaPipe pose+hand fps |
|-------|-----|------------------------|
| Raspberry Pi 4 | Cortex-A72 @ 1.8GHz | ~10-15 fps |
| KV260 (ARM only) | Cortex-A53 @ 1.333GHz | ~7-12 fps |
| Laptop | x86 i5/i7 | ~25-30 fps |

At 7-12 fps, total feedback latency becomes:
```
frame interval (83-140ms) + serial + motor spinup = ~150-200ms total
```

For **mudra correction during practice** (not live performance), 150-200ms is still very usable — the dancer is holding a position, not moving rapidly. It won't feel snappy but it will work.

---

#### With FPGA/DPU acceleration — where it gets interesting

AMD provides **Vitis AI** and a pre-built **DPU** IP block for the KV260. It can accelerate TFLite/ONNX model inference dramatically. AMD ships a pose estimation demo for the KV260 out of the box.

**If MediaPipe's BlazePose model runs through the DPU:**
- Estimated throughput: 25-30+ fps (matching a laptop)
- Latency back down to ~50-80ms total
- Completely self-contained, no laptop needed

This is advanced work — porting MediaPipe models to Vitis AI requires converting them to Xilinx's XIR format. But it's exactly the kind of Phase 7 contribution that makes this project extraordinary at a science fair or beyond.

---

#### Realistic path

```
Now          →  Laptop (easy, works today)
Science fair →  Laptop or Pi 4 (portable, known quantity)
Phase 2      →  KV260 ARM-only (prove it runs, measure fps drop)
Phase 3      →  KV260 + DPU acceleration (the real prize)
```

Don't try to jump to KV260+DPU for the science fair. Get it working on the laptop first, then treat KV260 as the next research chapter.

---

### LiPo for the Arduino Nano

The Nano accepts:
- **5V** via USB pin (regulated input)
- **7-12V** via VIN pin (has onboard regulator)

A single cell LiPo is 3.7V nominal — too low for VIN, needs a boost converter for 5V.

#### Voltage options

| Option | What it is | Cost | Notes |
|--------|-----------|------|-------|
| **2S LiPo (7.4V) → VIN** | Two cells in series, straight into VIN | ~$8 | Simplest wiring |
| **1S LiPo + boost converter to 5V** | Single cell + small DC-DC module | ~$6 | Lighter, smaller |
| **Adafruit PowerBoost 500C** | 1S LiPo + boost + USB charging circuit | ~$10 | Best — charges via USB, clean |

#### Current budget

| Load | Current draw |
|------|-------------|
| Arduino Nano idle | ~20mA |
| Arduino Nano active | ~40mA |
| 1 vibration motor buzzing | ~80-100mA |
| All 5 motors buzzing (worst case) | ~500mA |
| **Realistic average** (1-2 fingers at a time) | **~150-200mA** |

A **500mAh 1S LiPo** gives ~2.5 hours continuous use. Recharges in ~1 hour via USB with the PowerBoost 500C.

---

### Full wireless end-state architecture

```
┌─────────────────────────────────────────┐
│         KV260 (eventually)              │
│         or Laptop (now)                 │
│                                         │
│  USB webcam / CSI camera                │
│      ↓                                  │
│  MediaPipe → mudra_reference.py         │
│      ↓                                  │
│  Bluetooth serial (pyserial)            │
└──────────────┬──────────────────────────┘
               │ Bluetooth
               ▼
┌──────────────────────────────┐
│  Arduino Nano + HC-05        │
│  Adafruit PowerBoost 500C    │  ← charges via USB
│  1S LiPo 500mAh              │  ← ~2.5hrs runtime
│  (velcro wrist pouch)        │
└──────────────┬───────────────┘
               │ thin wires
        5x vibration motors
        (fingertips)
```

**Fully wireless. Fully self-contained on the dancer. Rechargeable.**

---

---

## Haptic Upgrade Path — LRA Motor + DRV2605L

### Current approach (prototype)
The 1027 coin ERM (Eccentric Rotating Mass) motors produce a simple buzz — on or off, or intensity-scaled via PWM. Good enough for "this finger is wrong."

### Upgrade: LRA + DRV2605L haptic driver IC

If you want to distinguish *how wrong* a finger is — e.g. slightly off vs completely wrong — or give different feedback per finger type, upgrade to:

- **LRA (Linear Resonant Actuator)** — produces precise, directional vibration pulses instead of a rotating buzz. Feels more like a distinct tap than a hum.
- **DRV2605L haptic driver IC** (Texas Instruments) — dedicated haptic controller with 123 built-in waveform effects. Connects via I2C (2 wires). Controls the LRA with precise waveforms.

### What this unlocks

| Feedback type | ERM (current) | LRA + DRV2605L |
|--------------|--------------|----------------|
| Finger wrong | buzz on | buzz on |
| Finger slightly off | same buzz | gentle tap |
| Finger completely wrong | same buzz | strong double-pulse |
| Finger just corrected | buzz off | satisfying click effect |
| Different fingers | same feel | different waveforms possible |

### Wiring (I2C, very simple)

```
DRV2605L SDA → Arduino SDA pin
DRV2605L SCL → Arduino SCL pin
DRV2605L VCC → 5V
DRV2605L GND → GND
LRA motor    → DRV2605L OUT+ / OUT-
```

One DRV2605L per finger (5 total), all sharing the same I2C bus with different addresses — or use a TCA9548A I2C multiplexer to give each its own address.

### Cost

| Part | Cost |
|------|------|
| 5x LRA motors (Adafruit #1201 or similar) | ~$15 |
| 5x DRV2605L breakout boards (Adafruit #2305) | ~$25 |
| TCA9548A I2C multiplexer (if needed) | ~$5 |
| **Total upgrade cost** | **~$45** |

### When to consider this

Not needed for the prototype or science fair demo. Consider it if:
- The simple buzz feedback turns out to feel too coarse to be useful
- You want to publish or present at a higher level (regional/national fair, paper)
- Phase 2 of the project after science fair validation
