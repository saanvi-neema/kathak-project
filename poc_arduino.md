# POC: Arduino + Vibration Motor

## Hardware Used
- Arduino Mega (or Uno)
- Vibration motor **module** (3-pin breakout board, not bare motor wires)
- Jumper wires

## Wiring
```
Module VCC → Arduino 5V
Module GND → Arduino GND
Module IN  → Arduino D3
```

## POC Sketch
Buzzes motor when `'1'` is received over Serial:

```cpp
const int MOTOR = 3;

void setup() {
  pinMode(MOTOR, OUTPUT);
  digitalWrite(MOTOR, LOW);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '1') {
      digitalWrite(MOTOR, HIGH);
      delay(300);
      digitalWrite(MOTOR, LOW);
    }
  }
}
```

## Test Method
1. Upload sketch via Arduino IDE
2. Open Serial Monitor (top right), baud = 9600
3. Type `1`, hit Send → motor buzzes for 300ms

## Full POC Chain Confirmed
Once the module test passed, the full end-to-end chain was verified:

```
Camera → 1.5s chunk → Flask → live_pipeline.py → haptic.py → pyserial → Arduino D3 → motor buzz
```

## Progression to Full Protocol
The POC sketch was replaced by `arduino/motor_test/motor_test.ino` which uses a two-byte serial protocol (`'F'` + finger index 0–4) to support 5 fingers independently on pins D3–D7.

## Notes
- Motor was held by hand against finger during POC testing (not yet soldered)
- No transistor used in POC — direct GPIO to module IN pin
- Transistor (2N2222) + flyback diode (1N4007) needed for the final soldered glove build
