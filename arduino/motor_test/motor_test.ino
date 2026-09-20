// Finger pin mapping: thumb=3, index=4, middle=5, ring=6, pinky=7
const int FINGER_PINS[] = {3, 4, 5, 6, 7};
const int BUZZ_MS = 300;

void setup() {
  for (int i = 0; i < 5; i++) {
    pinMode(FINGER_PINS[i], OUTPUT);
    digitalWrite(FINGER_PINS[i], LOW);
  }
  Serial.begin(9600);
}

void loop() {
  // Protocol: two bytes — 'F' then finger index (0-4)
  if (Serial.available() >= 2) {
    char cmd = Serial.read();
    int finger = Serial.read();
    if (cmd == 'F' && finger >= 0 && finger < 5) {
      digitalWrite(FINGER_PINS[finger], HIGH);
      delay(BUZZ_MS);
      digitalWrite(FINGER_PINS[finger], LOW);
    }
  }
}
