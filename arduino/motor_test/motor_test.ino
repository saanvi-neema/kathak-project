#include <SoftwareSerial.h>

SoftwareSerial BT(10, 11); // RX=pin10, TX=pin11

const int MOTOR = 3;
const int BUZZ_MS = 300;

void setup() {
  pinMode(MOTOR, OUTPUT);
  digitalWrite(MOTOR, LOW);
  Serial.begin(9600);   // USB debug
  BT.begin(9600);       // Bluetooth
  Serial.println("Ready");
}

void loop() {
  if (BT.available() > 0) {
    char c = BT.read();
    Serial.print("Got: ");
    Serial.println(c);
    if (c == '1') {
      Serial.println("Buzzing...");
      digitalWrite(MOTOR, HIGH);
      delay(BUZZ_MS);
      digitalWrite(MOTOR, LOW);
    }
  }
}
