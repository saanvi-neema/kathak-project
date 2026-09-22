# POC Wiring — Arduino Uno + HC-05 + Vibration Motor

## Components
- Arduino Uno
- HC-05 Bluetooth module (Inland, Microcenter)
- Vibration motor module (3-pin: VCC, GND, IN)
- Breadboard
- 1kΩ resistor
- 2kΩ resistor
- Jumper wires
- 9V battery (power) or USB cable (upload + power)

## Connections

### HC-05 → Arduino
| HC-05 Pin | Arduino Pin | Notes |
|---|---|---|
| VCC | 5V | Power |
| GND | GND | Ground |
| TXD | Pin 10 | Direct — no resistor needed |
| RXD | Pin 11 | Via voltage divider (see below) |
| KEY | — | Leave unconnected |
| STATE | — | Leave unconnected |

### Voltage Divider (HC-05 RXD protection)
HC-05 RXD expects 3.3V max. Arduino pin 11 outputs 5V.

```
Arduino pin 11 ── 1kΩ ──┬── HC-05 RXD
                         │
                        2kΩ
                         │
                        GND
```
Result: 5V × 2/(1+2) = 3.33V at HC-05 RXD ✓

### Vibration Motor Module → Arduino
| Motor Module Pin | Arduino Pin |
|---|---|
| VCC | 5V |
| GND | GND |
| IN | Pin 3 |

## Power
- **During development**: USB cable to laptop (powers Arduino + HC-05 + motor)
- **Wireless demo**: 9V battery to Arduino barrel jack (USB disconnected)

## Bluetooth
- Windows COM port: **COM7** (outgoing)
- Default pairing PIN: **1234**
- Baud rate: **9600**

## Software
- Arduino sketch: `arduino/motor_test/motor_test.ino`
  - SoftwareSerial on pins 10/11
  - Buzzes motor on pin 3 when `'1'` received
- Python test: `test_buzz.py`
  - Sends `'1'` to COM7 every 1.5 seconds
