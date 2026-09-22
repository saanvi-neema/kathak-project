import serial, time

PORT = "COM7"
BAUD = 9600

print(f"Opening {PORT} at {BAUD} baud...")
ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"Opened. Waiting 2s for Arduino reset...")
time.sleep(2)
print("Ready. Buzzing in loop — Ctrl+C to stop")

i = 1
while True:
    print(f"Buzz {i} — sending '1'...")
    bytes_sent = ser.write(b'1')
    print(f"  sent {bytes_sent} byte(s)")
    ser.flush()
    time.sleep(1.5)
    i += 1
