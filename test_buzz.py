import serial, time

ser = serial.Serial("COM5", 9600, timeout=1)
time.sleep(2)  # wait for Arduino reset

print("Buzzing in loop — Ctrl+C to stop")
i = 1
while True:
    print(f"Buzz {i}...")
    ser.write(bytes([ord('F'), 0]))
    i += 1
    time.sleep(1.5)
