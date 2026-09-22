import serial, time

for port in ["COM6", "COM7", "COM8"]:
    try:
        print(f"Trying {port}...")
        s = serial.Serial(port, 9600, timeout=2)
        print(f"SUCCESS: {port} opened")
        s.close()
    except Exception as e:
        print(f"FAIL {port}: {e}")
