import urllib.request, os

os.makedirs("models", exist_ok=True)

models = {
    "models/pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "models/hand_landmarker.task":      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "models/face_landmarker.task":      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}

for path, url in models.items():
    if os.path.exists(path):
        print(f"already exists: {path}")
        continue
    print(f"downloading {path} ...")
    urllib.request.urlretrieve(url, path)
    print(f"done: {os.path.getsize(path) // 1024 // 1024} MB")

print("all models ready.")
