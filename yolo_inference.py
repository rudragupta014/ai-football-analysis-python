from ultralytics import YOLO
import os

# Automatically detect current script folder
base_dir = os.path.dirname(os.path.abspath(__file__))

# Build full paths dynamically
model_path = os.path.join(base_dir, "models", "best.pt")   # ✅ correct path
video_path = os.path.join(base_dir, "input_videos", "video#2.mp4")

# Load and run YOLO
model = YOLO(model_path)
results = model.predict(source=video_path, save=True)

print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)
