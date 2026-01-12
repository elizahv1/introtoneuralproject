from ultralytics import YOLO
import os
model_path = r'D:\intro proje\runs\detect\train\weights\best.pt'
model = YOLO(model_path)
source_path = r'D:\intro proje\datasets\images\test'
results = model.predict(
    source=source_path, 
    save=True, 
    conf=0.25, 
    name='predict_nano'
)

print(f"Nano Prediction finished. Results are in: {results[0].save_dir}")