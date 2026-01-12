from ultralytics import YOLO
import os

model_path = r'D:\intro proje\runs\detect\model2_medium\weights\best.pt'
model = YOLO(model_path)
source_path = r'D:\intro proje\datasets\images\test'
results = model.predict(source=source_path, save=True, conf=0.25)

print(f"Prediction finished. Results are saved to: {results[0].save_dir}")