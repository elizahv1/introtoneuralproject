from ultralytics import YOLO

if __name__ == '__main__':    
    model = YOLO('yolov8m.pt')     
    model.train(
        data=r'D:\intro proje\conf.yaml',
        epochs=15,
        imgsz=320,     
        batch=32,      
        workers=0,     
        amp=True,      
        project='runs/detect',
        name='model2_medium',
        exist_ok=True
    )