from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt') 
    
    model.train(
        data=r'D:\intro proje\conf.yaml',
        epochs=15,
        imgsz=320,     
        batch=32,      
        workers=2,     
        amp=True,      
        exist_ok=True, 
       

    )
