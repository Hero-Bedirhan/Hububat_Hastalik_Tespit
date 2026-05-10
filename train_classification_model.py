from ultralytics import YOLO

def main():
    # Model Seçimi: Görüntü Sınıflandırma (Classification) için -cls uzantılı modeli kullanıyoruz.
    model = YOLO('yolov8s-cls.pt')

    # Veri seti yolu: Sınıflandırma için veri setinin ana klasör yolunu veriyoruz.
    data_path = 'dataset/classification_dataset'

    print("Sınıflandırma veri seti ile YOLOv8-Classification eğitimine başlanıyor...")
    
    # Eğitim ayarları
    model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        augment=True,
        project='runs/classification_results',
        name='classification_training',
        exist_ok=True
    )

if __name__ == '__main__':
    main()
