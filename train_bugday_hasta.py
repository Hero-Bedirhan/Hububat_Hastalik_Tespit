from ultralytics import YOLO

def main():
    # Model Seçimi: Görüntü Sınıflandırma (Classification) için -cls uzantılı modeli kullanıyoruz.
    model = YOLO('yolov8s-cls.pt')

    # Veri seti yolu: Sınıflandırma için veri setinin ana klasör yolunu veriyoruz.
    # YOLO classification için data.yaml dosyasına ihtiyaç duymaz, klasör isimlerini sınıf olarak algılar.
    data_path = '/home/bedirhan/Desktop/Vs_Code/Git/Hububat_Hastalik_Tespit/bugday_hasta'

    print("bugday_hasta veri seti ile Görüntü Sınıflandırma eğitimine başlanıyor...")
    
    # Eğitim ayarları (Öncekiyle aynı mantıkta)
    model.train(
        data=data_path,
        epochs=100,        # Maksimum 100 epoch.
        imgsz=640,         # Görüntü boyutu
        batch=16,          # Batch size
        patience=20,       # 20 tur gelişme olmazsa durdur
        augment=True,      # Veri çoğaltma otomatik uygulanır
        project='runs/classify',
        name='bugday_hasta_train', # Sonuçlar runs/classify/bugday_hasta_train içine kaydedilir
        exist_ok=True
    )

if __name__ == '__main__':
    main()
