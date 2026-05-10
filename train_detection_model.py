from ultralytics import YOLO
import os

def main():
    # Model Seçimi: YOLOv8s (Detection - Nesne Algılama) modelini kullanıyoruz.
    # Bu model hem hız hem de doğruluk açısından dengelidir.
    model = YOLO('yolov8s.pt')

    # Veri seti yapılandırma dosyasının yolu
    data_yaml_path = '/home/bedirhan/Desktop/Vs_Code/Git/Hububat_Hastalik_Tespit/dataset/detection_dataset/data.yaml'

    print("Birleştirilmiş veri seti ile YOLOv8 Eğitimine başlanıyor...")
    print(f"Veri seti: {data_yaml_path}")
    print("Sınıf sayısı: 10")
    
    # Eğitim ayarları
    # Yellow_Rust gibi az verili sınıflar için augmentasyon (veri artırma) parametrelerini optimize ediyoruz.
    model.train(
        data=data_yaml_path,
        epochs=100,            # 100 epoch eğitim
        imgsz=640,             # Görüntü boyutu 640x640
        batch=16,              # Batch size (belleğe göre ayarlanabilir)
        patience=20,           # 20 epoch boyunca iyileşme olmazsa erken durdurma (early stopping)
        
        # Veri Artırma (Augmentation) Ayarları:
        # Az veriye sahip sınıflar için bu ayarlar kritiktir.
        mosaic=1.0,            # 4 görüntüyü birleştirerek mozaik oluşturma (Daha karmaşık sahneler)
        mixup=0.1,             # İki görüntüyü karıştırma (Overfitting'i azaltır)
        degrees=10.0,          # Rastgele döndürme (derece)
        translate=0.1,         # Rastgele kaydırma
        scale=0.5,             # Rastgele ölçekleme
        shear=2.0,             # Rastgele yamultma
        flipud=0.0,            # Dikey çevirme (yapraklar için çok doğal olmayabilir, 0 bıraktık)
        fliplr=0.5,            # Yatay çevirme (Ayna görüntüsü)
        hsv_h=0.015,           # HSV Renk tonu değişimi
        hsv_s=0.7,             # HSV Doygunluk değişimi
        hsv_v=0.4,             # HSV Parlaklık değişimi
        
        project='runs/detection_results',
        name='detection_training',
        exist_ok=True,
        device=0               # Varsa GPU kullanımı (0), yoksa 'cpu' olarak otomatik ayarlanır
    )

    # Eğitim sonrası modeli kaydet
    print("Eğitim tamamlandı. Model 'runs/detect/merged_wheat_disease_train' klasörüne kaydedildi.")

if __name__ == '__main__':
    main()
