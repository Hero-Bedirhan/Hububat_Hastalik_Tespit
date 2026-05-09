from ultralytics import YOLO

# 1. Model Seçimi (yolov8s daha akıllı ve güçlüdür)
model = YOLO('yolov8s.pt')

# 2. Eğitim Parametreleri
# Yeni birleştirilmiş veri setimizin data.yaml yolunu veriyoruz
data_path = '/home/bedirhan/Desktop/Hububat_Hastalik_Tespit/Combined_Dataset/data.yaml'

# 3. Eğitimi Başlat
# Veri seti genişlediği ve model büyüdüğü için epochs değerini artırıyoruz.
# YOLO, erken öğrenirse kendi kendine eğitimi durduracaktır (patience=50).
model.train(
    data=data_path, 
    epochs=20, 
    imgsz=640, 
    patience=50, # 50 tur boyunca gelişme olmazsa durdurur
    augment=True # Veri çeşitlendirmeyi (augmentation) otomatik yapar
)
