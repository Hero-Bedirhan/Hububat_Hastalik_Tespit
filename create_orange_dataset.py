import os
import shutil
import random
from pathlib import Path

# Ayarlar
SOURCE_DIR = "dataset/classification_dataset/train"
DEST_DIR   = "dataset/orange_unlabeled_dataset"
IMAGES_PER_CLASS = 100  # Orange'ın hızlı çalışması için her sınıftan 100 resim (Toplam 500)

def create_orange_dataset():
    print(f"Orange için sınıfsız veri seti oluşturuluyor: {DEST_DIR}")
    
    # Hedef klasörü oluştur (varsa temizle)
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR)

    src_path = Path(SOURCE_DIR)
    classes = [d for d in src_path.iterdir() if d.is_dir()]
    
    all_selected_images = []
    
    # Her sınıftan resimleri topla
    for cls in classes:
        images = [f for f in cls.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        random.seed(42) # Her seferinde aynı resimleri seçmesi için
        selected = random.sample(images, min(IMAGES_PER_CLASS, len(images)))
        
        for img in selected:
            # İsmini sınıftan bağımsız yapalım ama gerçek sınıfı da loglayalım ki sonradan kontrol edilebilsin
            # Orange'da isimden anlamaması için karmaşık bir id veriyoruz
            all_selected_images.append({
                "path": img,
                "real_class": cls.name
            })
    
    # Resimleri karıştır
    random.shuffle(all_selected_images)
    
    # Kopyala ve isimlendir
    print(f"Toplam {len(all_selected_images)} fotoğraf kopyalanıyor ve isimlendiriliyor...")
    for i, item in enumerate(all_selected_images):
        # Örn: image_001.jpg, image_002.jpg (Sınıf bilgisi tamamen silindi!)
        new_name = f"image_{str(i+1).zfill(4)}{item['path'].suffix}"
        dest_path = os.path.join(DEST_DIR, new_name)
        shutil.copy2(item['path'], dest_path)
    
    print("\n✅ İŞLEM TAMAMLANDI!")
    print(f"Klasör hazır: {DEST_DIR}")
    print(f"Bu klasörü Orange'daki 'Image Import' widget'ına bağlayabilirsiniz.")

if __name__ == "__main__":
    create_orange_dataset()
