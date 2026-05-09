import os
import shutil

# Yeni birleştirilmiş veri seti yolları
base_dir = "/home/bedirhan/Desktop/Hububat_Hastalik_Tespit"
ds1_dir = os.path.join(base_dir, "Wheat-Disease.v1i.yolov8")
ds2_dir = os.path.join(base_dir, "Wheat Disease.v1i.yolov8")
out_dir = os.path.join(base_dir, "Combined_Dataset")

# Klasörleri oluştur
splits = ["train", "valid", "test"]
for split in splits:
    os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)

# Yeni sınıf listesi
# Dataset 1: ['Black Rust', 'Brown Rust', 'Healthy Wheat', 'Yellow Rust']
# Dataset 2: ['Powdery_Mildew', 'Septoria', 'Stem_Rust', 'Yellow_Rust']
# Yellow Rust her ikisinde de var (isim farkı var sadece)
new_classes = [
    "Black Rust",      # 0 (DS1: 0)
    "Brown Rust",      # 1 (DS1: 1)
    "Healthy Wheat",   # 2 (DS1: 2)
    "Yellow Rust",     # 3 (DS1: 3, DS2: 3)
    "Powdery Mildew",  # 4 (DS2: 0)
    "Septoria",        # 5 (DS2: 1)
    "Stem Rust"        # 6 (DS2: 2)
]

# Dataset 2 için sınıf eşleme (Mapping)
# Eski ID -> Yeni ID
ds2_mapping = {
    0: 4, # Powdery_Mildew -> Powdery Mildew
    1: 5, # Septoria -> Septoria
    2: 6, # Stem_Rust -> Stem Rust
    3: 3  # Yellow_Rust -> Yellow Rust
}

def process_dataset(ds_dir, prefix, mapping=None):
    for split in splits:
        img_dir = os.path.join(ds_dir, split, "images")
        lbl_dir = os.path.join(ds_dir, split, "labels")
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue
            
        for img_name in os.listdir(img_dir):
            # Dosya isimlerinin çakışmaması için önek ekliyoruz
            new_img_name = f"{prefix}_{img_name}"
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(out_dir, split, "images", new_img_name)
            shutil.copy(src_img, dst_img)
            
            # İlgili label dosyasını bul (uzantısı genelde .txt olur)
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = os.path.join(lbl_dir, lbl_name)
            dst_lbl = os.path.join(out_dir, split, "labels", f"{prefix}_{lbl_name}")
            
            if os.path.exists(src_lbl):
                if mapping is None:
                    # Dataset 1 için doğrudan kopyala (ID'ler aynı)
                    shutil.copy(src_lbl, dst_lbl)
                else:
                    # Dataset 2 için ID'leri eşlemeye göre değiştir
                    with open(src_lbl, "r") as f_in, open(dst_lbl, "w") as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            if parts:
                                old_class_id = int(parts[0])
                                if old_class_id in mapping:
                                    new_class_id = mapping[old_class_id]
                                    parts[0] = str(new_class_id)
                                    f_out.write(" ".join(parts) + "\n")

print("Dataset 1 işleniyor...")
process_dataset(ds1_dir, "ds1")

print("Dataset 2 işleniyor (Sınıflar yeniden numaralandırılıyor)...")
process_dataset(ds2_dir, "ds2", ds2_mapping)

# Yeni data.yaml oluştur
yaml_content = f"""train: {out_dir}/train/images
val: {out_dir}/valid/images
test: {out_dir}/test/images

nc: {len(new_classes)}
names: {new_classes}
"""

with open(os.path.join(out_dir, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("Birleştirme tamamlandı! Yeni veri seti: Combined_Dataset")
