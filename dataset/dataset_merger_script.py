
import os
import shutil
import random
from collections import defaultdict
import hashlib

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Class mappings
# bugday_new: 0:Powdery_Mildew, 1:Septoria, 2:Stem_Rust, 3:Yellow_Rust
# bugday_hasta: 4:BlackPoint, 5:FusariumFootRot, 6:HealthyLeaf, 7:LeafBlight, 8:WheatBlast
# bugday_overfitting: 9:wfd_dataset

CLASS_MAP = {
    'Powdery_Mildew': 0,
    'Septoria': 1,
    'Stem_Rust': 2,
    'Yellow_Rust': 3,
    'BlackPoint': 4,
    'FusariumFootRot': 5,
    'HealthyLeaf': 6,
    'LeafBlight': 7,
    'WheatBlast': 8,
    'wfd_dataset': 9
}

REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
REVERSE_CLASS_MAP[-1] = 'Background'

def process_datasets():
    all_data = [] # List of (image_path, label_content, primary_class_id)
    seen_hashes = {} # md5 -> (image_path, class_id)

    # 1. Process bugday_new (Detection)
    print("Processing bugday_new...")
    new_base = "bugday_new"
    for split in ['train', 'valid']:
        img_dir = os.path.join(new_base, split, 'images')
        lbl_dir = os.path.join(new_base, split, 'labels')
        if not os.path.exists(img_dir): continue
        
        for img_name in os.listdir(img_dir):
            if not img_name.endswith('.jpg'): continue
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, img_name.replace('.jpg', '.txt'))
            
            if not os.path.exists(lbl_path): continue
            
            h = get_md5(img_path)
            if h in seen_hashes:
                # Duplicate found, skip if it's the same image
                continue
            seen_hashes[h] = img_path
            
            with open(lbl_path, 'r') as f:
                lbl_content = f.read()
            
            # Determine primary class (for stratified split)
            # We use the first class mentioned in labels
            lines = lbl_content.strip().split('\n')
            if not lines or not lines[0].strip():
                primary_class = -1 # Background/Unknown
            else:
                try:
                    primary_class = int(lines[0].split()[0])
                except (IndexError, ValueError):
                    primary_class = -1
            
            all_data.append({
                'img_path': img_path,
                'lbl_content': lbl_content,
                'class_id': primary_class,
                'type': 'detection'
            })

    # 2. Process bugday_hasta (Classification)
    print("Processing bugday_hasta...")
    hasta_base = "bugday_hasta"
    hasta_classes = ['BlackPoint', 'FusariumFootRot', 'HealthyLeaf', 'LeafBlight', 'WheatBlast']
    for split in ['train', 'valid', 'test']:
        for cls_name in hasta_classes:
            cls_dir = os.path.join(hasta_base, split, cls_name)
            if not os.path.exists(cls_dir): continue
            
            cls_id = CLASS_MAP[cls_name]
            for img_name in os.listdir(cls_dir):
                if not img_name.endswith('.jpg'): continue
                img_path = os.path.join(cls_dir, img_name)
                
                h = get_md5(img_path)
                if h in seen_hashes: continue
                seen_hashes[h] = img_path
                
                # Create YOLO label: class_id 0.5 0.5 1.0 1.0
                lbl_content = f"{cls_id} 0.5 0.5 1.0 1.0\n"
                
                all_data.append({
                    'img_path': img_path,
                    'lbl_content': lbl_content,
                    'class_id': cls_id,
                    'type': 'classification'
                })

    # 3. Process bugday_overfitting (Classification)
    print("Processing bugday_overfitting...")
    over_base = "bugday_overfitting"
    over_classes = ['wfd_dataset']
    for split in ['train', 'test']:
        for cls_name in over_classes:
            cls_dir = os.path.join(over_base, split, cls_name)
            if not os.path.exists(cls_dir): continue
            
            cls_id = CLASS_MAP[cls_name]
            for img_name in os.listdir(cls_dir):
                if not img_name.endswith('.jpg'): continue
                img_path = os.path.join(cls_dir, img_name)
                
                h = get_md5(img_path)
                if h in seen_hashes: continue
                seen_hashes[h] = img_path
                
                lbl_content = f"{cls_id} 0.5 0.5 1.0 1.0\n"
                
                all_data.append({
                    'img_path': img_path,
                    'lbl_content': lbl_content,
                    'class_id': cls_id,
                    'type': 'classification'
                })

    # Shuffle and Split
    random.seed(42)
    random.shuffle(all_data)
    
    # Stratified Split (simple version: split per class)
    class_to_data = defaultdict(list)
    for item in all_data:
        class_to_data[item['class_id']].append(item)
    
    train_data = []
    val_data = []
    test_data = []
    
    for cls_id, items in class_to_data.items():
        n = len(items)
        n_train = int(n * 0.7)
        n_val = int(n * 0.2)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train+n_val])
        test_data.extend(items[n_train+n_val:])
        
    print(f"Total images: {len(all_data)}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Save to merged directory
    output_dir = "merged_dataset"
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
    def save_split(data, split_name):
        for i, item in enumerate(data):
            ext = os.path.splitext(item['img_path'])[1]
            new_name = f"{REVERSE_CLASS_MAP[item['class_id']]}_{i}{ext}"
            
            target_img = os.path.join(output_dir, 'images', split_name, new_name)
            target_lbl = os.path.join(output_dir, 'labels', split_name, new_name.replace(ext, '.txt'))
            
            shutil.copy(item['img_path'], target_img)
            with open(target_lbl, 'w') as f:
                f.write(item['lbl_content'])
                
    save_split(train_data, 'train')
    save_split(val_data, 'valid')
    save_split(test_data, 'test')
    
    # Create data.yaml
    yaml_content = f"""
train: ./images/train
val: ./images/valid
test: ./images/test

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())
        
    print("Merge complete! Dataset saved in 'merged_dataset'")

if __name__ == "__main__":
    process_datasets()
