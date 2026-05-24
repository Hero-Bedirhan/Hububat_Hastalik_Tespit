"""
Kümeleme Modeli Eğitimi (K-Means)
==================================
Hububat hastalık görüntülerinden ResNet18 ile özellik çıkarır,
K-Means kümeleme uygular ve modeli kaydeder.

Kullanım:
    python train_clustering_model.py

Çıktı:
    models/clustering_model.pkl
"""

import os
import sys
import numpy as np
import joblib
from pathlib import Path
from PIL import Image

import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ── AYARLAR ──────────────────────────────────────────────────────────────────
DATASET_PATH    = "dataset/classification_dataset"
MODEL_SAVE_PATH = "models/clustering_model.pkl"
N_CLUSTERS      = 5       # classification_dataset sınıf sayısı
IMG_SIZE        = 224
RANDOM_SEED     = 42

# ImageNet normalleştirmesi
TRANSFORM = tv_transforms.Compose([
    tv_transforms.Resize((IMG_SIZE, IMG_SIZE)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])


def build_feature_extractor() -> torch.nn.Module:
    """ResNet18'in son sınıflandırma katmanı kaldırılarak özellik çıkarıcı oluşturulur.
    Çıktı: 512 boyutlu embedding vektörü."""
    base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    extractor = torch.nn.Sequential(*list(base.children())[:-1])  # FC katmanı yok
    extractor.eval()
    return extractor


def extract_features(extractor: torch.nn.Module, image_paths: list) -> tuple:
    """Görüntü listesinden özellik matrisi çıkarır. Hatalı görseller atlanır."""
    feature_list, valid_paths = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)

    with torch.no_grad():
        for i, path in enumerate(image_paths):
            try:
                img    = Image.open(path).convert("RGB")
                tensor = TRANSFORM(img).unsqueeze(0).to(device)
                feat   = extractor(tensor).squeeze().cpu().numpy()  # (512,)
                feature_list.append(feat)
                valid_paths.append(path)
            except Exception as e:
                print(f"  ⚠  Atlandı: {Path(path).name} → {e}")
                continue

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(image_paths)} görüntü işlendi…")

    return np.array(feature_list), valid_paths


def collect_images(dataset_path: str) -> tuple:
    """Train split'indeki tüm görüntü yollarını ve sınıf etiketlerini toplar."""
    image_paths, labels, class_names = [], [], []
    dataset_path = Path(dataset_path)
    train_dir    = dataset_path / "train"

    if not train_dir.exists():
        print(f"  HATA: {train_dir} bulunamadı!")
        return [], [], []

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in class_names:
            class_names.append(class_name)
        class_id = class_names.index(class_name)

        imgs = [f for f in class_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        image_paths.extend([str(p) for p in imgs])
        labels.extend([class_id] * len(imgs))
        print(f"  ✔  {class_name}: {len(imgs)} görüntü")

    return image_paths, labels, class_names


def main():
    print("=" * 62)
    print("   K-Means Kümeleme Modeli Eğitimi — Hububat Hastalık Tespiti")
    print("=" * 62)

    # 1. Görüntü toplama
    print("\n[1/5] Veri seti taranıyor…")
    image_paths, labels, class_names = collect_images(DATASET_PATH)
    if not image_paths:
        sys.exit("HATA: Veri seti boş!")
    labels = np.array(labels)
    print(f"\n  Toplam görüntü : {len(image_paths)}")
    print(f"  Sınıflar ({len(class_names)}): {class_names}")

    # 2. Özellik çıkarma (ResNet18 backbone)
    print("\n[2/5] Özellik vektörleri çıkarılıyor (ResNet18)…")
    extractor = build_feature_extractor()
    features, valid_paths = extract_features(extractor, image_paths)
    # Yollar değiştiyse etiketleri güncelle
    valid_set = set(valid_paths)
    labels    = np.array([labels[i] for i, p in enumerate(image_paths) if p in valid_set])
    print(f"  Özellik matrisi: {features.shape}  (görüntü × 512 boyut)")

    # 3. Normalleştirme
    print("\n[3/5] StandardScaler normalleştirmesi uygulanıyor…")
    scaler          = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 4. K-Means kümeleme
    print(f"\n[4/5] K-Means kümeleme uygulanıyor (k={N_CLUSTERS})…")
    kmeans      = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=15, max_iter=500)
    cluster_ids = kmeans.fit_predict(features_scaled)

    if len(set(cluster_ids)) > 1:
        sil = silhouette_score(
            features_scaled, cluster_ids,
            sample_size=min(5000, len(features_scaled))
        )
        print(f"  Silhouette Skoru: {sil:.4f}  (1'e yakın = iyi kümeleme)")

    # Küme ↔ Sınıf çoğunluk eşlemesi
    cluster_to_class = {}
    cluster_class_dist = {}
    for c_id in range(N_CLUSTERS):
        mask    = cluster_ids == c_id
        c_lbls  = labels[mask]
        if len(c_lbls) == 0:
            cluster_to_class[c_id] = "Bilinmeyen"
            continue
        counts   = np.bincount(c_lbls, minlength=len(class_names))
        majority = int(np.argmax(counts))
        cluster_to_class[c_id]   = class_names[majority]
        cluster_class_dist[c_id] = {class_names[i]: int(counts[i]) for i in range(len(class_names))}

    print("\n  Küme → Hastalık Sınıfı Eşlemesi:")
    for c_id, c_name in cluster_to_class.items():
        dist_str = cluster_class_dist.get(c_id, {})
        print(f"    Küme {c_id} → {c_name:20s} | {dist_str}")

    # 5. Bilinmeyen hastalık eşiği (95. persentil)
    dists = [
        float(np.linalg.norm(features_scaled[i] - kmeans.cluster_centers_[cluster_ids[i]]))
        for i in range(len(features_scaled))
    ]
    unknown_threshold = float(np.percentile(dists, 95))
    print(f"\n  Yeni/Bilinmeyen hastalık eşiği (95. persentil): {unknown_threshold:.4f}")

    # Kaydet
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "kmeans"            : kmeans,
        "scaler"            : scaler,
        "cluster_to_class"  : cluster_to_class,
        "cluster_class_dist": cluster_class_dist,
        "class_names"       : class_names,
        "n_clusters"        : N_CLUSTERS,
        "unknown_threshold" : unknown_threshold,
    }, MODEL_SAVE_PATH)

    print(f"\n[5/5] Model kaydedildi → {MODEL_SAVE_PATH}")
    print("=" * 62)
    print("  Eğitim tamamlandı! 🎉  Artık app.py'de Kümeleme Modunu kullanabilirsiniz.")
    print("=" * 62)


if __name__ == "__main__":
    main()
