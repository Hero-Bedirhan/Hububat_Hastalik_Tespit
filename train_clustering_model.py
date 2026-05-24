"""
Kümeleme Modeli Eğitimi (K-Means) — YOLO Backbone Tabanlı
==========================================================
classification_model.pt'nin kendi backbone'unu özellik çıkarıcı
olarak kullanır. Ayrı bir ResNet18 / dış modele gerek yoktur.

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

# ── torch.load uyumluluk düzeltmesi (PyTorch 2.6+ weights_only=True sorunu) ─
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
_orig_load = torch.load
def _compat_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _compat_load
try:
    from ultralytics.nn.tasks import ClassificationModel
    torch.serialization.add_safe_globals([ClassificationModel])
except Exception:
    pass

import torchvision.transforms as T

from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ── AYARLAR ──────────────────────────────────────────────────────────────────
DATASET_PATH    = "dataset/classification_dataset"
YOLO_MODEL_PATH = "models/classification_model.pt"
MODEL_SAVE_PATH = "models/clustering_model.pkl"
N_CLUSTERS      = 5
RANDOM_SEED     = 42

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_backbone() -> torch.nn.Module:
    """YOLO sınıflandırma modelinin son Classify katmanını kaldırır.
    Geriye kalan backbone, hastalığa özgü özellik vektörleri üretir."""
    yolo     = YOLO(YOLO_MODEL_PATH)
    net      = yolo.model          # ClassificationModel (nn.Module)
    # net.model → tüm katmanların Sequential'ı; [-1] Classify head'dir
    backbone = torch.nn.Sequential(*list(net.model)[:-1])
    backbone.eval()
    print(f"  YOLO backbone yüklendi ({len(list(backbone.parameters()))} parametre tensörü)")
    return backbone


def extract_features(backbone: torch.nn.Module, image_paths: list) -> np.ndarray:
    """Her görüntüden tek bir düz özellik vektörü çıkarır (global avg pool)."""
    feats = []
    with torch.no_grad():
        for i, path in enumerate(image_paths):
            try:
                img    = Image.open(path).convert("RGB")
                tensor = TRANSFORM(img).unsqueeze(0)
                out    = backbone(tensor)
                # Spatial boyutlar varsa global average pool ile tek vektöre indir
                if out.dim() > 2:
                    out = out.mean(dim=[2, 3])
                feats.append(out.squeeze().cpu().numpy())
            except Exception as e:
                print(f"  ⚠  Atlandı: {Path(path).name} — {e}")

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(image_paths)} işlendi…")
    return np.array(feats)


def collect_images(dataset_path: str) -> tuple:
    """Train split'indeki görüntü yollarını ve etiketleri toplar."""
    paths, labels, class_names = [], [], []
    train_dir = Path(dataset_path) / "train"
    if not train_dir.exists():
        print(f"HATA: {train_dir} bulunamadı!")
        return [], [], []
    for cls_dir in sorted(train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cname = cls_dir.name
        if cname not in class_names:
            class_names.append(cname)
        cid  = class_names.index(cname)
        imgs = [str(f) for f in cls_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        paths.extend(imgs)
        labels.extend([cid] * len(imgs))
        print(f"  ✔  {cname}: {len(imgs)} görüntü")
    return paths, labels, class_names


def main():
    print("=" * 62)
    print("   K-Means Kümeleme — YOLO Backbone Tabanlı Özellik Çıkarma")
    print("=" * 62)

    # 1. Veri toplama
    print("\n[1/5] Veri seti taranıyor…")
    paths, labels, class_names = collect_images(DATASET_PATH)
    if not paths:
        sys.exit("HATA: Görüntü bulunamadı!")
    labels = np.array(labels)
    print(f"\n  Toplam: {len(paths)} görüntü  |  Sınıflar: {class_names}")

    # 2. YOLO backbone ile özellik çıkarma
    print("\n[2/5] YOLO backbone ile özellik çıkarılıyor…")
    backbone = build_backbone()
    features = extract_features(backbone, paths)
    print(f"  Özellik matrisi: {features.shape}")

    # 3. Normalleştirme
    print("\n[3/5] StandardScaler normalleştirmesi…")
    scaler          = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 4. K-Means
    print(f"\n[4/5] K-Means uygulanıyor (k={N_CLUSTERS})…")
    kmeans      = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED,
                         n_init=15, max_iter=500)
    cluster_ids = kmeans.fit_predict(features_scaled)

    if len(set(cluster_ids)) > 1:
        sil = silhouette_score(features_scaled, cluster_ids,
                               sample_size=min(5000, len(features_scaled)))
        print(f"  Silhouette Skoru: {sil:.4f}")

    # Küme ↔ Sınıf çoğunluk eşlemesi
    cluster_to_class, cluster_class_dist = {}, {}
    for c in range(N_CLUSTERS):
        mask   = cluster_ids == c
        c_lbls = labels[mask]
        if len(c_lbls) == 0:
            cluster_to_class[c] = "Bilinmeyen"
            continue
        counts  = np.bincount(c_lbls, minlength=len(class_names))
        maj     = int(np.argmax(counts))
        cluster_to_class[c]   = class_names[maj]
        cluster_class_dist[c] = {class_names[i]: int(counts[i]) for i in range(len(class_names))}

    print("\n  Küme → Sınıf Eşlemesi:")
    for c, name in cluster_to_class.items():
        print(f"    Küme {c} → {name:20s} | {cluster_class_dist.get(c, {})}")

    # Bilinmeyen hastalık eşiği (95. persentil)
    dists = [float(np.linalg.norm(features_scaled[i] - kmeans.cluster_centers_[cluster_ids[i]]))
             for i in range(len(features_scaled))]
    threshold = float(np.percentile(dists, 95))
    print(f"\n  Yeni hastalık eşiği (95. persentil): {threshold:.4f}")

    # 5. Kaydet
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "kmeans"            : kmeans,
        "scaler"            : scaler,
        "cluster_to_class"  : cluster_to_class,
        "cluster_class_dist": cluster_class_dist,
        "class_names"       : class_names,
        "n_clusters"        : N_CLUSTERS,
        "unknown_threshold" : threshold,
    }, MODEL_SAVE_PATH)

    print(f"\n[5/5] Model kaydedildi → {MODEL_SAVE_PATH}")
    print("=" * 62)
    print("  Eğitim tamamlandı! 🎉")
    print("=" * 62)


if __name__ == "__main__":
    main()
