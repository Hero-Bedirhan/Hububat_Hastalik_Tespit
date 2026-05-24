"""
Kümeleme Görselleştirme — Sınıfsız Ayrıştırma Demosu
======================================================
Veri setindeki fotoğrafları SINIFSIZ (etiket yok) olarak alır,
K-Means ile kümeler ve t-SNE grafiğiyle sonuçları görselleştirir.

K-Means'in gerçek sınıfları "keşfedip keşfedemediğini" gösterir.

Kullanım:
    python visualize_clustering.py

Çıktı:
    clustering_visualization.png
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as T
from ultralytics import YOLO

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ── torch.load uyumluluk düzeltmesi ─────────────────────────────────────────
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
_orig = torch.load
def _compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig(*args, **kwargs)
torch.load = _compat
try:
    from ultralytics.nn.tasks import ClassificationModel
    torch.serialization.add_safe_globals([ClassificationModel])
except Exception:
    pass

# ── AYARLAR ─────────────────────────────────────────────────────────────────
DATASET_PATH    = "dataset/classification_dataset/train"
YOLO_MODEL_PATH = "models/classification_model.pt"
OUTPUT_IMAGE    = "clustering_visualization.png"
N_CLUSTERS      = 5
MAX_PER_CLASS   = 200   # Hız için her sınıftan max görüntü sayısı
RANDOM_SEED     = 42

# Her hastalık sınıfı için renk ve Türkçe ad
CLASS_META = {
    "BlackPoint"     : {"color": "#2c3e50", "label": "Kara Nokta"},
    "FusariumFootRot": {"color": "#8B4513", "label": "Kök Çürüklüğü"},
    "HealthyLeaf"    : {"color": "#27ae60", "label": "Sağlıklı Yaprak"},
    "LeafBlight"     : {"color": "#e67e22", "label": "Yaprak Yanıklığı"},
    "WheatBlast"     : {"color": "#e74c3c", "label": "Buğday Yanıklığı"},
}
CLUSTER_COLORS = ["#3498db", "#9b59b6", "#f1c40f", "#1abc9c", "#e91e63"]

TRANSFORM = T.Compose([T.Resize((224, 224)), T.ToTensor()])


def build_backbone():
    yolo     = YOLO(YOLO_MODEL_PATH)
    net      = yolo.model
    backbone = torch.nn.Sequential(*list(net.model)[:-1])
    backbone.eval()
    return backbone


def collect_and_extract(backbone):
    """Görüntüleri etiket bilgisi olmadan yükler, özellik çıkarır."""
    paths, true_labels, class_names = [], [], []
    train_dir = Path(DATASET_PATH)

    for cls_dir in sorted(train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cname = cls_dir.name
        if cname not in class_names:
            class_names.append(cname)
        cid  = class_names.index(cname)
        imgs = [f for f in cls_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        imgs = imgs[:MAX_PER_CLASS]  # hız için sınırla
        paths.extend(imgs)
        true_labels.extend([cid] * len(imgs))
        print(f"  ✔  {cname}: {len(imgs)} görüntü yüklendi")

    print(f"\n  Toplam: {len(paths)} görüntü")
    print("  Özellik çıkarılıyor (sınıf bilgisi KULLANILMIYOR)…")

    features = []
    with torch.no_grad():
        for i, path in enumerate(paths):
            try:
                img = Image.open(path).convert("RGB")
                t   = TRANSFORM(img).unsqueeze(0)
                out = backbone(t)
                if out.dim() > 2:
                    out = out.mean(dim=[2, 3])
                features.append(out.squeeze().cpu().numpy())
            except Exception:
                features.append(np.zeros(512))
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(paths)} işlendi…")

    return np.array(features), np.array(true_labels), class_names


def main():
    print("=" * 62)
    print("   Kümeleme Görselleştirme — Sınıfsız Ayrıştırma Demosu")
    print("=" * 62)

    # 1. Özellik çıkarma (etiket bilgisi YOK)
    print("\n[1/4] YOLO backbone ile özellik çıkarılıyor…")
    backbone = build_backbone()
    features, true_labels, class_names = collect_and_extract(backbone)

    # 2. Normalleştirme + K-Means (k=5, etiket bilgisi YOK)
    print("\n[2/4] K-Means kümeleme uygulanıyor…")
    scaler          = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans          = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED,
                             n_init=15, max_iter=300)
    cluster_ids     = kmeans.fit_predict(features_scaled)

    # Kalite metrikleri
    ari = adjusted_rand_score(true_labels, cluster_ids)
    nmi = normalized_mutual_info_score(true_labels, cluster_ids)
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}  (1=mükemmel)")
    print(f"  Norm. Mutual Info   (NMI): {nmi:.4f}  (1=mükemmel)")

    # 3. t-SNE ile 2D'ye indir
    print("\n[3/4] t-SNE ile 2D görselleştirme hazırlanıyor…")
    pca_50    = PCA(n_components=50, random_state=RANDOM_SEED)
    feat_pca  = pca_50.fit_transform(features_scaled)
    tsne      = TSNE(n_components=2, random_state=RANDOM_SEED,
                     perplexity=40, verbose=0)
    tsne_2d   = tsne.fit_transform(feat_pca)

    # 4. Görselleştir
    print("\n[4/4] Grafik oluşturuluyor…")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Sol: K-Means kümeleri (sınıf bilgisi yok!)
    ax = axes[0]
    for c in range(N_CLUSTERS):
        mask = cluster_ids == c
        ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                   c=CLUSTER_COLORS[c], alpha=0.65, s=12, label=f"Küme {c+1}")
    ax.set_title("K-Means Kümeleri\n(Sınıf bilgisi kullanılmadı!)",
                 color="white", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor="#0f3460", edgecolor="#444",
              labelcolor="white", fontsize=9)
    ax.set_xlabel("t-SNE 1", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("t-SNE 2", color="#aaaaaa", fontsize=9)

    # Sağ: Gerçek sınıflar (gizlenmiş etiketlerin açılması)
    ax = axes[1]
    for cid, cname in enumerate(class_names):
        mask  = true_labels == cid
        meta  = CLASS_META.get(cname, {"color": "#888888", "label": cname})
        ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                   c=meta["color"], alpha=0.65, s=12, label=meta["label"])
    ax.set_title("Gerçek Hastalık Sınıfları\n(Etiketler açıklandı)",
                 color="white", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", facecolor="#0f3460", edgecolor="#444",
              labelcolor="white", fontsize=9)
    ax.set_xlabel("t-SNE 1", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("t-SNE 2", color="#aaaaaa", fontsize=9)

    # Başlık ve metrikler
    fig.suptitle(
        f"Kümeleme Analizi  ·  {len(features)} Görüntü  ·  {N_CLUSTERS} Küme\n"
        f"ARI: {ari:.3f}   NMI: {nmi:.3f}   "
        f"(1'e yaklaştıkça K-Means gerçek sınıfları daha iyi keşfetmiş demektir)",
        color="white", fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n  Grafik kaydedildi → {OUTPUT_IMAGE}")
    print("=" * 62)
    print(f"  ARI: {ari:.4f}  |  NMI: {nmi:.4f}")
    print("  Bu değerler K-Means'in etiket olmadan kaç hastalığı")
    print("  doğru grupladığını gösterir. Video için bu grafiği kullan!")
    print("=" * 62)


if __name__ == "__main__":
    main()
