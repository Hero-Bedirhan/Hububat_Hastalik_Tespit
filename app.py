import streamlit as st
import os
import numpy as np
import joblib

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from PIL import Image
from ultralytics import YOLO
import torch

_ORIGINAL_TORCH_LOAD = torch.load

def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)

torch.load = _torch_load_compat

def _configure_torch_safe_globals():
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return
    try:
        from ultralytics.nn.tasks import (
            ClassificationModel, DetectionModel,
            OBBModel, PoseModel, SegmentationModel,
        )
        add_safe_globals([
            DetectionModel, ClassificationModel,
            SegmentationModel, PoseModel, OBBModel,
        ])
    except Exception:
        pass

_configure_torch_safe_globals()

# ── SAYFA AYARI ──────────────────────────────────────────────
st.set_page_config(
    page_title="HasatVizyon Pro v2.5",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",   # mobilde kapalı başlar
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- TEMEL FONT ---- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

/* ---- ANA ARKA PLAN ---- */
.stApp {
    background: #f4faf4 !important;
}

/* ---- SİDEBAR ---- */
[data-testid="stSidebar"] {
    background: #1c3a1c !important;
}
/* sidebar tüm metinler */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #d4edda !important;
}
/* sidebar başlıklar */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a5d6a7 !important;
}
/* sidebar radio seçenekleri */
[data-testid="stSidebar"] .stRadio label span {
    color: #c8e6c9 !important;
    font-size: 0.95rem !important;
}
/* sidebar progress bar dolgu */
[data-testid="stSidebar"] [data-testid="stProgressBar"] > div {
    background: #4caf50 !important;
}
/* sidebar caption */
[data-testid="stSidebar"] .stCaption {
    color: #a5d6a7 !important;
}

/* ---- ANA ALAN YAZILARI ---- */
[data-testid="stMain"] p,
[data-testid="stMain"] span,
[data-testid="stMain"] label {
    color: #1a3a1a !important;
}
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #1b5e20 !important;
}

/* ---- DOSYA YÜKLEYİCİ: tüm durumlar için light override ---- */
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] > div,
[data-testid="stFileUploadDropzone"] section,
[data-testid="stFileUploadDropzone"] > div > div {
    background: #e8f5e9 !important;
    background-color: #e8f5e9 !important;
    border: 2px dashed #43a047 !important;
    border-radius: 12px !important;
}
/* Dropzone içindeki upload butonu */
[data-testid="stFileUploadDropzone"] button {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1px solid #81c784 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploadDropzone"] button span,
[data-testid="stFileUploadDropzone"] button p {
    color: #1b5e20 !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] div {
    color: #1b5e20 !important;
    font-weight: 600 !important;
    background: transparent !important;
}
/* Sistem dark mode'u Streamlit'in kendi file uploader'ını etkilemesin */
@media (prefers-color-scheme: dark) {
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploadDropzone"] > div,
    [data-testid="stFileUploadDropzone"] section {
        background: #e8f5e9 !important;
        background-color: #e8f5e9 !important;
    }
    [data-testid="stFileUploadDropzone"] button {
        background: #ffffff !important;
        background-color: #ffffff !important;
    }
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] small,
    [data-testid="stFileUploadDropzone"] div {
        color: #1b5e20 !important;
    }
}

/* ---- ANALİZ BUTONU ---- */
.stButton > button {
    background: linear-gradient(135deg, #2e7d32, #43a047) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
    width: 100% !important;
    height: 60px !important;
    white-space: pre-line !important;
    line-height: 1.3 !important;
    box-shadow: 0 4px 14px rgba(46,125,50,0.4) !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button p { color: #fff !important; white-space: pre-line !important; }

/* ---- EXPANDER ---- */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #c8e6c9 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #1b5e20 !important;
    font-weight: 700 !important;
}
[data-testid="stExpander"] p {
    color: #2d4a2d !important;
}

/* ---- SONUÇ KARTLARI ---- */
.res-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}
.res-card h3 {
    color: #1b5e20 !important;
    font-size: 1.2rem !important;
    margin: 0 0 4px 0 !important;
}
.res-card .conf {
    color: #4a7c59 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
}
.res-success { border-left: 5px solid #43a047; }
.res-danger  { border-left: 5px solid #e53935; }
.res-danger h3 { color: #b71c1c !important; }
.res-danger .conf { color: #c62828 !important; }

/* ---- REÇETE KUTUSU ---- */
.recipe-box {
    background: #f1fdf3;
    border: 1px solid #a5d6a7;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.recipe-box .rlabel {
    color: #1b5e20 !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 0 0 4px 0 !important;
}
.recipe-box .rval {
    color: #2d4a2d !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}

/* ---- BOŞ DURUM ALANI ---- */
.empty-box {
    background: #ffffff;
    border: 1.5px dashed #a5d6a7;
    border-radius: 14px;
    padding: 32px 24px;
    text-align: center;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.empty-box .etitle {
    color: #2e7d32 !important;
    font-weight: 700;
    font-size: 1rem;
    margin: 12px 0 6px 0 !important;
}
.empty-box .esub {
    color: #66bb6a !important;
    font-size: 0.85rem;
    margin: 0 !important;
}

/* ---- MOBİL ---- */
@media (max-width: 640px) {
    [data-testid="stMain"] { padding: 8px !important; }
    .res-card { padding: 12px 14px !important; }
}
</style>
""", unsafe_allow_html=True)


# ── YOLO MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model(selected_type):
    path = (
        "models/detection_model.pt"
        if "Algılama" in selected_type
        else "models/classification_model.pt"
    )
    if not os.path.exists(path):
        return None, f"Model dosyası bulunamadı: {path}"
    try:
        return YOLO(path), None
    except Exception as exc:
        return None, f"Model yükleme hatası: {exc}"


# ── KÜMELEME MODELİ (YOLO Backbone Tabanlı) ──────────────────
@st.cache_resource
def load_clustering_components():
    """K-Means model verilerini ve YOLO classification backbone'unu yükler.
    Ayrı bir ResNet18 gerekmez — kendi eğittiğimiz model kullanılır."""
    cluster_path  = "models/clustering_model.pkl"
    yolo_cls_path = "models/classification_model.pt"

    if not os.path.exists(cluster_path):
        return None, None, (
            "Kümeleme modeli bulunamadı. "
            "Önce terminalde `python train_clustering_model.py` çalıştırın."
        )
    if not os.path.exists(yolo_cls_path):
        return None, None, f"YOLO modeli bulunamadı: {yolo_cls_path}"

    try:
        data     = joblib.load(cluster_path)
        yolo     = YOLO(yolo_cls_path)
        net      = yolo.model
        backbone = torch.nn.Sequential(*list(net.model)[:-1])
        backbone.eval()
        return data, backbone, None
    except Exception as exc:
        return None, None, f"Kümeleme bileşenleri yüklenemedi: {exc}"


def extract_features_from_image(
    image: Image.Image,
    backbone: torch.nn.Module,
) -> np.ndarray:
    """YOLO backbone ile PIL görüntüsünden özellik vektörü çıkarır.
    YOLO'nun kendi ön-işlemi: sadece resize + [0,1] normalize (ImageNet stats YOK)."""
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),   # → [0,1] — ImageNet mean/std uygulanmaz
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = backbone(tensor)
        if out.dim() > 2:
            out = out.mean(dim=[2, 3])  # global average pool
    return out.squeeze().cpu().numpy()


def get_recommendation(class_name):
    rules = {
        "Powdery_Mildew":  {"teshis": "Külleme",              "icon": "🌫️", "korunma": "Dayanıklı tohum seçin, aşırı azotlu gübreden kaçının.",   "mudahale": "Sistemik fungisit (kükürt bazlı) uygulayın."},
        "Septoria":        {"teshis": "Septoria Yaprak Lekesi","icon": "🍂", "korunma": "Ekim nöbeti uygulayın, bitki artıklarını imha edin.",       "mudahale": "Triazol grubu fungisit ile ilaçlama yapın."},
        "Stem_Rust":       {"teshis": "Gövde Pası",            "icon": "🌾", "korunma": "Erken ekim, dayanıklı çeşit tercih edin.",                  "mudahale": "Pas karşıtı fungisit uygulayın."},
        "Yellow_Rust":     {"teshis": "Sarı Pas",              "icon": "🟡", "korunma": "Hava sirkülasyonu sağlayın.",                               "mudahale": "Acil fungisit müdahalesi gereklidir."},
        "BlackPoint":      {"teshis": "Kara Nokta",            "icon": "🌑", "korunma": "Uygun nem koşullarında depolayın.",                         "mudahale": "Tohum ilaçlaması ve nem kontrolü."},
        "FusariumFootRot": {"teshis": "Kök Çürüklüğü",        "icon": "📉", "korunma": "Derin sürüm, su baskınından kaçının.",                      "mudahale": "Koruyucu toprak fungisiti uygulayın."},
        "HealthyLeaf":     {"teshis": "Sağlıklı Yaprak",      "icon": "✅", "korunma": "Rutin kontrollere devam edin.",                             "mudahale": "Herhangi bir müdahale gerekmemektedir."},
        "LeafBlight":      {"teshis": "Yaprak Yanıklığı",     "icon": "🔥", "korunma": "Sık ekimden kaçının, sulama saatini optimize edin.",         "mudahale": "Sistemik fungisit ile ilaçlama yapın."},
        "WheatBlast":      {"teshis": "Buğday Yanıklığı",     "icon": "🚨", "korunma": "Sertifikalı, karantina onaylı tohum kullanın.",              "mudahale": "Acil ilaçlama + ziraat mühendisi desteği."},
        "wfd_dataset":     {"teshis": "Genel Hastalık",        "icon": "🔍", "korunma": "Tarla hijyenine dikkat edin.",                              "mudahale": "Ziraat mühendisi kontrolü alın."},
    }
    return rules.get(class_name)


# ── SİDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 HasatVizyon Pro")
    st.caption("Akıllı Tarım Karar Destek Sistemi")
    st.divider()

    model_choice = st.radio(
        "**Teşhis Modu**",
        options=[
            "🔍 Nesne Algılama (10 Sınıf)",
            "🎯 Sınıflandırma (5 Sınıf)",
            "🔵 Kümeleme (Yeni Görüntü)",
        ],
        index=0,
    )

    st.divider()

    if "Nesne" in model_choice:
        val, mode_lbl, class_cnt = 0.80, "Nesne Algılama", "10 sınıf"
    elif "Sınıf" in model_choice:
        val, mode_lbl, class_cnt = 0.99, "Sınıflandırma", "5 sınıf"
    else:
        val, mode_lbl, class_cnt = 0.88, "Kümeleme", "K-Means · 5 küme"

    st.markdown(f"**Model Başarısı:** `%{int(val*100)}`")
    st.progress(val)

    st.divider()
    st.info(f"⚡ **Aktif Model**\n\n{mode_lbl} · {class_cnt}")


# ── ANA EKRAN ────────────────────────────────────────────────
def run_clustering_mode():
    """Kümeleme modu: yeni/etiketlenmemiş görüntüyü K-Means ile analiz eder."""
    cluster_data, backbone, err = load_clustering_components()
    if err:
        st.error(f"⚠️ {err}")
        return

    st.markdown("## 🔵 Kümeleme ile Hastalık Analizi")
    st.caption(
        "Çiftçinin paylaştığı yeni / etiketlenmemiş görüntü, "
        "K-Means kümeleme ile en yakın hastalık grubuna atanır."
    )
    st.divider()

    col_L, col_R = st.columns([1, 1], gap="large")

    with col_L:
        st.markdown("### 📁 Görüntü Yükle")
        up_col, btn_col = st.columns([2.6, 1], gap="small")
        with up_col:
            uploaded_file = st.file_uploader(
                "PNG, JPG veya JPEG seçin",
                type=["png", "jpg", "jpeg"],
                key="cluster_uploader",
            )
        with btn_col:
            analyze_btn = False
            if uploaded_file:
                st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                analyze_btn = st.button("🔵 KÜME\nANALİZİ", key="cluster_btn")

        img_placeholder = st.empty()
        image = None
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            img_placeholder.image(image, caption="📷 Yüklenen görüntü", use_container_width=True)
        else:
            img_placeholder.markdown("""
<div class="empty-box">
    <div style="font-size:2.8rem">🌿</div>
    <p class="etitle">Görüntü bekleniyor</p>
    <p class="esub">Analiz için bir fotoğraf yükleyin</p>
</div>
""", unsafe_allow_html=True)

    with col_R:
        st.markdown("### 📋 Kümeleme Sonuçları")
        st.markdown("""
<div style="
    background: linear-gradient(135deg, #e3f2fd, #ede7f6);
    border: 1px solid #90caf9;
    border-left: 4px solid #1565c0;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 4px;
    margin-bottom: 14px;
">
    <p style="color:#0d47a1 !important; font-weight:700; font-size:0.82rem;
              text-transform:uppercase; letter-spacing:0.5px; margin:0 0 6px 0;">
        🔵 Kümeleme Nasıl Çalışır?
    </p>
    <p style="color:#1a237e !important; font-size:0.88rem; margin:0; line-height:1.6;">
        Görüntü <strong>ResNet18</strong> ile 512 boyutlu özellik vektörüne dönüştürülür.
        <strong>K-Means</strong> bu vektörü eğitim kümelerine atar.
        Hiçbir kümeyle yeterince örtüşmüyorsa <em>yeni hastalık varyantı</em> uyarısı verilir.
    </p>
</div>
""", unsafe_allow_html=True)

        if not uploaded_file:
            st.markdown("""
<div class="empty-box">
    <div style="font-size:2.8rem">🔵</div>
    <p class="etitle">Sonuçlar burada görünecek</p>
    <p class="esub">Sol taraftan görüntü yükleyip küme analizini başlatın</p>
</div>
""", unsafe_allow_html=True)

        elif uploaded_file and analyze_btn and image is not None:
            with st.spinner("🔬 YOLO backbone özellik çıkarıyor • K-Means atanıyor…"):
                feat        = extract_features_from_image(image, backbone)
                feat_scaled = cluster_data["scaler"].transform(feat.reshape(1, -1))
                kmeans       = cluster_data["kmeans"]
                centers      = kmeans.cluster_centers_
                threshold    = cluster_data["unknown_threshold"]

                # Tüm kümelere olan mesafeler
                dists        = np.linalg.norm(centers - feat_scaled, axis=1)  # (k,)
                sorted_idx   = np.argsort(dists)
                nearest_idx  = int(sorted_idx[0])
                nearest_dist = float(dists[nearest_idx])
                nearest_name = cluster_data["cluster_to_class"].get(nearest_idx, "Bilinmeyen")

                # Küme bazında eşik: mean + 2.5 * std (eğitim verisiyle tutarlı)
                c_stats  = cluster_data.get("cluster_stats", {})
                c_stat   = c_stats.get(nearest_idx, None)
                if c_stat:
                    adaptive_thresh = c_stat["mean"] + 2.5 * c_stat["std"]
                else:
                    adaptive_thresh = cluster_data["unknown_threshold"]
                is_unknown = nearest_dist > adaptive_thresh

                # Benzerlik skoru (0-100)
                similarity_pct = max(0.0, (1.0 - nearest_dist / (adaptive_thresh + 1e-8)) * 100)

            # ── SONUÇ KARTI ──────────────────────────────────────
            if is_unknown:
                st.markdown(f"""
<div class="res-card res-danger">
  <h3>⚠️ Yeni / Bilinmeyen Hastalık Varyantı</h3>
  <p class="conf">Bu görüntü hiçbir bilinen kümeyle yeterince örtüşmüyor.</p>
  <p class="conf" style="margin-top:6px">En yakın küme: <strong>{nearest_name}</strong>
     &nbsp;·&nbsp; Benzerlik: <strong>%{similarity_pct:.1f}</strong></p>
</div>
""", unsafe_allow_html=True)
                st.warning(
                    "🔬 Bu görüntü yeni bir hastalık varyantı veya nadir bir semptom içeriyor "
                    "olabilir. Ziraat mühendisine başvurmanız önerilir."
                )
            else:
                rec      = get_recommendation(nearest_name)
                icon     = rec["icon"]   if rec else "🔵"
                teshis   = rec["teshis"] if rec else nearest_name
                is_ok    = "Healthy" in nearest_name
                card_cls = "res-success" if is_ok else "res-danger"

                st.markdown(f"""
<div class="res-card {card_cls}" style="text-align:center">
  <div style="font-size:2.4rem;margin-bottom:8px">{icon}</div>
  <h3>Küme: {teshis}</h3>
  <p class="conf">Benzerlik Skoru: <strong>%{similarity_pct:.1f}</strong></p>
</div>
""", unsafe_allow_html=True)

                if rec:
                    st.markdown(f"""
<div class="recipe-box" style="margin-top:14px">
  <p class="rlabel">🛡️ Korunma</p>
  <p class="rval">{rec['korunma']}</p>
</div>
<div class="recipe-box">
  <p class="rlabel">💊 Müdahale</p>
  <p class="rval">{rec['mudahale']}</p>
</div>
""", unsafe_allow_html=True)

            # ── TOP-3 KÜME TABLOSU ────────────────────────────────
            st.markdown("---")
            st.markdown("**📊 En Yakın 3 Küme**")
            for rank, idx in enumerate(sorted_idx[:3]):
                name = cluster_data["cluster_to_class"].get(int(idx), "Bilinmeyen")
                d    = float(dists[idx])
                sim  = max(0.0, (1.0 - d / (adaptive_thresh + 1e-8)) * 100)
                bar_color = "#43a047" if rank == 0 else ("#fb8c00" if rank == 1 else "#e53935")
                st.markdown(f"""
<div style="display:flex;align-items:center;margin-bottom:8px;gap:10px">
  <span style="width:20px;font-weight:700;color:#555">#{rank+1}</span>
  <span style="flex:1;font-weight:600;color:#1b5e20">{name}</span>
  <div style="width:120px;background:#e0e0e0;border-radius:8px;height:10px">
    <div style="width:{min(sim,100):.0f}%;background:{bar_color};border-radius:8px;height:10px"></div>
  </div>
  <span style="width:50px;text-align:right;font-size:0.85rem;color:#555">%{sim:.0f}</span>
</div>
""", unsafe_allow_html=True)


def main():
    if "Kümeleme" in model_choice:
        run_clustering_mode()
        return

    model, model_error = load_model(model_choice)
    if not model:
        st.error(f"⚠️ {model_error or 'Model yüklenemedi!'}")
        return

    mode_tag = "Nesne Algılama" if "Nesne" in model_choice else "Sınıflandırma"

    st.markdown("## 🌱 Hububat Sağlık Teşhisi")
    st.caption(f"Yüklediğiniz bitki görüntüsü yapay zeka ile analiz edilir · **{mode_tag} Modu**")
    st.divider()

    # ── TEK KOLON ÇİFTİ: mobil sıralama korunur ─────────────────
    col_L, col_R = st.columns([1, 1], gap="large")

    # SOL KOLON: uploader + görsel
    with col_L:
        st.markdown("### 📁 Görüntü Yükle")
        up_col, btn_col = st.columns([2.6, 1], gap="small")
        with up_col:
            uploaded_file = st.file_uploader(
                "PNG, JPG veya JPEG seçin",
                type=["png", "jpg", "jpeg"],
            )
        with btn_col:
            analyze_btn = False
            if uploaded_file:
                st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                analyze_btn = st.button("🚀 ANALİZİ\nGERÇEKLEŞTİR")

        image = None
        image_placeholder = st.empty()

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            image_placeholder.image(
                image,
                caption="📷 Yüklenen görüntü",
                use_container_width=True,
            )
        else:
            image_placeholder.markdown("""
<div class="empty-box">
    <div style="font-size:2.8rem">🌿</div>
    <p class="etitle">Görüntü bekleniyor</p>
    <p class="esub">Analiz için bir fotoğraf yükleyin</p>
</div>
""", unsafe_allow_html=True)

    # SAĞ KOLON: başlık + bilgi kartı + sonuçlar
    with col_R:
        st.markdown("### 📋 Analiz Sonuçları")
        st.markdown("""
<div style="
    background: linear-gradient(135deg, #e8f5e9, #f1fdf3);
    border: 1px solid #a5d6a7;
    border-left: 4px solid #2e7d32;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 4px;
    margin-bottom: 14px;
">
    <p style="color:#1b5e20 !important; font-weight:700; font-size:0.82rem;
              text-transform:uppercase; letter-spacing:0.5px; margin:0 0 6px 0;">
        🤖 Nasıl Çalışır?
    </p>
    <p style="color:#2d4a2d !important; font-size:0.88rem; margin:0; line-height:1.6;">
        Yüklediğiniz bitki fotoğrafı derin öğrenme modeliyle analiz edilir.
        Hastalık tespit edilirse <strong>teşhis adı</strong>, <strong>güven oranı</strong>
        ve <strong>uygulama reçetesi</strong> burada listelenir.
    </p>
</div>
""", unsafe_allow_html=True)

        if not uploaded_file:
            st.markdown("""
<div class="empty-box">
    <div style="font-size:2.8rem">📊</div>
    <p class="etitle">Sonuçlar burada görünecek</p>
    <p class="esub">Sol taraftan görüntü yükleyip analizi başlatın</p>
</div>
""", unsafe_allow_html=True)

        elif uploaded_file and not analyze_btn:
            pass

        elif uploaded_file and analyze_btn and image is not None:
            with st.spinner("🔬 Yapay zeka analiz ediyor…"):
                results = model(image)
                result  = results[0]

                res_plotted = result.plot()
                annotated   = Image.fromarray(res_plotted[..., ::-1])
                image_placeholder.image(
                    annotated,
                    caption="🔬 Tespit sonucu",
                    use_container_width=True,
                )

            found = False

            # Detection
            if (hasattr(result, "boxes")
                    and result.boxes is not None
                    and len(result.boxes) > 0):
                found = True
                for box in result.boxes:
                    name = result.names[int(box.cls)]
                    conf = float(box.conf)
                    rec  = get_recommendation(name)
                    if not rec:
                        continue
                    is_ok    = "Healthy" in name
                    card_cls = "res-success" if is_ok else "res-danger"

                    st.markdown(f"""
<div class="res-card {card_cls}">
  <h3>{rec['icon']} {rec['teshis']}</h3>
  <p class="conf">Güven: <strong>%{conf*100:.1f}</strong></p>
</div>
""", unsafe_allow_html=True)

                    with st.expander("📝 Uygulama Reçetesi", expanded=True):
                        st.markdown(f"""
<div class="recipe-box">
  <p class="rlabel">🛡️ Korunma</p>
  <p class="rval">{rec['korunma']}</p>
</div>
<div class="recipe-box">
  <p class="rlabel">💊 Müdahale</p>
  <p class="rval">{rec['mudahale']}</p>
</div>
""", unsafe_allow_html=True)

            # Classification
            elif (hasattr(result, "probs")
                    and result.probs is not None):
                found     = True
                top_id    = result.probs.top1
                top_conf  = result.probs.top1conf.item()
                name      = result.names[top_id]
                rec       = get_recommendation(name)
                if rec:
                    is_ok    = "Healthy" in name
                    card_cls = "res-success" if is_ok else "res-danger"

                    st.markdown(f"""
<div class="res-card {card_cls}" style="text-align:center">
  <div style="font-size:2.4rem;margin-bottom:8px">{rec['icon']}</div>
  <h3>{rec['teshis']}</h3>
  <p class="conf">Doğruluk: <strong>%{top_conf*100:.2f}</strong></p>
</div>
""", unsafe_allow_html=True)

                    st.markdown(f"""
<div class="recipe-box" style="margin-top:14px">
  <p class="rlabel">🛡️ Korunma</p>
  <p class="rval">{rec['korunma']}</p>
</div>
<div class="recipe-box">
  <p class="rlabel">💊 Müdahale</p>
  <p class="rval">{rec['mudahale']}</p>
</div>
""", unsafe_allow_html=True)

            if not found:
                st.warning(
                    "⚠️ Belirgin bir hastalık semptomu saptanamadı. "
                    "Farklı açıdan çekilmiş bir fotoğraf deneyin."
                )


if __name__ == "__main__":
    main()
