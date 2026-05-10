import streamlit as st
import os

# Trusted local checkpoint files are loaded with legacy behavior for PyTorch 2.6+.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from PIL import Image
from ultralytics import YOLO
import numpy as np
import torch


_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_compat(*args, **kwargs):
    """Keep backward-compatible checkpoint loading for trusted local models."""
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_compat


def _configure_torch_safe_globals():
    """Allow trusted Ultralytics model classes for PyTorch 2.6+ safe loading."""
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    try:
        from ultralytics.nn.tasks import (
            ClassificationModel,
            DetectionModel,
            OBBModel,
            PoseModel,
            SegmentationModel,
        )

        add_safe_globals(
            [
                DetectionModel,
                ClassificationModel,
                SegmentationModel,
                PoseModel,
                OBBModel,
            ]
        )
    except Exception:
        # If internals change between versions, fall back to default behavior.
        pass


_configure_torch_safe_globals()

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="HasatVizyon Pro v2.5",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ÖZEL CSS (Sağ Panel Beyaz, Sol Panel Koyu) ---
st.markdown("""
    <style>
    /* 1. GENEL ARKA PLAN */
    .stApp {
        background-color: #ffffff !important;
    }

    /* 2. SOL PANEL (Sidebar - Koyu Kalacak) */
    [data-testid="stSidebar"] {
        background-color: #2b3035 !important;
        border-right: 1px solid #1a1d20;
    }
    [data-testid="stSidebar"] * {
        color: #f8f9fa !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #f8f9fa !important;
    }

    /* 3. SAĞ PANEL (Ana İçerik - Beyaz) */
    [data-testid="stMain"] {
        background-color: #ffffff !important;
        color: #212529 !important;
    }

    /* 4. ANA EKRAN BAŞLIĞI */
    .main-header {
        color: #2e7d32 !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
    }

    /* 5. ANALİZ SONUÇ KARTLARI (BEYAZ ARKA PLAN - KOYU YAZI) */
    .res-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .res-card h2, .res-card h3, .res-card h4 {
        color: #1b5e20 !important;
        margin: 0 !important;
    }
    .res-card p {
        color: #495057 !important;
        font-weight: 600 !important;
        margin-top: 5px !important;
    }
    
    /* Başarı/Tehlike Durumları için Renkli Kenarlık */
    .res-success { border-left: 8px solid #2e7d32 !important; }
    .res-danger { border-left: 8px solid #c62828 !important; }

    /* 6. REÇETE VE DİĞER YAZILAR */
    .stMarkdown, .stExpander, p, h3 {
        color: #000000 !important;
    }
    
    [data-testid="stMain"] h3 {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* DOSYA YÜKLEME METİNLERİ (Siyah) */
    [data-testid="stFileUploadDropzone"] div, 
    [data-testid="stFileUploadDropzone"] span, 
    [data-testid="stFileUploadDropzone"] small {
        color: #000000 !important;
    }
    
    /* 7. BUTON */
    .stButton>button {
        background: linear-gradient(to right, #2e7d32, #1b5e20) !important;
        border-radius: 12px;
        border: none;
        padding: 12px;
        width: 100%;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2);
    }
    .stButton>button, .stButton>button p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Kart Yapısı */
    .stCard {
        background: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_model(selected_type):
    if "Algılama" in selected_type:
        path = "models/detection_model.pt"
    else:
        path = "models/classification_model.pt"
    if not os.path.exists(path):
        return None, f"Model dosyası bulunamadı: {path}"

    try:
        return YOLO(path), None
    except Exception as exc:
        return None, f"Model yükleme hatası: {exc}"

def get_recommendation(class_name):
    rules = {
        "Powdery_Mildew": {"teshis": "Külleme (Powdery Mildew)", "icon": "🌫️", "korunma": "Dayanıklı tohum seçimi.", "mudahale": "Sistemik fungisit."},
        "Septoria": {"teshis": "Septoria Yaprak Lekesi", "icon": "🍂", "korunma": "Ekim nöbeti.", "mudahale": "İlaçlama."},
        "Stem_Rust": {"teshis": "Gövde Pası (Stem Rust)", "icon": "🌾", "korunma": "Erken ekim.", "mudahale": "Pas ilaçları."},
        "Yellow_Rust": {"teshis": "Sarı Pas (Yellow Rust)", "icon": "🟡", "korunma": "Hava sirkülasyonu.", "mudahale": "Acil müdahale."},
        "BlackPoint": {"teshis": "Kara Nokta (Black Point)", "icon": "🌚", "korunma": "Uygun depolama.", "mudahale": "Tohum ilaçlaması."},
        "FusariumFootRot": {"teshis": "Kök Çürüklüğü", "icon": "📉", "korunma": "Derin sürüm.", "mudahale": "Koruyucu ilaçlama."},
        "HealthyLeaf": {"teshis": "Sağlıklı Yaprak", "icon": "✅", "korunma": "Rutin kontrol.", "mudahale": "Gerek yok."},
        "LeafBlight": {"teshis": "Yaprak Yanıklığı", "icon": "🔥", "korunma": "Sık ekimden kaçınma.", "mudahale": "Sistemik fungisit."},
        "WheatBlast": {"teshis": "Buğday Yanıklığı (Wheat Blast)", "icon": "🚨", "korunma": "Tohum karantinası.", "mudahale": "Acil ilaçlama."},
        "wfd_dataset": {"teshis": "Genel Hastalık", "icon": "🔍", "korunma": "Tarla hijyeni.", "mudahale": "Mühendis kontrolü."}
    }
    return rules.get(class_name, None)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #4caf50 !important;'>🌾 HasatVizyon Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #ced4da !important;'>Akıllı Tarım Karar Destek</p>", unsafe_allow_html=True)
    st.markdown("---")
    model_choice = st.radio("Teşhis Modu Seçin:", options=["🔍 Nesne Algılama (10 Sınıf)", "🎯 Sınıflandırma (5 Sınıf)"], index=0)
    st.markdown("---")
    st.write("📈 **Model Verimliliği**")
    val = 0.80 if "Nesne" in model_choice else 0.99
    st.progress(val)
    st.caption(f"Aktif Başarı Oranı: %{val*100:.1f}")

# --- ANA EKRAN ---
def main():
    model, model_error = load_model(model_choice)
    if not model:
        st.error(model_error or "Model yüklenemedi!")
        return

    st.markdown("<h2 class='main-header'>🌱 Hububat Sağlık Teşhisi</h2>", unsafe_allow_html=True)
    st.markdown("---")

    col_main, col_report = st.columns([1.1, 0.9], gap="large")

    with col_main:
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Dosya seçin", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
        
        analyze_btn = False
        if uploaded_file:
            analyze_btn = st.button("🚀 ANALİZİ GERÇEKLEŞTİR")
            image = Image.open(uploaded_file).convert("RGB")
            image_placeholder = st.empty()
            image_placeholder.image(image, caption="Analiz Bekleniyor...", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_report:
        if uploaded_file and analyze_btn:
            with st.spinner('🔬 Analiz ediliyor...'):
                results = model(image)
                result = results[0]
                
                res_plotted = result.plot()
                annotated_image = Image.fromarray(res_plotted[..., ::-1])
                image_placeholder.image(annotated_image, caption="İşaretlenmiş Veri", use_column_width=True)
                
                st.markdown("### 📋 Analiz Bulguları")
                
                found = False
                # Detection
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    found = True
                    for box in result.boxes:
                        name = result.names[int(box.cls)]
                        conf = float(box.conf)
                        rec = get_recommendation(name)
                        if rec:
                            cl = "res-success" if "Healthy" in name else "res-danger"
                            st.markdown(f"<div class='res-card {cl}'><h3>{rec['icon']} {rec['teshis']}</h3><p>Güven Seviyesi: %{conf*100:.1f}</p></div>", unsafe_allow_html=True)
                            with st.expander("📝 Uygulama Reçetesi", expanded=True):
                                st.write(f"🛡️ **Korunma:** {rec['korunma']}")
                                st.write(f"💊 **Müdahale:** {rec['mudahale']}")
                
                # Classification
                elif hasattr(result, 'probs') and result.probs is not None:
                    found = True
                    top_id = result.probs.top1
                    top_conf = result.probs.top1conf.item()
                    name = result.names[top_id]
                    rec = get_recommendation(name)
                    if rec:
                        cl = "res-success" if "Healthy" in name else "res-danger"
                        st.markdown(f"<div class='res-card {cl}' style='text-align:center;'><h2>{rec['icon']} {rec['teshis']}</h2><h4>Doğruluk Payı: %{top_conf*100:.2f}</h4></div>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.info(f"🛡️ **Korunma:** {rec['korunma']}")
                        st.warning(f"💊 **Müdahale:** {rec['mudahale']}")

                if not found:
                    st.warning("⚠️ Belirti saptanamadı.")

if __name__ == "__main__":
    main()
