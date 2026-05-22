import streamlit as st
import os

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
            ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel,
        )
        add_safe_globals([DetectionModel, ClassificationModel, SegmentationModel, PoseModel, OBBModel])
    except Exception:
        pass

_configure_torch_safe_globals()

st.set_page_config(
    page_title="HasatVizyon Pro v2.5",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    /* GENEL */
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #f0f9ff 100%) !important;
        min-height: 100vh;
    }

    /* SİDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2e1a 0%, #2d4a2d 100%) !important;
        border-right: 2px solid #4caf50 !important;
    }
    [data-testid="stSidebar"] * { color: #e8f5e9 !important; }
    [data-testid="stSidebar"] h1 { color: #69f0ae !important; }
    [data-testid="stSidebar"] .stRadio label {
        color: #c8e6c9 !important;
        font-size: 0.95rem !important;
        padding: 6px 0 !important;
    }
    [data-testid="stSidebar"] .stCaption p { color: #a5d6a7 !important; }
    [data-testid="stSidebar"] .stProgress > div > div {
        background: linear-gradient(90deg, #4caf50, #69f0ae) !important;
    }
    [data-testid="stSidebar"] .stProgress { background: #1b3a1b !important; border-radius: 8px; }

    /* ANA İÇERİK */
    [data-testid="stMain"] {
        background: transparent !important;
        color: #1a2e1a !important;
    }

    /* BAŞLIK */
    .main-header {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: clamp(1.6rem, 4vw, 2.4rem) !important;
        line-height: 1.2 !important;
    }
    .sub-header {
        color: #4a7c59 !important;
        font-size: clamp(0.85rem, 2vw, 1rem) !important;
        margin-top: -8px !important;
        margin-bottom: 20px !important;
    }

    /* UPLOAD KARTI */
    .upload-card {
        background: #ffffff;
        border: 2px dashed #66bb6a;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(46,125,50,0.08);
        transition: border-color 0.3s;
    }
    .upload-card:hover { border-color: #2e7d32; }

    /* DOSYA YÜKLEYİCİ YAZILARI */
    [data-testid="stFileUploadDropzone"] {
        background: #f1fdf3 !important;
        border: 2px dashed #81c784 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploadDropzone"] * { color: #2e7d32 !important; font-weight: 600 !important; }
    [data-testid="stFileUploadDropzone"] svg { fill: #4caf50 !important; }

    /* BUTON */
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #43a047) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        width: 100% !important;
        box-shadow: 0 6px 20px rgba(46,125,50,0.35) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.5px !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
        box-shadow: 0 8px 25px rgba(46,125,50,0.5) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button p { color: #ffffff !important; }

    /* SONUÇ ALANI BAŞLIĞI */
    .results-header {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: #ffffff !important;
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 15px rgba(46,125,50,0.3);
    }

    /* SONUÇ KARTI - BAŞARILI */
    .res-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.07);
        border: 1.5px solid #e8f5e9;
    }
    .res-card h2, .res-card h3 {
        color: #1b5e20 !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
        font-size: clamp(1.1rem, 3vw, 1.5rem) !important;
    }
    .res-card .conf-text {
        color: #4a7c59 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        margin: 0 !important;
    }
    .res-success { border-left: 6px solid #43a047 !important; }
    .res-danger  { border-left: 6px solid #e53935 !important; }
    .res-danger h2, .res-danger h3 { color: #c62828 !important; }

    /* REÇETE KARTLARI */
    .recipe-card {
        background: #f8fffe;
        border: 1.5px solid #c8e6c9;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .recipe-card .label {
        color: #1b5e20 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .recipe-card .value {
        color: #2d4a2d !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
    }
    .recipe-danger {
        background: #fff8f8;
        border-color: #ffcdd2;
    }

    /* EXPANDER */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1.5px solid #c8e6c9 !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary { color: #2e7d32 !important; font-weight: 700 !important; }
    [data-testid="stExpander"] p { color: #2d4a2d !important; font-weight: 500 !important; }

    /* WARNING / INFO kutuları */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* GENİŞ EKRAN: yükleme bölgesi tam yükseklikte */
    @media (min-width: 768px) {
        .upload-section { min-height: 480px; }
    }

    /* MOBİL RESPONSIVE */
    @media (max-width: 768px) {
        .main-header { font-size: 1.4rem !important; }
        .upload-card { padding: 14px !important; }
        .res-card { padding: 14px 16px !important; }
    }

    /* Genel metin renkleri */
    .stMarkdown p, .stMarkdown li { color: #1a2e1a !important; }
    h1, h2, h3, h4 { color: #1b5e20 !important; }
    p { color: #2d4a2d !important; }

    /* Bilgi çubuğu */
    .info-badge {
        display: inline-block;
        background: #e8f5e9;
        color: #1b5e20 !important;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid #a5d6a7;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)


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
        "Powdery_Mildew":    {"teshis": "Külleme",             "icon": "🌫️", "korunma": "Dayanıklı tohum seçimi yapın, aşırı azotlu gübreden kaçının.", "mudahale": "Sistemik fungisit (kükürt bazlı) uygulayın."},
        "Septoria":          {"teshis": "Septoria Yaprak Lekesi","icon": "🍂", "korunma": "Ekim nöbeti uygulayın, bitki artıklarını imha edin.",          "mudahale": "Triazol grubu fungisit ile ilaçlama yapın."},
        "Stem_Rust":         {"teshis": "Gövde Pası",           "icon": "🌾", "korunma": "Erken ekim, dayanıklı çeşit tercih edin.",                     "mudahale": "Pas karşıtı fungisit uygulayın."},
        "Yellow_Rust":       {"teshis": "Sarı Pas",             "icon": "🟡", "korunma": "İyi hava sirkülasyonu sağlayın.",                              "mudahale": "Acil fungisit müdahalesi gereklidir."},
        "BlackPoint":        {"teshis": "Kara Nokta",           "icon": "🌑", "korunma": "Uygun koşullarda depolama yapın.",                             "mudahale": "Tohum ilaçlaması ve nem kontrolü."},
        "FusariumFootRot":   {"teshis": "Kök Çürüklüğü",       "icon": "📉", "korunma": "Derin sürüm, su baskınından kaçının.",                         "mudahale": "Koruyucu toprak fungisiti uygulayın."},
        "HealthyLeaf":       {"teshis": "Sağlıklı Yaprak",     "icon": "✅", "korunma": "Rutin kontrollere devam edin.",                                "mudahale": "Herhangi bir müdahale gerekmemektedir."},
        "LeafBlight":        {"teshis": "Yaprak Yanıklığı",    "icon": "🔥", "korunma": "Sık ekimden kaçının, sulama saatini optimize edin.",            "mudahale": "Sistemik fungisit ile ilaçlama yapın."},
        "WheatBlast":        {"teshis": "Buğday Yanıklığı",    "icon": "🚨", "korunma": "Tohum karantinası, sertifikalı tohum kullanın.",               "mudahale": "Acil ilaçlama + tarım uzmanı desteği."},
        "wfd_dataset":       {"teshis": "Genel Hastalık",      "icon": "🔍", "korunma": "Tarla hijyenine dikkat edin.",                                 "mudahale": "Ziraat mühendisi kontrolü alın."},
    }
    return rules.get(class_name, None)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px 0;'>
            <div style='font-size:2.8rem;'>🌾</div>
            <h1 style='color:#69f0ae !important; font-size:1.4rem; margin:4px 0;'>HasatVizyon Pro</h1>
            <p style='color:#a5d6a7 !important; font-size:0.82rem; margin:0;'>Akıllı Tarım Karar Destek Sistemi</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2d4a2d; margin: 0 0 16px 0;'>", unsafe_allow_html=True)

    st.markdown("<p style='color:#81c784 !important; font-size:0.78rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;'>TEŞHİS MODU</p>", unsafe_allow_html=True)
    model_choice = st.radio(
        label="Teşhis Modu",
        options=["🔍 Nesne Algılama (10 Sınıf)", "🎯 Sınıflandırma (5 Sınıf)"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2d4a2d; margin: 16px 0;'>", unsafe_allow_html=True)

    val = 0.80 if "Nesne" in model_choice else 0.99
    pct = int(val * 100)
    st.markdown(f"""
        <p style='color:#81c784 !important; font-size:0.78rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;'>MODEL PERFORMANSI</p>
        <div style='background:#1b3a1b; border-radius:8px; height:10px; overflow:hidden; margin-bottom:6px;'>
            <div style='background:linear-gradient(90deg,#4caf50,#69f0ae); width:{pct}%; height:100%; border-radius:8px;'></div>
        </div>
        <p style='color:#a5d6a7 !important; font-size:0.82rem; text-align:right; margin:0;'>Başarı Oranı: <strong style="color:#69f0ae !important;">%{pct}</strong></p>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2d4a2d; margin: 16px 0;'>", unsafe_allow_html=True)

    mode_label = "Nesne Algılama" if "Nesne" in model_choice else "Sınıflandırma"
    class_count = "10 hastalık sınıfı" if "Nesne" in model_choice else "5 hastalık sınıfı"
    st.markdown(f"""
        <div style='background:#1b3a1b; border-radius:12px; padding:14px 16px;'>
            <p style='color:#69f0ae !important; font-size:0.85rem; font-weight:700; margin:0 0 6px 0;'>⚡ Aktif Model</p>
            <p style='color:#c8e6c9 !important; font-size:0.9rem; margin:0 0 4px 0;'>{mode_label}</p>
            <p style='color:#81c784 !important; font-size:0.78rem; margin:0;'>{class_count} tanımlı</p>
        </div>
    """, unsafe_allow_html=True)


# ---------- ANA EKRAN ----------
def main():
    model, model_error = load_model(model_choice)
    if not model:
        st.error(f"⚠️ {model_error or 'Model yüklenemedi!'}")
        return

    # Başlık
    mode_tag = "Nesne Algılama" if "Nesne" in model_choice else "Sınıflandırma"
    st.markdown(f"""
        <h1 class='main-header'>🌱 Hububat Sağlık Teşhisi</h1>
        <p class='sub-header'>Yüklediğiniz bitki görüntüsü yapay zeka ile analiz edilir · <strong>{mode_tag} Modu</strong></p>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#c8e6c9; margin-bottom:24px;'>", unsafe_allow_html=True)

    # Responsive kolonlar — mobilde tek, masaüstünde yan yana
    col_main, col_report = st.columns([1.1, 0.9], gap="large")

    with col_main:
        st.markdown("""
            <div style='background:#ffffff; border:2px dashed #66bb6a; border-radius:16px;
                        padding:6px 20px 20px 20px; box-shadow:0 4px 20px rgba(46,125,50,0.08);'>
                <p style='color:#2e7d32 !important; font-weight:700; font-size:0.95rem; margin:14px 0 8px 0;'>
                    📁 Görüntü Yükle
                </p>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Fotoğraf seçin (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        analyze_btn = False
        image = None
        image_placeholder = None

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            image_placeholder = st.empty()
            image_placeholder.image(image, caption="📷 Yüklenen Görüntü — Analiz bekleniyor", use_column_width=True)

            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            analyze_btn = st.button("🚀 ANALİZİ GERÇEKLEŞTİR")
        else:
            st.markdown("""
                <div style='text-align:center; padding:40px 20px; color:#4a7c59 !important;'>
                    <div style='font-size:3rem; margin-bottom:12px;'>🌿</div>
                    <p style='color:#4a7c59 !important; font-size:0.95rem; font-weight:600;'>
                        Analiz etmek istediğiniz bitki<br>fotoğrafını yukarıya yükleyin
                    </p>
                    <p style='color:#81c784 !important; font-size:0.8rem;'>PNG · JPG · JPEG desteklenir</p>
                </div>
            """, unsafe_allow_html=True)

    with col_report:
        if uploaded_file and analyze_btn and image is not None:
            with st.spinner("🔬 Yapay zeka analiz ediyor..."):
                results = model(image)
                result = results[0]

                res_plotted = result.plot()
                annotated = Image.fromarray(res_plotted[..., ::-1])
                if image_placeholder:
                    image_placeholder.image(annotated, caption="🔬 Tespit Sonucu — İşaretlenmiş Görüntü", use_column_width=True)

            st.markdown("""
                <div class='results-header'>
                    📋 Analiz Bulguları
                </div>
            """, unsafe_allow_html=True)

            found = False

            # --- DETECTION ---
            if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                found = True
                for box in result.boxes:
                    name = result.names[int(box.cls)]
                    conf = float(box.conf)
                    rec = get_recommendation(name)
                    if rec:
                        is_healthy = "Healthy" in name
                        card_cls = "res-success" if is_healthy else "res-danger"
                        conf_color = "#2e7d32" if is_healthy else "#c62828"

                        st.markdown(f"""
                            <div class='res-card {card_cls}'>
                                <h3>{rec['icon']} {rec['teshis']}</h3>
                                <p class='conf-text' style='color:{conf_color} !important;'>
                                    Güven Seviyesi: <strong>%{conf*100:.1f}</strong>
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                            <div class='recipe-card'>
                                <p class='label'>🛡️ Korunma Yöntemi</p>
                                <p class='value'>{rec['korunma']}</p>
                            </div>
                            <div class='recipe-card recipe-danger'>
                                <p class='label'>💊 Müdahale</p>
                                <p class='value'>{rec['mudahale']}</p>
                            </div>
                        """, unsafe_allow_html=True)

            # --- CLASSIFICATION ---
            elif hasattr(result, "probs") and result.probs is not None:
                found = True
                top_id = result.probs.top1
                top_conf = result.probs.top1conf.item()
                name = result.names[top_id]
                rec = get_recommendation(name)
                if rec:
                    is_healthy = "Healthy" in name
                    card_cls = "res-success" if is_healthy else "res-danger"
                    conf_color = "#2e7d32" if is_healthy else "#c62828"

                    st.markdown(f"""
                        <div class='res-card {card_cls}' style='text-align:center;'>
                            <div style='font-size:2.5rem; margin-bottom:8px;'>{rec['icon']}</div>
                            <h2>{rec['teshis']}</h2>
                            <p class='conf-text' style='color:{conf_color} !important; font-size:1rem !important;'>
                                Doğruluk Payı: <strong>%{top_conf*100:.2f}</strong>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                        <div class='recipe-card' style='margin-top:16px;'>
                            <p class='label'>🛡️ Korunma Yöntemi</p>
                            <p class='value'>{rec['korunma']}</p>
                        </div>
                        <div class='recipe-card recipe-danger'>
                            <p class='label'>💊 Müdahale</p>
                            <p class='value'>{rec['mudahale']}</p>
                        </div>
                    """, unsafe_allow_html=True)

            if not found:
                st.markdown("""
                    <div style='background:#fff8e1; border:1.5px solid #ffe082; border-radius:12px; padding:20px; text-align:center;'>
                        <div style='font-size:2rem; margin-bottom:8px;'>⚠️</div>
                        <p style='color:#f57f17 !important; font-weight:700; margin:0;'>Belirgin bir hastalık semptomu saptanamadı.</p>
                        <p style='color:#f9a825 !important; font-size:0.85rem; margin:8px 0 0 0;'>Farklı açıdan çekilmiş bir fotoğraf deneyin.</p>
                    </div>
                """, unsafe_allow_html=True)

        elif not uploaded_file:
            st.markdown("""
                <div style='background:#ffffff; border:1.5px solid #e8f5e9; border-radius:16px;
                            padding:48px 24px; text-align:center; box-shadow:0 4px 16px rgba(0,0,0,0.05);'>
                    <div style='font-size:3rem; margin-bottom:16px; opacity:0.5;'>📊</div>
                    <p style='color:#4a7c59 !important; font-weight:700; font-size:1rem; margin:0 0 8px 0;'>
                        Analiz Sonuçları Burada Görünecek
                    </p>
                    <p style='color:#81c784 !important; font-size:0.85rem; margin:0;'>
                        Sol taraftan bir görüntü yükleyin ve analizi başlatın.
                    </p>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
