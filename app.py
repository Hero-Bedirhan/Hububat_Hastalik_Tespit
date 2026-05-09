import streamlit as st
import tempfile
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Sayfa yapılandırması
st.set_page_config(
    page_title="Hububat Hastalık Tespit ve Karar Destek Sistemi",
    page_icon="🌾",
    layout="centered"
)

# Eğitilen modelin yolunu belirliyoruz. 
# st.cache_resource kullanarak modelin her etkileşimde yeniden yüklenmesini önlüyoruz (hız kazandırır).
@st.cache_resource
def load_model():
    model_path = "/home/bedirhan/Desktop/Hububat_Hastalik_Tespit/runs/detect/train-4/weights/best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model dosyası bulunamadı: {model_path}. Lütfen eğitimin tamamlandığından emin olun.")
        st.stop()
    return YOLO(model_path)

model = load_model()

def get_recommendation(class_name):
    """
    Sınıflandırma sonucuna göre kural tabanlı tavsiye (birliktelik/iş kuralları) motoru.
    """
    rules = {
        "Yellow Rust": {
            "teshis": "Sarı Pas Hastalığı",
            "korunma": "Erken ekimden kaçınılmalı, sık ekim yapılmamalıdır.",
            "mudahale": "Triazol grubu fungisitler ile yeşil aksam ilaçlaması yapılmalıdır."
        },
        "Brown Rust": {
            "teshis": "Kahverengi Pas",
            "korunma": "Dayanıklı çeşitler seçilmeli, ara konukçu bitkiler yok edilmelidir.",
            "mudahale": "Sistemik mantar ilaçları uygulanmalı, potasyum takviyesi verilmelidir."
        },
        "Black Rust": {
            "teshis": "Kara Pas",
            "korunma": "Ara konukçu olan kadın tuzluğu bitkileri uzaklaştırılmalıdır.",
            "mudahale": "Hastalık şiddeti artmadan uygun fungisitlerle ilaçlamaya başlanmalıdır."
        },
        "Stem Rust": {
            "teshis": "Kök Pası (Stem Rust)",
            "korunma": "Ara konukçu kadın tuzluğu uzaklaştırılmalı, dayanıklı tohum kullanılmalıdır.",
            "mudahale": "Hastalık başlangıcında uygun sistemik fungisitler uygulanmalıdır."
        },
        "Powdery Mildew": {
            "teshis": "Külleme Hastalığı",
            "korunma": "Sık ekimden ve aşırı azotlu gübrelemeden kaçınılmalıdır.",
            "mudahale": "Hastalık belirtileri görülür görülmez kükürtlü veya triazol grubu ilaçlar kullanılmalıdır."
        },
        "Septoria": {
            "teshis": "Septoria Yaprak Lekesi",
            "korunma": "Ekim nöbeti (rotasyon) uygulanmalı, hastalıklı bitki artıkları tarladan uzaklaştırılmalıdır.",
            "mudahale": "Yağışlı dönemlerde koruyucu amaçlı fungisit uygulaması yapılmalıdır."
        },
        "Healthy Wheat": {
            "teshis": "Bitki Sağlıklı",
            "korunma": "Rutin bakımlara devam edilmelidir.",
            "mudahale": "İlaçlamaya gerek yoktur."
        }
    }
    # Eğer model sınıf isimlerini küçük harf veya farklı formatta döndürürse diye bir kontrol:
    for key in rules.keys():
        if key.lower().replace("-", " ") in class_name.lower().replace("-", " "):
            return rules[key]
    
    return rules.get(class_name, None)

def main():
    st.title("🌾 Hububat Hastalık Tespit ve Karar Destek Sistemi")
    st.markdown("Bu sistem, tarımsal hububat yetiştiriciliğinde ekin yapraklarındaki hastalıkları tespit etmek ve uygun tedavi/korunma yöntemlerini sunmak için tasarlanmıştır. (Yerel YOLOv8 Modeli)")

    # Kullanıcıdan görüntü alma alanı
    uploaded_file = st.file_uploader(
        "Lütfen analiz için bir yaprak fotoğrafı yükleyin (.jpg, .png)", 
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        # Görüntüyü yükleme
        image = Image.open(uploaded_file).convert("RGB")
        
        # Orijinal resmi gösterme alanı
        image_placeholder = st.empty()
        image_placeholder.image(image, caption='Yüklenen Görüntü', use_container_width=True)

        if st.button("Analiz Yap"):
            with st.spinner('Yerel model ile analiz ediliyor, lütfen bekleyin...'):
                try:
                    # YOLO modelini kullanarak tahmin işlemi
                    results = model(image)
                    result = results[0] # İlk resim (zaten 1 resim verdik)
                    
                    if len(result.boxes) == 0:
                        st.warning("Görüntüde belirgin bir hastalık sınıfı tespit edilemedi.")
                    else:
                        # Ultralytics'in kendi çizim metodunu kullanıyoruz (plot). 
                        # Numpy dizisi olarak BGR formatında döner, bunu RGB'ye çevirip PIL formatına alıyoruz.
                        res_plotted = result.plot()
                        annotated_image = Image.fromarray(res_plotted[..., ::-1])  # BGR to RGB
                        
                        # Orijinal resim yerine işaretlenmiş resmi koy
                        image_placeholder.image(annotated_image, caption='Tespit Edilen Hastalık Bölgeleri (İşaretlenmiş)', use_container_width=True)

                        st.subheader("📊 Analiz Sonucu")
                        
                        # Tespit edilen kutulardan en yüksek güven skoruna sahip olanı bulma
                        boxes = result.boxes
                        confidences = boxes.conf.cpu().numpy()
                        class_ids = boxes.cls.cpu().numpy()
                        
                        # En yüksek skorun indeksini buluyoruz
                        max_conf_idx = np.argmax(confidences)
                        
                        top_class_id = int(class_ids[max_conf_idx])
                        top_confidence = confidences[max_conf_idx]
                        
                        # Sınıf ID'sini isme çevirme (model.names sözlüğünü kullanarak)
                        class_name = result.names[top_class_id]

                        # Karar Destek Motoru üzerinden tavsiye kurallarını çekme
                        rec = get_recommendation(class_name)
                        
                        if rec:
                            # Teşhise özel Streamlit görsel bileşenlerinin kullanımı
                            if "Healthy" in class_name or "Healthy Wheat" == class_name:
                                st.success(f"Teşhis: {rec['teshis']} (En Yüksek Güven Skoru: %{top_confidence*100:.2f})")
                                st.info(f"🛡️ **Korunma:** {rec['korunma']}")
                                st.info(f"🌿 **Müdahale:** {rec['mudahale']}")
                            else:
                                st.error(f"Teşhis: {rec['teshis']} (En Yüksek Güven Skoru: %{top_confidence*100:.2f})")
                                st.warning(f"🛡️ **Korunma:** {rec['korunma']}")
                                st.warning(f"💊 **Müdahale:** {rec['mudahale']}")
                        else:
                            st.warning(f"Bilinmeyen sınıf tespit edildi: {class_name} (Güven Skoru: %{top_confidence*100:.2f})")

                except Exception as e:
                    st.error(f"Beklenmeyen Bir Hata Oluştu: {e}")

if __name__ == "__main__":
    main()