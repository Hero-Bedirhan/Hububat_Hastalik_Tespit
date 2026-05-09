import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO

# Sayfa yapılandırması
st.set_page_config(
    page_title="Hububat Hastalık Tespit ve Karar Destek Sistemi",
    page_icon="🌾",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = "/home/bedirhan/Desktop/Vs_Code/Git/Hububat_Hastalik_Tespit/model/bugday_model.pt"
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
        "BlackPoint": {
            "teshis": "Kara Nokta (Black Point)",
            "korunma": "Hasat döneminde yağışlı ortamlardan kaçınılmalı, dayanıklı çeşitler seçilmeli.",
            "mudahale": "Tohum ilaçlaması yapılmalı, ürün uygun nem koşullarında depolanmalıdır."
        },
        "FusariumFootRot": {
            "teshis": "Kök ve Kökboğazı Çürüklüğü (Fusarium Foot Rot)",
            "korunma": "Derin sürüm yapılmalı ve ekim nöbetine dikkat edilmeli.",
            "mudahale": "Kimyasal mücadelesi zordur; koruyucu olarak tohum ilaçlaması şarttır."
        },
        "HealthyLeaf": {
            "teshis": "Sağlıklı Yaprak",
            "korunma": "Rutin bakım ve gübrelemeye devam ediniz.",
            "mudahale": "Herhangi bir ilaçlamaya gerek yoktur."
        },
        "LeafBlight": {
            "teshis": "Yaprak Yanıklığı (Leaf Blight)",
            "korunma": "Sık ekimden kaçınılmalı, aşırı sulama yapılmamalıdır.",
            "mudahale": "Belirtiler ilk görüldüğünde uygun sistemik fungisitler uygulanmalıdır."
        },
        "WheatBlast": {
            "teshis": "Buğday Yanıklığı (Wheat Blast)",
            "korunma": "Hastalıklı bölgelerden tohum alınmamalı, sertifikalı tohum kullanılmalıdır.",
            "mudahale": "Hastalık çok hızlı yayılır, acil olarak geniş spektrumlu fungisitlerle müdahale edilmelidir."
        }
    }
    
    for key in rules.keys():
        if key.lower() in class_name.lower():
            return rules[key]
    
    return None

def main():
    st.title("🌾 Hububat Hastalık Tespit ve Karar Destek Sistemi")
    st.markdown("Bu sistem, tarımsal hububat yetiştiriciliğinde hastalıkları tespit etmek ve uygun tedavi/korunma yöntemlerini sunmak için tasarlanmıştır. (YOLO Sınıflandırma Modeli)")

    uploaded_file = st.file_uploader(
        "Lütfen analiz için bir yaprak fotoğrafı yükleyin (.jpg, .png)", 
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_placeholder = st.empty()
        image_placeholder.image(image, caption='Yüklenen Görüntü', use_column_width=True)

        if st.button("Analiz Yap"):
            with st.spinner('Yerel sınıflandırma modeli ile analiz ediliyor, lütfen bekleyin...'):
                try:
                    # YOLO modelini kullanarak sınıflandırma tahmini
                    results = model(image)
                    result = results[0] 
                    
                    if result.probs is None:
                        st.warning("Görüntü analiz edilemedi. Sınıflandırma sonucu bulunamadı.")
                    else:
                        # Sonuçları görselleştirme
                        res_plotted = result.plot()
                        annotated_image = Image.fromarray(res_plotted[..., ::-1])
                        image_placeholder.image(annotated_image, caption='Analiz Sonucu (Sınıf ve Güven Skoru)', use_column_width=True)

                        st.subheader("📊 Karar Destek Sistemi Analiz Sonucu")
                        
                        # Classification objesinden en yüksek güven skorunu çekme
                        top_class_id = result.probs.top1
                        top_confidence = result.probs.top1conf.item()
                        class_name = result.names[top_class_id]

                        rec = get_recommendation(class_name)
                        
                        if rec:
                            if "Healthy" in class_name:
                                st.success(f"Teşhis: {rec['teshis']} (Güven Skoru: %{top_confidence*100:.2f})")
                                st.info(f"🛡️ **Korunma:** {rec['korunma']}")
                                st.info(f"🌿 **Müdahale:** {rec['mudahale']}")
                            else:
                                st.error(f"Teşhis: {rec['teshis']} (Güven Skoru: %{top_confidence*100:.2f})")
                                st.warning(f"🛡️ **Korunma:** {rec['korunma']}")
                                st.warning(f"💊 **Müdahale:** {rec['mudahale']}")
                        else:
                            st.warning(f"Bilinmeyen sınıf tespit edildi: {class_name} (Güven Skoru: %{top_confidence*100:.2f})")

                except Exception as e:
                    st.error(f"Beklenmeyen Bir Hata Oluştu: {e}")

if __name__ == "__main__":
    main()
