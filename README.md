# 🌾 Hububat Hastalık Tespit ve Karar Destek Sistemi

Bu proje, tarımsal üretimde verimliliği artırmak ve ekin kayıplarını minimize etmek amacıyla geliştirilmiş, yapay zeka tabanlı bir hastalık teşhis sistemidir. Sistem, buğday yapraklarındaki hastalıkları derin öğrenme algoritmaları kullanarak yüksek doğrulukla analiz eder ve çiftçilere yönelik tedavi/korunma önerileri sunar.

## 📊 Model ve Başarı Oranı
Proje kapsamında **YOLOv8-cls (Sınıflandırma)** mimarisi kullanılmıştır. Yapılan eğitimler sonucunda model, test verileri üzerinde **%99.76** gibi olağanüstü bir doğruluk oranına ulaşmıştır.

- **Mimari:** YOLOv8s-Classification
- **Doğruluk (Accuracy):** %99.76
- **Kayıp (Loss):** 0.0049 (6. Epoch sonunda)

## 📁 Veri Seti Bilgileri
Eğitim sürecinde [Roboflow Wheat Disease Detection](https://universe.roboflow.com/agliculture/wheat-disease-detection-zsn0p) veri seti kullanılmıştır. Veri seti, buğdayda en sık görülen ve ekonomik zarara yol açan hastalıkları kapsayan binlerce görüntüden oluşmaktadır.

### Desteklenen Hastalık Sınıfları
1.  **BlackPoint (Kara Nokta):** Tanelerde ve yapraklarda koyu renkli bozulmalar.
2.  **FusariumFootRot (Kök/Kökboğazı Çürüklüğü):** Bitki gövde tabanında çürüme.
3.  **LeafBlight (Yaprak Yanıklığı):** Yaprak yüzeyinde geniş kurumuş alanlar.
4.  **WheatBlast (Buğday Yanıklığı):** Başak ve yapraklarda hızlı yayılan tehlikeli yanıklık.
5.  **HealthyLeaf (Sağlıklı Yaprak):** Kontrol grubu, hastalık belirtisi olmayan ekinler.

## 🚀 Karar Destek Sistemi Özellikleri
Uygulama sadece bir teşhis aracı değil, aynı zamanda bir **Karar Destek Sistemi (KDS)** olarak çalışır:
- **Anlık Analiz:** Yüklenen fotoğraf saniyeler içinde işlenir.
- **Teşhis Raporu:** Hastalığın adı ve güven skoru (%) belirtilir.
- **🛡️ Korunma Önerileri:** Hastalığın bulaşmasını önlemek için kültürel önlemler.
- **💊 Müdahale Planı:** Tespit edilen hastalığa uygun fungisit ve tedavi yöntemleri.

## 🛠️ Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.8+
- Sanal ortam (venv) önerilir.

### Adımlar
1.  **Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Uygulamayı Başlatın:**
    ```bash
    streamlit run app.py
    ```

## 📂 Proje Yapısı
- `app.py`: Streamlit web arayüzü ve KDS motoru.
- `model/bugday_model.pt`: %99.7 başarılı eğitilmiş model ağırlıkları.
- `bugday_hasta/`: Eğitimde kullanılan veri seti yapısı.
- `runs/bugday_hasta_final/`: Eğitim istatistikleri ve performans grafikleri.

---
*Bu çalışma, akıllı tarım teknolojileri ve derin öğrenmenin gücünü birleştirerek sürdürülebilir tarıma katkı sağlamayı amaçlamaktadır.*
