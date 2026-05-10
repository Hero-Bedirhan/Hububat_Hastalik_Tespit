# 🌾 HasatVizyon Pro v2.5

HasatVizyon Pro, tarımsal üretimde karşılaşılan hububat hastalıklarını teşhis etmek amacıyla geliştirilmiş, hibrit yapay zeka mimarisine sahip gelişmiş bir **Karar Destek Sistemi (KDS)**'dir. Sistem, kullanıcıya iki farklı derin öğrenme yaklaşımını bir arada sunarak hem hızlı teşhis hem de detaylı görsel analiz imkanı sağlar.

## 🌗 Hibrit Model Mimarisi
Sistemimiz, kullanım amacına göre optimize edilmiş iki ana model üzerinde yükselmektedir:

### 1. Nesne Algılama Modeli (YOLOv8-Detection)
*   **Kapsam:** 10 Farklı Sınıf (8 Hastalık + 1 Genel Belirti + 1 Sağlıklı)
*   **Özellik:** Hastalık belirtilerini yaprak üzerinde **bounding box** ile işaretleyerek görsel kanıt sunar.
*   **Başarı:** %80.0 mAP50 (Yüksek çeşitlilikte üstün performans).
*   **Kullanım:** Detaylı arazi incelemesi ve hastalığın yayılım alanını görmek için idealdir.

### 2. Sınıflandırma Modeli (YOLOv8-Classification)
*   **Kapsam:** 5 Temel Sınıf (En yaygın ekonomik zararlı hastalıklar).
*   **Özellik:** Görüntünün tamamını analiz ederek anında teşhis koyar.
*   **Başarı:** %99.7 Accuracy (Kusursuza yakın kararlılık).
*   **Kullanım:** Çok hızlı ön teşhis ve genel bitki sağlığı kontrolü için idealdir.

## 🛠️ Teknolojik Altyapı
Neden bu teknolojileri seçtik?
*   **Python & PyTorch:** Modern yapay zeka dünyasının en güçlü ve esnek ikilisi.
*   **YOLOv8 (Ultralytics):** Sektör standardı hız ve doğruluk. RTX 5050 GPU ile 8ms çıkarım hızı.
*   **Streamlit Pro:** Responsive, mobil uyumlu ve şık bir son kullanıcı arayüzü.
*   **CUDA:** Donanım ivmelendirme ile gerçek zamanlı analiz kapasitesi.

## 📂 Proje Yapısı
*   `app.py`: "HasatVizyon Pro" hibrit kullanıcı arayüzü.
*   `models/detection_model.pt`: 10 sınıflı gelişmiş algılama ağırlıkları.
*   `models/classification_model.pt`: 5 sınıflı yüksek kararlılıklı sınıflandırma ağırlıkları.
*   `dataset/detection_dataset/`: 10 sınıflı birleştirilmiş veri seti (9,592 görüntü).
*   `dataset/classification_dataset/`: 5 sınıflı orijinal veri seti.
*   `runs/`: Her iki modelin eğitim metrikleri, loss grafikleri ve confusion matrix raporları.

## 🚀 Başlatma
```bash
pip install -r requirements.txt
streamlit run app.py
```

---
*Hazırlayan: Bedirhan | HasatVizyon AI - Sürdürülebilir Tarım İçin Teknoloji*
