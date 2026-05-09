# Akıllı Tarım: Hububat Hastalık Tespiti ve Karar Destek Sistemi

## Yönetici Özeti
Bu proje, tarımsal verimliliği artırmak ve ürün kayıplarını en aza indirmek amacıyla geliştirilmiş, yapay zeka tabanlı bir **Yönetim Bilişim Sistemi (YBS)** ve **Karar Destek Sistemi (KDS)** uygulamasıdır. Arpa, buğday gibi hububatlarda sıkça görülen pas hastalıkları ve diğer yaprak lezyonları, gelişmiş bir bilgisayarlı görü modeli (YOLOv8) kullanılarak tespit edilmekte ve sistem içerisine gömülü veri madenciliği tabanlı kural motoru sayesinde çiftçilere anlık tedavi, ilaçlama ve korunma yöntemleri sunulmaktadır.

---

## Veri Madenciliği Aşamaları (CRISP-DM Metodolojisi)

Projenin geliştirilmesinde, endüstri standardı olan Veri Madenciliği Süreci **CRISP-DM** (Cross-Industry Standard Process for Data Mining) metodolojisi referans alınmıştır.

### 1. Problemin Anlaşılması (Business Understanding)
Tarım sektöründe bitki hastalıklarının geç fark edilmesi, yanlış veya gereksiz pestisit kullanımına ve dolayısıyla hem ekonomik kayıplara hem de ekolojik hasara yol açmaktadır. Bu projenin temel iş problemi; çiftçilerin sahada karşılaştıkları yaprak anormalliklerini bir uzman beklemeden saniyeler içinde teşhis edebilmesini sağlamak ve yanlış müdahaleleri önleyecek doğru tarımsal tavsiyeleri üretmektir.

### 2. Verinin Anlaşılması ve Hazırlanması
Modelin hastalıkları doğru öğrenebilmesi için, gerçek saha koşullarını yansıtan açık kaynaklı tarımsal görüntü veri setleri kullanılmıştır. Veriler YOLOv8 nesne tespiti (object detection) formatına uygun olarak sınıflandırılmış (Sarı Pas, Kara Pas, Kahverengi Pas, Külleme, Septoria, Kök Pası ve Sağlıklı) ve koordinat etiketleri (bounding boxes) `.txt` formatında düzenlenmiştir. Ayrıca farklı ışık ve açı koşullarına karşı veri setinin direnci, çeşitli veri artırma (data augmentation) teknikleriyle güçlendirilmiştir.

### 3. Modelleme
Görüntü işleme ve nesne tespiti alanında son teknoloji olan **YOLOv8** (You Only Look Once) mimarisi tercih edilmiştir. Model, bulut tabanlı bir API kullanılmadan, tamamen lokal ortamda (VS Code) eğitilmiştir. 
Saha verileri ile yapılan eğitimler sonucunda projenin kendi ağırlık dosyası (`best.pt`) elde edilmiştir. Bu sayede sistem dışa bağımlı olmadan otonom bir şekilde çalışabilmektedir.

### 4. Tavsiye Motoru (Birliktelik Kuralları)
Bu proje sadece bir sınıflandırma aracı değil, aynı zamanda proaktif bir karar destek motorudur. Modelin çıktı ürettiği her bir hastalığa karşılık gelen deterministik kural dizileri (If-Then-Else mantığı) tasarlanmıştır. 
* **Girdi:** Modelin tespit ettiği hastalık sınıfı (Örn: "Yellow Rust").
* **Karar Motoru Çıktısı:** Tespit edilen anormalliğin klinik adı, korunma stratejisi (kültürel önlemler) ve kimyasal/biyolojik müdahale (uygun ilaç, vitamin tavsiyesi) yöntemleri eşzamanlı olarak kullanıcıya iletilir.

---

## Sistem Mimarisi ve Akış

Kullanıcı ile sistem arasındaki etkileşim süreci şu adımlardan oluşmaktadır:
1. **Veri Girişi:** Kullanıcı, mobil veya masaüstü cihazından Streamlit tabanlı web arayüzüne giriş yaparak tarlada çektiği yaprak fotoğrafını (`.jpg`, `.png`) sisteme yükler.
2. **Ön İşleme ve Çıkarım (Inference):** Yüklenen görüntü lokal YOLOv8 modeli (`best.pt`) tarafından işlenerek hastalıklı bölgeler resim üzerinde koordinat bazlı (bounding box) işaretlenir.
3. **Karar Destek Devreye Girme:** Modelin tespit ettiği sınıf ismi, arka planda çalışan Kural Tabanlı Tavsiye Motoruna iletilir.
4. **Sonuç Gösterimi:** Kullanıcıya yüklediği resmin işaretlenmiş hali ile birlikte, teşhis edilen hastalık, görsel güven skoru (confidence) ve atılması gereken kritik adımlar (korunma, müdahale) anında sunulur.

---

## Kullanılan Teknolojiler

* **Programlama Dili:** Python 3.x
* **Nesne Tespiti (Derin Öğrenme):** Ultralytics YOLOv8
* **Web Arayüzü:** Streamlit
* **Görüntü İşleme:** OpenCV, Pillow (PIL), NumPy
* **Ortam Yönetimi:** `venv` (Sanal Ortam)

---

## Kurulum ve Çalıştırma (How to Run)

Projeyi kendi bilgisayarınızda veya yerel bir sunucuda çalıştırmak için aşağıdaki adımları sırasıyla terminalinizde uygulayın:

**1. Proje dizinine gidin ve sanal ortam (venv) oluşturun:**
```bash
python -m venv venv
```

**2. Sanal ortamı aktif hale getirin:**
*Linux / macOS için:*
```bash
source venv/bin/activate
```
*Windows için:*
```bash
venv\Scripts\activate
```

**3. Gerekli kütüphaneleri (bağımlılıkları) yükleyin:**
```bash
pip install ultralytics streamlit Pillow numpy
```

**4. Karar Destek Sistemini başlatın:**
```bash
streamlit run app.py
```
Sistem başlatıldıktan sonra tarayıcınızda `http://localhost:8501` adresi üzerinden arayüze erişebilirsiniz.

---

## Gelecek Hedefleri (Geliştirme Vizyonu)

* **Mobil Uygulama Entegrasyonu:** Tarlada internet bağlantısının olmadığı durumlar için Edge AI yöntemleriyle çalışan çevrimdışı (offline) mobil uygulamanın geliştirilmesi.
* **Drone (İHA) Destekli Otonom İlaçlama:** Sistemin tarımsal dronelara entegre edilerek, hastalıklı bölgelerin haritalandırılması ve sadece noktasal (spot) ilaçlama yaparak kimyasal atık miktarının en aza indirilmesi.
* **Veritabanı Entegrasyonu:** Kullanıcıların tarattığı yaprak verilerinin ve teşhislerin bir veritabanında toplanarak, bölgesel salgın haritalarının (Epidemioloji haritası) çıkarılması ve devlet/kooperatif kurumlarına öngörü raporları sunulması.
