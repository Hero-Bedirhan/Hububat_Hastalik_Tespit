# 🌾 Hububat Hastalık Tespit ve Karar Destek Sistemi Sunumu

## 1. Proje Özeti
Bu proje, küresel gıda güvenliği ve tarımsal verimlilik için kritik öneme sahip olan buğday ekinlerindeki hastalıkları yapay zeka yardımıyla tespit etmek amacıyla geliştirilmiştir. Geliştirilen sistem, sadece teşhis koymakla kalmayıp, çiftçiye bilimsel verilere dayalı bir yol haritası sunan bir **Karar Destek Sistemi (KDS)** niteliğindedir.

---

## 2. Problem ve Çözüm
### 🚩 Problem
*   Hastalıkların çıplak gözle yanlış veya geç teşhis edilmesi.
*   Yanlış ilaç kullanımı nedeniyle artan maliyetler ve çevre kirliliği.
*   Zaman kaybı ve ekin verimliliğinde ciddi düşüşler.

### ✅ Çözüm
*   Derin öğrenme tabanlı hızlı ve kesin teşhis.
*   Mobil uyumlu web arayüzü ile tarlada anlık analiz.
*   Hastalığa özel kültürel ve kimyasal müdahale önerileri.

---

## 3. Veri Seti: Roboflow Entegrasyonu
Projenin temelini, [Roboflow Wheat Disease Detection](https://universe.roboflow.com/agliculture/wheat-disease-detection-zsn0p) veri seti oluşturmaktadır.

*   **Toplam Sınıf Sayısı:** 5 (4 Hastalık, 1 Sağlıklı)
*   **İçerik:** Binlerce yüksek çözünürlüklü buğday yaprağı ve tanesi görüntüsü.
*   **Veri Dağılımı:** Eğitim, Doğrulama (Validation) ve Test aşamaları için dengeli bir şekilde bölünmüştür.

---

## 4. Teknik Metodoloji: Neden Sınıflandırma?
Proje geliştirme sürecinde iki farklı yaklaşım test edilmiştir:

1.  **Nesne Tespiti (Object Detection):** Görüntüde kutu çizerek leke arama yöntemi. Bu yöntemde karmaşık arka planlar nedeniyle başarı oranı **%44** seviyelerinde kalmıştır.
2.  **Görüntü Sınıflandırma (Image Classification):** Görüntünün tamamını analiz eden YOLOv8-cls mimarisi. Bu yönteme geçiş yapılarak modelin odağı netleştirilmiş ve başarı oranı **%99.76**'ya çıkarılmıştır.

---

## 5. Eğitim Süreci ve Performans
Model, **YOLOv8s-Classification** mimarisi üzerinde eğitilmiştir.

*   **Hedef Epoch:** 100
*   **Erken Durdurma (Patience):** 20
*   **Sonuç:** Model, 6. epoch itibariyle en yüksek verime ulaşarak eğitimi başarıyla tamamlamıştır.
*   **Doğruluk Oranı (Accuracy):** %99.76
*   **Kayıp (Loss) Değeri:** 0.0049 (Neredeyse kusursuz öğrenme)

---

## 6. Karar Destek Sistemi (KDS) Motoru
Sistem, teşhis edilen hastalığa göre şu bilgileri üretir:
*   **Teşhis:** Hastalığın bilimsel ve halk arasındaki adı.
*   **Güven Skoru:** Yapay zekanın teşhisten ne kadar emin olduğu.
*   **🛡️ Korunma:** Gelecek ekim dönemleri ve bulaşma riskine karşı kültürel önlemler.
*   **💊 Müdahale:** Acil durumlarda kullanılması gereken ilaç grupları ve uygulama yöntemleri.

---

## 7. Kullanıcı Arayüzü
**Streamlit** platformu üzerinde geliştirilen web arayüzü sayesinde:
*   Kullanıcı sürükle-bırak yöntemiyle fotoğraf yükler.
*   Model saniyeler içinde çıkarım (inference) yapar.
*   Sonuçlar anlaşılır grafikler ve uyarı kutuları ile sunulur.

---

## 8. Gelecek Vizyonu
Projenin ölçeklendirilebilir yapısı sayesinde kısa ve orta vadeli hedeflerimiz şunlardır:

*   **Mobil Uygulama Entegrasyonu:** Çiftçilerin tarlada internete ihtiyaç duymadan, telefon kamerasıyla canlı çekim yaparak anlık hastalık teşhisi alabileceği bir mobil uygulama geliştirilmesi.
*   **Tam Otomatik Tedavi Reçetesi:** Sadece hastalık ismi değil; hastalığın evresine göre hangi ilacın (fungisit), hangi dozda ve hangi gübreleme programıyla uygulanacağını söyleyen gelişmiş bir veritabanı entegrasyonu.
*   **İlaç ve Etken Madde Rehberi:** Teşhis sonrası, hastalığı tamamen ortadan kaldıracak en uygun zirai ilaçların ve bu ilaçların içindeki spesifik etken maddelerin (aktif içeriklerin) otomatik olarak listelenmesi.
*   **Zenginleştirilmiş Veri Setleri:** Farklı iklim ve toprak koşullarına ait tedavi sonuçlarını içeren yeni veri setleriyle modelin eğitilmesi, böylece çiftçiye araştırma yapma gereği bırakmadan "kişiselleştirilmiş eylem planı" sunulması.
*   **İHA (Drone) ve Uydu Entegrasyonu:** Geniş arazilerde hastalık yayılım hızını tahmin eden bölge bazlı risk haritalarının oluşturulması.

## 9. Kullanılan Teknolojiler ve Tercih Nedenleri

Projenin geliştirilmesinde seçilen her teknoloji, sistemin hızı, doğruluğu ve kullanıcı dostu olması hedeflenerek belirlenmiştir:

*   **Python:** Yapay zeka ve veri bilimi dünyasının standart dili olması, zengin kütüphane desteği ve hızlı geliştirme imkanı sunması nedeniyle tercih edilmiştir.
*   **YOLOv8 (Ultralytics):** Nesne tespiti ve sınıflandırma alanında dünyanın en hızlı ve en isabetli algoritmalarından biridir. Düşük donanımlı cihazlarda (telefonlar, hafif bilgisayarlar) bile yüksek performansla çalışabilmesi, tarlada kullanım vizyonumuz için kritiktir.
*   **Streamlit:** Karmaşık web geliştirme süreçlerine girmeden, sadece Python kullanarak modern, mobil uyumlu ve veri odaklı arayüzler oluşturmamıza olanak sağlar. Çiftçinin sistemi kolayca kullanabilmesi için en hızlı ve etkili yoldur.
*   **PyTorch:** YOLOv8'in temelinde yatan derin öğrenme kütüphanesidir. GPU (ekran kartı) gücünü en verimli şekilde kullanarak eğitim süresini kısaltır ve modelin karmaşık kalıpları daha iyi öğrenmesini sağlar.

---
**Hazırlayan:** Bedirhan
**Tarih:** 9 Mayıs 2026
**Teknoloji:** Python, YOLOv8, Streamlit, PyTorch
