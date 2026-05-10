# 🌾 Hububat Hastalık Tespit ve Karar Destek Sistemi

## 1. Proje Özeti
Bu sistem, hububat yaprak görüntülerini analiz ederek hastalık belirtilerini tespit eden ve sonucu doğrudan tarımsal eyleme dönüştüren bir yapay zeka platformudur. Sistem sadece teşhis koymakla kalmaz; görsel kanıt, güven düzeyi ve uygulamaya dönük öneri üretir.

---

## 2. Saha Problemi
### 🚩 Temel Zorluklar
- Benzer görünümlü hastalıkların karıştırılması
- Geç teşhis nedeniyle verim kaybı
- Yanlış veya gecikmiş müdahale nedeniyle maliyet artışı
- Sadece sınıf adı veren araçların karar sürecinde yetersiz kalması

### ✅ Sistem Yaklaşımı
- Hastalık sınıfını otomatik belirleme
- Belirtiyi yaprak üzerinde konumsal işaretleme
- Güven skoru ile sonuç şeffaflığı sağlama
- Her sonuca özel korunma ve müdahale önerisi sunma

---

## 3. Sistem Mimarisi
### A. Nesne Algılama Modu (YOLOv8 Detection)
- 10 sınıfta tespit yapar
- Çoklu lezyonu tek görselde ayrı kutularla gösterir
- Arazi gözlemi ve lokal yayılım analizi için uygundur

### B. Sınıflandırma Modu (YOLOv8 Classification)
- 5 sınıfta hızlı genel değerlendirme sağlar
- Görüntünün tamamından baskın sınıfı çıkarır
- Ön tarama ve hızlı kontrol senaryolarında etkilidir

---

## 4. Uçtan Uca Çalışma Akışı
1. Kullanıcı analiz modunu seçer.
2. Yaprak görseli sisteme yüklenir.
3. Seçilen model çıkarım (inference) yapar.
4. Sonuçlar sınıf, güven skoru ve görsel işaretleme ile raporlanır.
5. Karar destek katmanı korunma ve müdahale planı üretir.

---

## 5. Veri Altyapısı
- Algılama veri seti: 10 sınıf, etiketli görüntüler
- Sınıflandırma veri seti: 5 sınıf, dengeli klasör yapısı
- Eğitim/validasyon/test ayrımı ile performans izleme
- Eğitim çıktıları `runs` klasöründe metrik ve ağırlık olarak saklanır

---

## 6. Karar Destek Çıktıları
Her analiz sonunda sistem şu bilgileri üretir:
- **Teşhis:** Hastalığın adı
- **Güven Düzeyi:** Modelin karar güveni
- **Görsel Kanıt:** İşaretlenmiş hastalık bölgeleri
- **Korunma Önerisi:** Önleyici saha uygulamaları
- **Müdahale Önerisi:** Uygun tedavi/ilaçlama yönlendirmesi

Bu nedenle platform, yalnızca "hastalık adı" değil, uygulanabilir aksiyon çerçevesi verir.

---

## 7. Kullanıcı Arayüzü
Streamlit tabanlı arayüz;
- hızlı görsel yükleme,
- tek tık analiz,
- sonuç kartları,
- öneri panelleri
ile teknik olmayan kullanıcılar için anlaşılır bir deneyim sağlar.

---

## 8. Kullanım Alanları
- Tarla ön kontrol süreçleri
- Zirai danışmanlıkta hızlı ilk değerlendirme
- Eğitim ve demonstrasyon amaçlı dijital tarım uygulamaları
- Hastalık takibi ve müdahale planlama süreçleri

---

## 9. Teknoloji Yığını
- Python
- PyTorch
- Ultralytics YOLOv8
- Streamlit
- CUDA

---
Hazırlayan: Bedirhan
Tarih: 10 Mayıs 2026
