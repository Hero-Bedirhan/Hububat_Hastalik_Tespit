# 🌾 HasatVizyon Pro: Akıllı Tarım Sunumu

## 1. Proje Vizyonu
Hububat üretiminde hastalıkların yanlış teşhis edilmesini önlemek ve çiftçiye **anlık, bilimsel ve uygulanabilir** bir yol haritası sunmak amacıyla geliştirilmiş hibrit bir KDS ekosistemidir.

---

## 2. Neden Hibrit Mimari?
Tek bir model yerine iki farklı uzmanlık alanını birleştirdik:

### 🎯 Odaklı Teşhis (Detection)
*   **Görev:** Hastalığı bul ve işaretle.
*   **Sınıf Sayısı:** 10
*   **Avantaj:** Yaprak üzerindeki lezyonları tek tek sayabilir, hastalığın şiddetini görselleştirir.

### ⚡ Hızlı Doğrulama (Classification)
*   **Görev:** Genel sağlığı raporla.
*   **Sınıf Sayısı:** 5
*   **Avantaj:** %99.7 başarı ile hata payını sıfıra indirir, mobil cihazlarda ışık hızında çalışır.

---

## 3. Veri Seti Derinliği
*   **Toplam Görüntü:** 10.000'e yakın profesyonel çekim.
*   **Teknik:** MD5 bazlı temizlik ve Tabakalı Bölme (Stratified Split).
*   **Çeşitlilik:** 8 farklı buğday hastalığı ve sağlıklı kontrol grubu.

---

## 4. Karar Destek Sistemi (KDS) Motoru
Sistem sadece "Hastalık var" demez:
1.  **Teşhis:** Hastalığın adını koyar.
2.  **Güven:** Tahmin gücünü (%) belirtir.
3.  **🛡️ Korunma:** Gelecek sezon için kültürel önlem listesi sunar.
4.  **💊 Müdahale:** Acil durumlar için spesifik fungisit önerisi yapar.

---

## 5. Performans Metrikleri
| Metrik | Detection Modu | Classification Modu |
| :--- | :--- | :--- |
| **Başarı Oranı** | %80.0 (mAP50) | %99.7 (Accuracy) |
| **Sınıf Sayısı** | 10 Sınıf | 5 Sınıf |
| **Hız (RTX 5050)** | 8.2 ms | 3.5 ms |
| **Kullanım Amacı** | Detaylı Analiz | Hızlı Ön Eleme |

---

## 6. Gelecek Planları
*   🚜 **Drone Entegrasyonu:** Otonom arazi tarama.
*   📱 **Mobil Uygulama:** Çevrimdışı (offline) teşhis desteği.
*   🌦️ **İklim Entegrasyonu:** Hava durumuna göre hastalık risk tahmini.

---
**Hazırlayan:** Bedirhan
**HasatVizyon AI v2.5**
