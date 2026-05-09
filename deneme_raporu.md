# 🧪 Model Deneme ve Analiz Raporu

Bu rapor, projenin geliştirme sürecindeki model denemelerini, başarılarını ve başarısızlık nedenlerini belgelemek amacıyla oluşturulmuştur.

---

## ❌ Deneme #1: Yüksek Skorlu Overfitting (Aşırı Öğrenme) Vakası
**Tarih:** 9 Mayıs 2026  
**Model:** YOLOv8s-cls (10 Epoch)  
**Raporlanan Skor:** %99.93 Accuracy  
**Durum:** BAŞARISIZ (Canlı sistemde hatalı tahmin)

### 1. Bulgular ve Hata Analizi
Eğitim sırasında model kağıt üzerinde kusursuz (%99.93) bir skor vermiş olsa da, canlı testlerde sadece "Wheat Foliar Disease" (Buğday Yaprak Hastalığı) tahminini yaptığı görülmüştür. Bu durumun teknik nedenleri şunlardır:

*   **Aşırı Öğrenme (Overfitting):** Model, veri setindeki görüntüleri ayırt etmek yerine ezberlemiştir. Özellikle arka plan veya belirli veri seti karakteristikleri (Roboflow'dan gelen etiketleme hataları veya veri benzerliği) modelin yanılmasına neden olmuştur.
*   **Sınıf Yanlılığı (Bias):** "Wheat Foliar Disease" sınıfı, veri setinde çok baskın veya diğer sınıfların özelliklerini de kapsayan (şemsiye bir terim gibi) bir yapıda olabilir. Model, en düşük riskli gördüğü bu sınıfa yönelmiştir.
*   **Kayıp Fonksiyonu Aldatmacası:** 10 epok gibi kısa sürede ulaşılan 0.003 val_loss, gerçek dünyadaki çeşitliliği karşılayamamış, sadece eğitim ortamındaki dar veride geçerli kalmıştır.

### 2. Alınan Aksiyonlar
*   Hatalı model (`best.pt`) sistemden kaldırıldı.
*   Önceki kararlı ve güvenilir modele (%99.76) geri dönüldü (`model/bugday_model.pt` restore edildi).
*   Sistem tekrar kararlı hale getirildi.

---

## ✅ Güncel Durum: Kararlı Model (Production)
*   **Doğruluk:** %99.76
*   **Karakteristik:** Farklı hastalık sınıflarını (BlackPoint, Fusarium vb.) ayırt edebiliyor.
*   **Öneri:** Yeni bir eğitim yapılmadan önce veri setindeki "Wheat Foliar Disease" sınıfının diğer sınıflarla olan benzerliği incelenmeli ve veri temizliği (cleaning) yapılmalıdır.

---
*Hazırlayan: Gemini CLI - Analiz Raporu*
