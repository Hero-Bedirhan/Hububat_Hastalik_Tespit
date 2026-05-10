# 🌾 HasatVizyon Pro

HasatVizyon Pro, hububat yaprak görüntülerinden hastalık analizi yapan ve sonucu tarımsal aksiyona dönüştüren bir Karar Destek Sistemi'dir. Sistem sadece teşhis koymakla kalmaz; hastalığın nerede görüldüğünü işaretler, güven skorunu sunar ve her sonuç için korunma ile müdahale önerisi üretir.

## Sistem Ne Yapar?
1. Kullanıcıdan yaprak görseli alır.
2. Seçilen analiz moduna göre modeli çalıştırır.
3. Hastalık veya sağlıklı durum bilgisini güven düzeyiyle raporlar.
4. Uygun durumlarda yaprak üzerindeki belirtileri kutu (bounding box) ile gösterir.
5. Sonucu agronomik öneri metnine dönüştürür.

## Analiz Modları
### 1. Nesne Algılama (YOLOv8 Detection)
- 10 sınıflı kapsamla belirtileri konumsal olarak tespit eder.
- Çoklu belirtiyi aynı görselde ayrı ayrı yakalayabilir.
- Sonucu hem sınıf hem güven skoru hem de görsel kanıtla verir.

### 2. Sınıflandırma (YOLOv8 Classification)
- 5 sınıflı hızlı genel değerlendirme sunar.
- Görüntünün tamamından baskın sınıfı çıkarır.
- Ön tarama ve hızlı saha kontrol senaryolarında etkilidir.

## Karar Destek Katmanı
Her sınıf için sistem aşağıdaki çıktıları üretir:
- Teşhis adı (yerel ifade + teknik karşılık)
- Korunma önerisi (kültürel önlemler, hijyen, ekim yaklaşımı)
- Müdahale önerisi (uygun ilaçlama veya izleme yönlendirmesi)

Bu yapı sayesinde sonuç ekranı yalnızca "hastalık adı" değil, uygulanabilir tarımsal eylem planı da içerir.

## Teknik Altyapı
- Python
- PyTorch
- Ultralytics YOLOv8
- Streamlit
- CUDA (uygun donanımda hızlandırma)

## Proje Yapısı
- app.py: Streamlit tabanlı kullanıcı arayüzü ve analiz akışı
- train_detection_model.py: Nesne algılama modeli eğitim süreci
- train_classification_model.py: Sınıflandırma modeli eğitim süreci
- models/detection_model.pt: Algılama modeli ağırlıkları
- models/classification_model.pt: Sınıflandırma modeli ağırlıkları
- dataset/detection_dataset: Algılama veri seti ve etiketleri
- dataset/classification_dataset: Sınıflandırma veri seti
- runs: Eğitim çıktıları, metrikler ve ağırlıklar

## Çalıştırma
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Kullanım Akışı
1. Uygulamayı açın.
2. Sol panelden analiz modunu seçin.
3. Görsel yükleyin.
4. Analizi başlatın.
5. Teşhis, güven skoru, işaretlenmiş görsel ve öneri kartlarını inceleyin.

---
Hazırlayan: Bedirhan
