# 🚀 Zaman Serisi Analizi: Black-Box (Derin Öğrenme) ve Açıklanabilir (Olasılıksal Otomat) Modellerin Karşılaştırılması

## Proje Hakkında (YazLab 2)
Bu proje, PDF isterlerine %100 uygun olarak zaman serisi anormallik tespiti ve analizi üzerinde iki farklı paradigmanın karşılaştırmasını gerçekleştirmektedir:

- **Derin Öğrenme (Black-Box) Modelleri**: LSTM, GRU, 1D-CNN
- **Olasılıksal Otomata (Interpretable) Modeli**: PAA → SAX → Sliding Window → Geçiş Olasılık Matrisi (TPM) ve State (NetworkX) Diyagramları.

Projeyle birlikte, makine öğrenmesi modellerinin gürültülü verilere (Gaussian Noise) dayanıklılığı, açıklanabilirlik seviyesi ve "Statistical Significance" (Wilcoxon Testi) gibi tüm metrikler detaylarıyla incelenmiştir.

---

## 📸 Grafiksel Sistem Analizleri (Outputs)
Otomat modelinin çalışma prensibini kanıtlayan, kod mimarisi tarafından otomatik üretilmiş olan **Durum (State) Diyagramları** ve **Isı Haritaları**:

<p align="center">
  <img src="outputs/automata_state_diagram.png" width="45%" title="NetworkX State Diyagramı" />
  <img src="outputs/transition_heatmap.png" width="45%" title="Olasılık Isı Haritası" /> 
</p>

*(Not: Test sonuçlarına ait tüm Karşılaştırma Grafikleri ve Confusion Matrisleri, projeyi çalıştırdığınız anda `outputs/` klasörüne otomatik olarak kaydedilmektedir.)*

---

## 📊 Başarı Oranları ve Skorlar (SKAB GroupKFold Sonuçları)
Sistemin 5 Parçalı Çapraz Doğrulama (GroupKFold) ve 3 farklı Rastgele Tohum (Seed) kullanılarak test edilen modellerin **F1-Score** performansları:

- **GRU Modeli:** `%83.89` *(En Yüksek Performans)*
- **LSTM Modeli:** `%83.27` 
- **1D-CNN Modeli:** `%82.56`
- **Olasılıksal Otomata:** `%4.31` *(Dengesiz veriden kaynaklı açıklanabilir ancak düşük başarı)*

---

## 🛡️ Güçlü Modüler Test Mimarisi (Doğrulama)
Projeye, sistemin bir çalışma zamanı hatası fırlatmasını veya matematiksel sızıntı yaşamasını engelleyen **14 adet** modüler Birim Test (Unit Test) yazılmıştır. PAA/SAX mantığından, Pytorch boyut testlerine kadar tüm yapı saniyeler içinde kanıtlanabilir.

```bash
$ python -m unittest discover tests/ -v

# test_sax_patterns ... ok
# test_lstm_forward ... ok
# test_wilcoxon_test ... ok
# ----------------------------------------------------------------------
# Ran 14 tests in 0.058s
# OK
```

---

## ⚙️ Kurulum ve Bağımlılıklar

Proje içerisindeki sanal ortamınıza (venv) aşağıdaki komutla tüm paketleri yükleyebilirsiniz:
```bash
pip install -r requirements.txt
```

### Proje Nasıl Çalıştırılır?

#### 1. Ana Eğitimi Başlatın (Tüm Modelleri Eğit)
```bash
python main.py
```
*(Tüm modeller GroupKFold çapraz doğrulama mekanizmasıyla çalışır. Bu işlem 15-20 dk sürebilir. Sonuçlar JSON olarak `outputs/` altına yazılır.)*

#### 2. İstatistikleri ve Görselleri Çıkarın (Sunum Çıktıları)
Sistem eğitimden ürettiği o JSON verilerini okuyup yukarıda gördüğünüz tüm grafik materyallerini çizer:
```bash
python generate_report.py
```

## 🏗️ Proje Mimarisi

```text
yazlab2/
├── configs/
│   └── config.yaml              # Merkezi Ayarlar
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset
│   │   └── preprocessor.py      # SKAB (GroupKFold) ve BATADAL işleyici
├── models/                      # DL ve Automata (Laplace Smoothing) Katmanları
├── tests/                       # 14 Adet Modüler Doğrulama Testi
├── outputs/                     # Tüm PNG Grafikleri ve JSON sonuçları
├── main.py                      # Ana Eğitim Modülü
├── generate_report.py           # Raporlama ve Çizim (NetworkX) Modülü
└── requirements.txt             
```

## İstatistiksel Karşılaştırma (Wilcoxon Testi)
Derin Öğrenme Modelleri ile Otomat Modeli arasındaki anlamlı farklılıkları ölçmek adına `scipy.stats.wilcoxon` modülünü kullanılmıştır.

## Unseen (Bilinmeyen) Pattern Yönetimi
Hiç görülmemiş bir pattern ile karşılaşıldığında **Levenshtein Uzaklığı** kullanılır. Hiç geçiş bulunamayan Olasılık Matrisi düğümlerinde ise **Laplace Smoothing** yöntemi aktif olarak uygulanarak sistemin çökmesi (Crash) tamamen engellenmiştir.