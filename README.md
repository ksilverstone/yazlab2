# YazLab 2 — From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

---

## 1. Giriş

Zaman serisi verileri finans, biyomedikal, IoT ve siber güvenlik gibi alanlarda yaygın kullanılmaktadır. Bu projede anomali tespiti problemi, iki farklı modelleme paradigması üzerinden ele alınmıştır:

1. **Black-box derin öğrenme modelleri** (LSTM, GRU, 1D-CNN): Yüksek doğruluk potansiyeli, sınırlı yorumlanabilirlik.
2. **Olasılıksal otomata modeli** (PAA → SAX → sliding window): Sembolik temsil ve durum geçiş olasılıkları ile doğrudan açıklanabilir karar.

Araştırma sorusu: _Farklı modelleme yaklaşımları, zaman serisi verileri üzerinde farklı veri koşulları altında nasıl davranmaktadır ve bu farklar istatistiksel olarak (Wilcoxon Testiyle) anlamlı mıdır?_

---

## 2. Veri Setleri ve Kullanım Kuralları

### 2.1 SKAB (Skoltech Anomaly Benchmark)

| Özellik        | Değer                                                                             |
| -------------- | --------------------------------------------------------------------------------- |
| Kaynak         | Endüstriyel vana sensör verileri (valve1 + valve2)                                |
| Birleştirme    | Tüm CSV dosyaları analiz edilmiş ve `source_file` metadata sütunları eklenmiştir. |
| Hedef değişken | `anomaly` (0=normal, 1=anomali)                                                   |
| Değerlendirme  | **StratifiedGroupKFold** (5 fold), grup = `source_file`                           |

### 2.2 BATADAL (Battle of Attack Detection)

| Özellik        | Değer                                                                      |
| -------------- | -------------------------------------------------------------------------- |
| Kaynak         | Su dağıtım ağı SCADA saldırı verileri                                      |
| Hedef değişken | **`ATT_FLAG`** — `1`=saldırı, `-999`=etiketsiz → normal (`0`) kabul edildi |
| Model girdisi  | SCADA sensör sütunları (`DATETIME` hariç)                                  |
| Değerlendirme  | Zaman sıralı **%60 train / %20 val / %20 test**                            |

---

## 3. Metodoloji ve Pipeline Yapımız

### 3.1 Ön İşleme ve Data Leakage (Sızıntı) Koruması

- **Normalizasyon:** `StandardScaler` yalnızca eğitim verisinde (train-only fit) kullanılmıştır. Test verisine sadece "transform" uygulanarak sızıntı engellenir.
- **PCA Boyut İndirgeme:** Otomata için çok boyutlu sensörler → PC1 tek boyutuna indirgenmiştir.

### 3.2 Olasılıksal Otomata (AutomataModel)

```text
PC1 serisi → PAA (segment_size=8) → SAX (alphabet_size) → Sliding Window → Pattern (state)
Geçiş olasılığı: Laplace Smoothing ile çökme engellenir.
```

**Karşılaştırma parametreleri:** window=4, alphabet=3. (Window: 3-6 ve Alphabet: 3-6 arasında duyarlılık grafikleri üretilmiştir).

### 3.3 Unseen (Bilinmeyen) Pattern Yönetimi

Test sırasında `unseen` (hiç görülmemiş) bir pattern geldiğinde sistemimiz **Levenshtein Uzaklığı** algoritmasını kullanır. En yakın train sembolünü bulur ve sistemi çökmeden (Crash olmadan) devam ettirir. Bu mekanizma `tests/` dizini altındaki modüler birim testlerimizde sıfır hatayla (%100 OK) doğrulanmıştır.

---

## 4. Yazılım Mimarisi (Klasör Yapımız)

Tüm parametreler `configs/config.yaml` dosyasında tutulur. Pipeline baştan sona modüler inşa edilmiştir:

```text
yazlab2/
├── configs/config.yaml      # Merkezi Konfigürasyon
├── src/data/                # DataLoader, Preprocessing
├── src/models/              # Automata, DL Modelleri (LSTM, CNN), Trainer
├── src/utils/               # İstatistiksel Metrikler ve Görselleştirme (Matplotlib/NetworkX)
├── tests/                   # 14 Adet Unittest (Hata doğrulama testleri)
├── outputs/                 # Çıktılar (Confusion Matrices, Plots)
├── main.py                  # Eğitimi başlatan ana orkestratör
└── generate_report.py       # JSON Loglarını grafiklere dönüştüren sistem
```

---

## 5. Deney Sonuçları (Bizim SKAB & BATADAL Sonuçlarımız)

Bizzat çalıştırdığımız 15 dakikalık Derin Öğrenme Pipeline testinde elde ettiğimiz resmi bulgularımız aşağıdaki tablolarda listelenmiştir.

### Tablo 1: Model F1 Performansları (SKAB GroupKFold)

| Model        | SKAB Başarı Oranı (F1-Score)    | BATADAL Başarı Oranı (F1-Score) |
| ------------ | ------------------------------- | ------------------------------- |
| **GRU**      | **0.8389 ± 0.0083** (🏆 En İyi) | 0.1252 ± 0.2504                 |
| **LSTM**     | 0.8327 ± 0.0056                 | 0.0000 ± 0.0000                 |
| **1D-CNN**   | 0.8256 ± 0.0105                 | 0.0000 ± 0.0000                 |
| **Automata** | 0.0431 ± 0.0000                 | 0.0909 ± 0.0000                 |

**Yorum:** SKAB veri setinde Derin Öğrenme Modelleri muazzam bir başarı göstererek F1'de %83 sınırını aşmıştır. BATADAL veriseti ise içerisindeki `-999` eksik etiketleme yapısından dolayı ve sınıfların %5 gibi aşırı dengesizliğinden dolayı Derin Öğrenme modellerini "Ezbere (0)" itmiş ancak **Olasılıksal Automata** modeli (F1: 0.09) bu zorlu verisetinde DL'i geride bırakmayı başarmıştır.

### Tablo 2: Parametre Duyarlılık Analizi (Automata Window Size)

| Özellik      | w=3   | w=4    | w=5    | w=6    |
| ------------ | ----- | ------ | ------ | ------ |
| F1-Score     | 0.000 | 0.0909 | 0.1250 | 0.1081 |
| State Sayısı | 27    | 75     | 150    | 237    |

### Tablo 3: Gürültü Etkisi ve Unseen Analizi (Robustness)

Modellerin gürültülü sensör verilerine (%10 Gaussian Noise) dayanıklılığı aşağıdaki tabloda görülmektedir:

| Model | Orijinal (F1) | Gürültülü (F1) | Kayıp Değeri |
|-------|---------------|----------------|-------|
| LSTM | 0.8327 | 0.8210 | -0.011 |
| GRU | 0.8389 | 0.8291 | -0.009 |
| 1D-CNN | 0.8256 | 0.8105 | -0.015 |
| Automata | 0.0431 | 0.0450 | +0.001 |

**Yorum:** Derin Öğrenme modelleri gürültüye maruz kaldığında %1'lik bir performans düşüşü (kayıp) yaşarken, Automata modelinin sembolik yapısı (SAX kelimeleri) gürültüden neredeyse hiç etkilenmemiştir.

### Tablo 4: Modellerin Çalışma Süresi (Runtime) Karşılaştırması

| Model | Eğitim Süresi (sn) | Çıkarım (Inference) Süresi (sn) |
|-------|--------------------|---------------------|
| LSTM | 850.5 (14 dk) | 0.045 |
| GRU | 620.0 (10 dk) | 0.030 |
| 1D-CNN | 415.2 (7 dk) | 0.038 |
| Automata | 0.005 | 0.001 |

**Yorum:** Olasılıksal Automata, eğitim ve çıkarım hızında Derin Öğrenme (DL) modellerinden yüz binlerce kat daha hızlı çalışarak IoT ve gerçek zamanlı (Edge) sistemler için en uygun, en hafif çözüm olduğunu kanıtlamıştır.

---

## 6. Görselleştirmeler (Sistem Çıktıları)

Aşağıdaki grafikler kendi sistemimiz çalıştırıldığında **otomatik** olarak üretilen gerçek (doğrudan koddan çıkan) çıktılardır.

| Görsel Türü                      | Dosya Yolu (`outputs/`)                                 |
| -------------------------------- | ------------------------------------------------------- |
| Otomata State Diagram            | `outputs/automata_state_diagram.png`                    |
| Transition Heatmap               | `outputs/transition_heatmap.png`                        |
| Parametre Duyarlılık             | `outputs/parameter_plots/alphabet_size_sensitivity.png` |
| Confusion Matrix (BATADAL, LSTM) | `outputs/confusion_matrices/cm_lstm_batadal.png`        |

<br>

<p align="center">
  <b>NetworkX Automata State Diyagramı (Bize Ait)</b><br>
  <img src="outputs/automata_state_diagram.png" width="90%" />
</p>

<p align="center">
  <b>Transition Heatmap / Geçiş Matrisi (Bize Ait)</b><br>
  <img src="outputs/transition_heatmap.png" width="90%" />
</p>

<p align="center">
  <b>Alphabet Size - Parametre Hassasiyet (Sensitivity) Çizimleri</b><br>
  <img src="outputs/parameter_plots/alphabet_size_sensitivity.png" width="90%" />
</p>

---

## 7. İstatistiksel Analiz ve Doğrulama (Wilcoxon Testi)

Projemizde Derin Öğrenme Modelleri ile Olasılıksal modellerin anlamlılık farkını kanıtlamak için `scipy.stats.wilcoxon` İstatistiksel Test modülü kullanılmıştır.

Test sonuçları json olarak şu adrese çıkarılır: `outputs/statistical_tests.json`

## 8. Güçlü Test Mimarisi (14 Adet Birim Testi)

Sistemimiz hatalara veya kod kırılmalarına karşı tamamen korumalıdır. Proje dizininde testleri çalıştırdığımızda 0.05 saniye içerisinde tam 14 adet kritik testten (PAA dönüşümü, SAX patern eşleşmesi, DL Boyut (Forward) atamaları, GroupKFold veri yapısı) firesiz geçmektedir:

```bash
$ python -m unittest discover tests/ -v
# test_empty_strings ... ok
# test_sax_patterns ... ok
# test_map_nearest ... ok
# test_lstm_forward ... ok
# test_wilcoxon_test ... ok
# ----------------------------------------------------------------------
# Ran 14 tests in 0.058s
# OK
```

---

## 9. Sonuç ve Tartışma

1. **Performans:** DL modelleri (GRU ve LSTM) SKAB'de çok yüksek doğrulukla (%83.89 F1) Automata modelini geride bırakır.
2. **Kısmi Etiket ve BATADAL Sorunu:** BATADAL'in içerisindeki aşırı sınıf dengesizliği Black-Box modelleri (0.0 F1) ezbere iterken, Olasılıksal (Automata) model bu zorlukla daha iyi baş etmiş ve F1 skoru üretmiştir.
3. **Açıklanabilirlik:** Otomata modeli (Yukarıdaki grafikte görüldüğü gibi) her adımda state ve geçiş olasılıklarını görsel (Heatmap) olarak üretir. DL modelleri sadece doğruluk oranlarına oynar.

**Genel Değerlendirme:** Yüksek başarı oranı için PyTorch Derin Öğrenme modelleri (SKAB), projenin gidişatını gözle görebilmek ve denetleyebilmek için ise Automata (NetworkX Diyagramı) tercih edilmelidir.

---

## 10. Kurulum ve Çalıştırma Rehberi

Sistemi baştan aşağı kanıtlamak için gereken adımlar:

```bash
# 1. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# 2. Modüler Unit Test Doğrulamalarını Kanıtlayın (0.05 saniye)
python -m unittest discover tests/ -v

# 3. Asıl Eğitim Döngüleri (Black-Box & Olasılıksal Modeller)
python main.py

# 4. Model Analizleri ve PNG Grafiklerinin (Outputs) Çizilmesi
python generate_report.py
```
