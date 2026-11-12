# Swin Transformer ile Görüntü Sınıflandırma

Bu projede **pretrained Swin-Tiny** modeli kullanılarak Oxford-IIIT Pet veri setindeki kedi ve köpek ırkları sınıflandırılacaktır.

Model ImageNet üzerinde önceden eğitilmiş ağırlıklarla başlatılacak, Oxford-IIIT Pet veri setine göre fine-tune edilecek ve test veri seti üzerinde performansı ölçülecektir.

Projenin sonunda modelin farklı stage’lerde ürettiği feature map’ler de görselleştirilecektir.

---

# Proje Akışı

| Aşama | Ne Yapacağız? |
|---|---|
| **Proje Altyapısı ve Veri Setinin Hazırlanması** | Sanal ortam kurulacak, gerekli kütüphaneler yüklenecek ve Oxford-IIIT Pet veri seti Train / Validation / Test olarak hazırlanacaktır. |
| **Pretrained Swin-Tiny Modelinin Hazırlanması** | ImageNet üzerinde önceden eğitilmiş Swin-Tiny modeli yüklenecek ve 37 sınıflı probleme uyarlanacaktır. |
| **Swin-Tiny Modelinin Fine-Tune Edilmesi** | Pretrained model Oxford-IIIT Pet veri seti üzerinde eğitilecektir. |
| **Model Performansının Değerlendirilmesi** | Test Accuracy, Loss, Precision, Recall, F1-Score ve Confusion Matrix sonuçları incelenecektir. |
| **Hierarchical Feature Map’lerin Görselleştirilmesi** | Swin Transformer’ın Stage 1–4 çıktıları alınarak feature map’ler görselleştirilecektir. |

---

# Kullanılan Veri Seti

Projede **Oxford-IIIT Pet Dataset** kullanılmaktadır.

Veri seti farklı kedi ve köpek ırklarına ait gerçek görüntülerden oluşmaktadır.

```text
Sınıf Sayısı       : 37
Problem            : Multi-Class Image Classification
Görüntü Türü       : Kedi ve köpek fotoğrafları
Training Kaynağı   : trainval
Test Kaynağı       : test
Model Giriş Boyutu : 224x224
```

`trainval` veri seti proje içerisinde:

```text
%80 → Training
%20 → Validation
```

şeklinde ayrılmaktadır.

Veri setinin resmi `test` bölümü yalnızca final model performansını ölçmek için kullanılmaktadır.

---

# Kullanılan Model

Projede ImageNet-1k üzerinde pretrained:

```text
swin_tiny_patch4_window7_224.ms_in1k
```

modeli kullanılmaktadır.

Modelin orijinal classification katmanı Oxford-IIIT Pet veri setine göre:

```text
1000 sınıf
    ↓
37 sınıf
```

olarak değiştirilmektedir.

---

# Proje Yapısı

```text
swin-transformer/
│
├── data/
├── models/
├── outputs/
│
├── requirements.txt
├── config.py
└── swin_transformer.py
```

Dosyaların görevleri:

```text
requirements.txt
→ Projede kullanılacak Python kütüphanelerini içerir.

config.py
→ Model, veri seti ve eğitim parametrelerini içerir.

swin_transformer.py
→ Veri hazırlama, model oluşturma, fine-tuning,
  değerlendirme ve feature map görselleştirme
  işlemlerinin tamamını içerir.
```

---

# Virtual Environment Oluşturma

Proje klasörünü oluşturun:

```bash
mkdir swin-transformer
```

Proje klasörüne girin:

```bash
cd swin-transformer
```

Virtual environment oluşturun:

```bash
python -m venv venv
```

Windows üzerinde aktifleştirin:

```bash
venv\Scripts\activate
```

pip’i güncelleyin:

```bash
python -m pip install --upgrade pip
```

---

# Requirements

`requirements.txt`:

```text
torch
torchvision
timm
tqdm
matplotlib
scikit-learn
```

Kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

---

# Proje Parametreleri

Temel proje ayarları `config.py` içerisinde bulunmaktadır.

```text
MODEL_NAME      = swin_tiny_patch4_window7_224.ms_in1k
NUM_CLASSES     = 37
IMAGE_SIZE      = 224

BATCH_SIZE      = 16
NUM_WORKERS     = 0

EPOCHS          = 3
LEARNING_RATE   = 5e-5
WEIGHT_DECAY    = 1e-4

VAL_RATIO       = 0.20
SEED            = 42
```

Proje CPU üzerinde çalışacak şekilde hazırlanmıştır.

---

# Projenin Çalıştırılması

Ana proje dosyasını çalıştırın:

```bash
python swin_transformer.py
```

İlk çalıştırmada `config.py` içerisinde:

```python
TRAIN_MODEL = True
```

olmalıdır.

Bu durumda:

```text
Veri Setini Hazırla
        ↓
Pretrained Swin-Tiny Modelini Yükle
        ↓
Fine-Tuning
        ↓
Validation
        ↓
En İyi Modeli Kaydet
        ↓
Test
        ↓
Confusion Matrix
        ↓
Örnek Tahminler
        ↓
Stage Feature Map’leri
```

işlemleri gerçekleştirilecektir.

---

# Model Eğitimi

Model Oxford-IIIT Pet training veri setinde **3 epoch** boyunca fine-tune edilmektedir.

Her epoch sonunda terminalde:

```text
Train Loss
Train Accuracy

Validation Loss
Validation Accuracy

Learning Rate
```

değerleri gösterilmektedir.

Örnek:

```text
Epoch 1/3 |
Train Loss: 2.1534 |
Train Accuracy: 42.10% |
Val Loss: 1.2145 |
Val Accuracy: 68.75%
```

Validation Accuracy değeri yükseldiğinde model otomatik olarak kaydedilir.

Kaydedilen model:

```text
models/swin_tiny_best.pt
```

---

# Model Performansının Değerlendirilmesi

Training tamamlandıktan sonra en iyi model otomatik olarak yüklenir.

Model test veri seti üzerinde değerlendirilir.

Terminalde:

```text
Test Loss
Test Accuracy
```

değerleri gösterilir.

Ayrıca her sınıf için:

```text
Precision
Recall
F1-Score
```

değerleri hesaplanır.

---

# Confusion Matrix

Test sonuçlarından Confusion Matrix oluşturulur.

Çıktı:

```text
outputs/confusion_matrix.png
```

dosyasına kaydedilir.

---

# Örnek Tahminler

Test veri setinden örnek görüntüler seçilerek model tahminleri oluşturulur.

Her görüntü için:

```text
Gerçek Sınıf
Tahmin Edilen Sınıf
```

gösterilir.

Çıktı:

```text
outputs/sample_predictions.png
```

dosyasına kaydedilir.

---

# Stage Feature Map’leri

Fine-tune edilmiş Swin-Tiny modelinin dört farklı stage çıktısı alınır.

```text
Stage 1 → 56x56
Stage 2 → 28x28
Stage 3 → 14x14
Stage 4 → 7x7
```

Stage çıktıları heatmap formatında görselleştirilir.

Çıktı:

```text
outputs/swin_hierarchical_features.png
```

---

# Feature Map Overlay

Stage feature map’leri ayrıca orijinal görüntünün üzerine bindirilir.

Çıktı:

```text
outputs/swin_feature_overlays.png
```

---

# Modeli Tekrar Eğitmeden Çalıştırma

Model bir kez eğitildikten sonra `config.py` içerisindeki:

```python
TRAIN_MODEL = False
```

olarak değiştirilebilir.

Bu durumda:

```text
models/swin_tiny_best.pt
```

dosyasındaki model doğrudan yüklenir.

Training tekrar yapılmadan:

```text
Test Accuracy
Classification Report
Confusion Matrix
Örnek Tahminler
Stage Feature Map’leri
```

oluşturulur.

---

# Proje Çıktıları

```text
models/
└── swin_tiny_best.pt
```

```text
outputs/
├── confusion_matrix.png
├── sample_predictions.png
├── swin_hierarchical_features.png
└── swin_feature_overlays.png
```

---

# Proje Akışının Özeti

```text
Oxford-IIIT Pet
       ↓
Train / Validation / Test
       ↓
Pretrained Swin-Tiny
       ↓
3 Epoch Fine-Tuning
       ↓
En İyi Modeli Kaydet
       ↓
Test Accuracy
       ↓
Classification Report
       ↓
Confusion Matrix
       ↓
Örnek Tahminler
       ↓
Stage 1–4 Feature Map’leri
       ↓
Feature Map Görselleştirme
```