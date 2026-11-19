# DINOv3 ile Görüntü Feature Extraction ve Linear Probing

Bu projede **pretrained DINOv3 ViT-S/16** modeli kullanılarak STL-10 görüntülerinden feature vektörleri çıkarılacaktır.

DINOv3 backbone tamamen **frozen** tutulacak ve model ağırlıkları değiştirilmeden kullanılacaktır.

Çıkarılan feature'lar kullanılarak:

- Linear Probe ile görüntü sınıflandırma
- Model performansının değerlendirilmesi
- PCA ile feature space görselleştirme
- Cosine Similarity ile benzer görüntü bulma

işlemleri gerçekleştirilecektir.

---

# Proje Akışı

| Aşama | Ne Yapacağız? | Dosya |
|---|---|---|
| **Proje Altyapısı ve STL-10 Veri Setinin Hazırlanması** | Gerekli kütüphaneler kurulacak ve train/test görüntüleri hazırlanacaktır. | `01_extract_features.py` |
| **Pretrained DINOv3 Modelinin Hazırlanması** | DINOv3 ViT-S/16 modeli yüklenecek ve backbone tamamen freeze edilecektir. | `01_extract_features.py` |
| **DINOv3 ile Feature Extraction** | Train ve test görüntüleri embedding vektörlerine dönüştürülecektir. | `01_extract_features.py` |
| **Linear Probe Eğitimi** | DINOv3 feature'ları üzerinde yalnızca bir Linear Layer eğitilecektir. | `02_train_linear_probe.py` |
| **Model Performansının Değerlendirilmesi** | Test Accuracy, Precision, Recall, F1-Score ve Confusion Matrix hesaplanacaktır. | `03_analyze_features.py` |
| **Feature Space Görselleştirme** | DINOv3 embedding'leri PCA ile iki boyuta indirilecektir. | `03_analyze_features.py` |
| **Benzer Görüntülerin Bulunması** | Cosine Similarity kullanılarak query görüntüsüne en yakın görüntüler bulunacaktır. | `03_analyze_features.py` |

---

# Kullanılan Veri Seti

Projede **STL-10** veri seti kullanılmaktadır.

```text
Sınıf Sayısı       : 10
Train Görüntüsü    : 5.000
Görüntü Boyutu     : 96x96
Model Giriş Boyutu : DINOv3 processor tarafından hazırlanır
```

STL-10 içerisindeki sınıflar:

```text
airplane
bird
car
cat
deer
dog
horse
monkey
ship
truck
```

CPU üzerinde daha hızlı çalışmak için başlangıçta:

```python
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 500
```

kullanılmaktadır.

İstenirse bu değerler artırılabilir.

---

# Kullanılan Model

Projede:

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

modeli kullanılmaktadır.

Model DINOv3 ViT-S/16 mimarisidir.

Projede DINOv3 backbone eğitilmeyecektir.

```text
DINOv3
↓
FROZEN
↓
Image Embedding
```

Modelin tüm parametrelerinde:

```python
parameter.requires_grad = False
```

kullanılmaktadır.

Bu nedenle:

```text
DINOv3 Eğitilebilir Parametre : 0
```

olacaktır.

---

# Hugging Face Model Erişimi

DINOv3 modeli gated model olduğu için ilk kullanım öncesinde model erişim koşullarının Hugging Face hesabı üzerinden kabul edilmesi gerekir.

Daha sonra terminalden:

```bash
hf auth login
```

komutu ile Hugging Face hesabına giriş yapılabilir.

---

# Proje Yapısı

```text
dinov3-project/
│
├── features/
│
├── models/
│
├── outputs/
│
├── requirements.txt
├── config.py
├── 01_extract_features.py
├── 02_train_linear_probe.py
└── 03_analyze_features.py
```

Dosyaların görevleri:

```text
requirements.txt
→ Projede kullanılan Python kütüphanelerini içerir.

config.py
→ Model, veri seti, Linear Probe ve analiz parametrelerini içerir.

01_extract_features.py
→ STL-10 görüntülerini pretrained DINOv3 ile embedding'lere dönüştürür.

02_train_linear_probe.py
→ DINOv3 embedding'leri üzerinde Linear Probe eğitir.

03_analyze_features.py
→ Test performansını hesaplar, Confusion Matrix ve PCA oluşturur,
  Cosine Similarity ile benzer görüntüleri bulur.
```

---

# Virtual Environment Oluşturma

Proje klasörüne girin:

```bash
cd dinov3-project
```

Virtual environment oluşturun:

```bash
python -m venv venv
```

Windows üzerinde aktifleştirin:

```bash
venv\Scripts\activate
```

pip'i güncelleyin:

```bash
python -m pip install --upgrade pip
```

---

# Requirements

`requirements.txt`:

```text
torch
transformers>=5.0.0
datasets
huggingface_hub
Pillow
tqdm
matplotlib
scikit-learn
numpy
```

Kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

---

# Proje Parametreleri

Temel ayarlar `config.py` içerisinde bulunmaktadır.

```text
MODEL_NAME        = facebook/dinov3-vits16-pretrain-lvd1689m

TRAIN_SAMPLES     = 1000
TEST_SAMPLES      = 500

BATCH_SIZE        = 8
NUM_WORKERS       = 0

LINEAR_EPOCHS     = 20
LINEAR_LR         = 1e-3
WEIGHT_DECAY      = 1e-4

VAL_RATIO         = 0.20

QUERY_INDEX       = 0
TOP_K             = 5

SEED              = 42
```

Proje CPU üzerinde çalışacak şekilde hazırlanmıştır.

---

# 1. DINOv3 ile Feature Extraction

İlk olarak:

```bash
python 01_extract_features.py
```

çalıştırılır.

Akış:

```text
STL-10
   ↓
Pretrained DINOv3
   ↓
Backbone Freeze
   ↓
Train Images
   ↓
Feature Extraction
   ↓
Train Embeddings
```

Aynı işlem test görüntüleri için de gerçekleştirilir.

```text
Test Images
   ↓
DINOv3
   ↓
Test Embeddings
```

Her görüntü bir feature vektörüne dönüştürülür.

Örnek:

```text
Image
  ↓
DINOv3
  ↓
[0.21, -0.54, 0.73, ..., 0.18]
```

---

# Feature Dosyalarının Kaydedilmesi

Feature extraction tamamlandığında:

```text
features/
├── train_features.pt
└── test_features.pt
```

dosyaları oluşturulur.

Bu dosyalarda:

```text
features
labels
class_names
```

bilgileri saklanmaktadır.

Bu aşamadan sonra DINOv3 modelini tekrar çalıştırmaya gerek yoktur.

---

# 2. Linear Probe Eğitimi

Feature extraction tamamlandıktan sonra:

```bash
python 02_train_linear_probe.py
```

çalıştırılır.

Linear Probe yapısı:

```text
DINOv3 Feature
      ↓
Embedding
      ↓
Linear Layer
      ↓
10 STL-10 Sınıfı
```

Bu aşamada DINOv3 eğitilmez.

Sadece:

```text
Linear Layer
```

ağırlıkları güncellenir.

Training verisinin:

```text
%80 → Training
%20 → Validation
```

olarak ayrılmasıyla Linear Probe eğitilir.

---

# Linear Probe Training

Her epoch sonunda:

```text
Train Loss
Train Accuracy
Validation Loss
Validation Accuracy
```

değerleri gösterilir.

Örnek:

```text
Epoch 1/20 |
Train Loss: 2.0134 |
Train Accuracy: 38.75% |
Val Loss: 1.7142 |
Val Accuracy: 52.50%
```

Validation Accuracy yükseldiğinde classifier kaydedilir.

Kaydedilen model:

```text
models/
└── dinov3_linear_probe.pt
```

---

# 3. Feature Analizi

Feature extraction ve Linear Probe training tamamlandıktan sonra:

```bash
python 03_analyze_features.py
```

çalıştırılır.

Bu dosya üç temel analiz gerçekleştirir.

```text
Classification Performance
PCA Feature Space
Similar Image Search
```

---

# Model Performansının Değerlendirilmesi

Linear Probe test embedding'leri üzerinde değerlendirilir.

Terminalde:

```text
Test Accuracy
Precision
Recall
F1-Score
```

değerleri gösterilir.

Örnek:

```text
========== TEST ==========

Test Accuracy: 78.40%
```

Her sınıf için Classification Report da terminale yazdırılır.

---

# Confusion Matrix

Test tahminleri kullanılarak Confusion Matrix oluşturulur.

Çıktı:

```text
outputs/
└── dinov3_confusion_matrix.png
```

Bu görsel modelin hangi STL-10 sınıflarını doğru veya yanlış ayırdığını incelemek için kullanılabilir.

---

# PCA Feature Space

DINOv3 tarafından oluşturulan feature vektörleri yüksek boyutludur.

Feature'lar PCA kullanılarak:

```text
DINOv3 Embedding
       ↓
      PCA
       ↓
2 Boyutlu Feature Space
```

haline getirilir.

Her STL-10 sınıfı grafik üzerinde ayrı olarak gösterilir.

Çıktı:

```text
outputs/
└── dinov3_pca.png
```

Bu görsel sayesinde farklı sınıfların DINOv3 feature space içerisinde nasıl dağıldığı gözlemlenebilir.

---

# Benzer Görüntü Bulma

Projede DINOv3 embedding'leri görüntü benzerliği için de kullanılmaktadır.

Akış:

```text
Query Image
     ↓
DINOv3 Embedding
     ↓
Cosine Similarity
     ↓
En Yakın Feature'lar
     ↓
En Benzer 5 Görüntü
```

`config.py` içerisinde:

```python
QUERY_INDEX = 0
TOP_K = 5
```

kullanılmaktadır.

`QUERY_INDEX` değiştirilerek farklı bir görüntü seçilebilir.

`TOP_K` değeri değiştirilerek gösterilecek benzer görüntü sayısı artırılabilir.

---

# Cosine Similarity

Query görüntüsünün embedding'i bütün test embedding'leriyle karşılaştırılır.

Sonuçlar:

```text
Query

↓

Similarity: 0.94
Similarity: 0.91
Similarity: 0.87
Similarity: 0.84
Similarity: 0.81
```

şeklinde sıralanır.

En yüksek similarity değerine sahip görüntüler query görüntüsüne feature space açısından en yakın görüntülerdir.

Çıktı:

```text
outputs/
└── dinov3_similar_images.png
```

---

# Projenin Çalıştırılması

Dosyalar sırasıyla çalıştırılmalıdır.

## 1. Feature Extraction

```bash
python 01_extract_features.py
```

## 2. Linear Probe Training

```bash
python 02_train_linear_probe.py
```

## 3. Feature Analysis

```bash
python 03_analyze_features.py
```

Tam akış:

```text
01_extract_features.py
        ↓
DINOv3 Frozen Backbone
        ↓
Train / Test Embeddings
        ↓
02_train_linear_probe.py
        ↓
Linear Probe
        ↓
03_analyze_features.py
        ↓
Classification
PCA
Similar Image Search
```

---

# Proje Çıktıları

Proje tamamlandığında:

```text
dinov3-project/
│
├── features/
│   ├── train_features.pt
│   └── test_features.pt
│
├── models/
│   └── dinov3_linear_probe.pt
│
├── outputs/
│   ├── dinov3_confusion_matrix.png
│   ├── dinov3_pca.png
│   └── dinov3_similar_images.png
│
├── requirements.txt
├── config.py
├── 01_extract_features.py
├── 02_train_linear_probe.py
└── 03_analyze_features.py
```

---

# Proje Akışının Özeti

```text
STL-10
   ↓
Pretrained DINOv3 ViT-S/16
   ↓
Backbone FROZEN
   ↓
Image Feature Extraction
   ↓
384-D Embeddings
   ↓
   ┌──────────────────────┐
   │                      │
   ↓                      ↓
Linear Probe        Feature Analysis
   ↓                 ↓           ↓
Classification     PCA     Cosine Similarity
   ↓                             ↓
Accuracy                  Similar Images
```

---

# Projenin Temel Çıktısı

Bu projede DINOv3 backbone yeniden eğitilmemektedir.

Modelin önceden öğrendiği feature representation'lar doğrudan kullanılmaktadır.

Projenin temel çıktıları:

```text
DINOv3 Embeddings
        +
Linear Probe Accuracy
        +
PCA Feature Space
        +
Similar Image Search
```

olacaktır.

Bu sayede pretrained DINOv3 feature'larının farklı görüntü işleme görevlerinde yeniden model eğitmeden nasıl kullanılabileceği uygulamalı olarak gösterilmektedir.