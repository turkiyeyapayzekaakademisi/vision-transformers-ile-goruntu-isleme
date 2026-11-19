# CLIP ile Zero-Shot Görüntü Sınıflandırma ve Text-to-Image Search

Bu projede **pretrained CLIP ViT-B/32** modeli kullanılarak STL-10 veri seti üzerinde iki farklı uygulama gerçekleştirilecektir.

Projede herhangi bir training veya fine-tuning yapılmayacaktır. CLIP modeli tamamen pretrained ve frozen olarak kullanılacaktır.

Proje kapsamında:

- Zero-Shot Image Classification
- Model performansının değerlendirilmesi
- Confusion Matrix oluşturulması
- Örnek tahminlerin görselleştirilmesi
- Text-to-Image Search

uygulamaları gerçekleştirilecektir.

---

# Proje Akışı

| Aşama | Ne Yapacağız? | Dosya |
|---|---|---|
| **Proje Altyapısı ve STL-10 Veri Setinin Hazırlanması** | Gerekli kütüphaneler kurulacak ve STL-10 test görüntüleri hazırlanacaktır. | `01_zero_shot_classification.py` |
| **Pretrained CLIP Modelinin Hazırlanması** | `openai/clip-vit-base-patch32` modeli yüklenecek ve tamamen frozen kullanılacaktır. | `01_zero_shot_classification.py` |
| **Image ve Text Embedding'lerinin Oluşturulması** | Görüntüler ve sınıf prompt'ları CLIP embedding space'e dönüştürülecektir. | `01_zero_shot_classification.py` |
| **Zero-Shot Image Classification** | Training yapılmadan görüntüler metin sınıfları ile eşleştirilerek sınıflandırılacaktır. | `01_zero_shot_classification.py` |
| **Zero-Shot Performansının Değerlendirilmesi** | Accuracy, Precision, Recall, F1-Score ve Confusion Matrix oluşturulacaktır. | `01_zero_shot_classification.py` |
| **Text-to-Image Search** | Doğal dil sorgusuna en yakın görüntüler CLIP embedding'leri kullanılarak bulunacaktır. | `02_text_to_image_search.py` |

---

# Kullanılan Veri Seti

Projede **STL-10** veri setinin `test` bölümü kullanılmaktadır.

```text
Sınıf Sayısı       : 10
Test Görüntüsü     : 8.000
Görüntü Boyutu     : 96x96
Kullanılan Örnek   : 500
Training           : Yok
```

STL-10 sınıfları:

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

CPU üzerinde daha hızlı çalışmak için:

```python
TEST_SAMPLES = 500
```

kullanılmaktadır.

Veri seti önce sabit bir seed ile karıştırılır ve daha sonra 500 görüntü seçilir.

---

# Kullanılan Model

Projede:

```text
openai/clip-vit-base-patch32
```

modeli kullanılmaktadır.

Model tamamen pretrained olarak çalışacaktır.

```text
CLIP
↓
FROZEN
↓
Training Yok
```

Model parametreleri:

```python
parameter.requires_grad = False
```

olarak ayarlanmaktadır.

Bu nedenle:

```text
Eğitilebilir Parametre : 0
```

olacaktır.

---

# Proje Yapısı

```text
clip-project/
│
├── outputs/
│
├── requirements.txt
├── config.py
├── 01_zero_shot_classification.py
└── 02_text_to_image_search.py
```

Dosyaların görevleri:

```text
requirements.txt
→ Projede kullanılan Python kütüphanelerini içerir.

config.py
→ Model, veri seti, sınıf prompt'ları ve arama parametrelerini içerir.

01_zero_shot_classification.py
→ STL-10 görüntülerini CLIP ile zero-shot olarak sınıflandırır,
  performans metriklerini ve Confusion Matrix'i oluşturur.

02_text_to_image_search.py
→ Doğal dil sorgusunu CLIP text embedding'e dönüştürür
  ve en benzer görüntüleri bulur.
```

---

# Virtual Environment Oluşturma

Proje klasörüne girin:

```bash
cd clip-project
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
torchvision
transformers
datasets
Pillow
tqdm
matplotlib
scikit-learn
```

Kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

`torchvision`, vision modellerinin image processor işlemlerinde gerekli olabileceği için projeye dahil edilmiştir.

---

# Proje Parametreleri

Temel ayarlar `config.py` içerisinde bulunmaktadır.

```text
MODEL_NAME      = openai/clip-vit-base-patch32

TEST_SAMPLES    = 500
BATCH_SIZE      = 8
NUM_WORKERS     = 0

SEED            = 42

TEXT_QUERY      = a photo of a dog
TOP_K           = 5
```

Proje CPU üzerinde çalışacak şekilde hazırlanmıştır.

---

# Zero-Shot Sınıf Prompt'ları

STL-10 sınıfları CLIP modeline doğal dil metinleri olarak verilmektedir.

```text
a photo of an airplane
a photo of a bird
a photo of a car
a photo of a cat
a photo of a deer
a photo of a dog
a photo of a horse
a photo of a monkey
a photo of a ship
a photo of a truck
```

Bu prompt'lar `config.py` içerisinde tanımlanmaktadır.

---

# 1. Zero-Shot Image Classification

İlk olarak:

```bash
python 01_zero_shot_classification.py
```

çalıştırılır.

Akış:

```text
STL-10 Test Images
        ↓
CLIP Image Encoder
        ↓
Image Embedding
        ↕
10 Class Prompts
        ↓
Text Embeddings
        ↓
Image-Text Similarity
        ↓
En Yüksek Skor
        ↓
Zero-Shot Prediction
```

Bu aşamada herhangi bir model eğitimi yapılmaz.

```text
Epoch          : 0
Backpropagation: Yok
Fine-Tuning    : Yok
```

---

# Zero-Shot Classification

Her görüntü CLIP Image Encoder tarafından embedding'e dönüştürülür.

Aynı zamanda 10 sınıf prompt'u CLIP Text Encoder tarafından embedding'e dönüştürülür.

```text
Image Embedding
       ↕
Text Embeddings
```

Similarity skorları karşılaştırılır.

Örneğin:

```text
a photo of a dog      → 0.87
a photo of a cat      → 0.42
a photo of a horse    → 0.31
a photo of a car      → 0.05
```

En yüksek similarity değerine sahip sınıf model tahmini olarak seçilir.

---

# Model Performansının Değerlendirilmesi

500 test görüntüsü üzerinde:

```text
Zero-Shot Accuracy
Precision
Recall
F1-Score
```

hesaplanmaktadır.

Terminal çıktısı örneği:

```text
========== TEST RESULT ==========

Zero-Shot Accuracy: 74.20%

Doğru Tahmin: 371/500
```

Ayrıca her sınıf için Classification Report terminale yazdırılır.

---

# Confusion Matrix

Zero-shot tahminleri kullanılarak Confusion Matrix oluşturulur.

Çıktı:

```text
outputs/
└── clip_confusion_matrix.png
```

Bu görsel CLIP'in hangi sınıfları doğru veya yanlış eşleştirdiğini incelemek için kullanılabilir.

---

# Örnek Zero-Shot Tahminler

Test veri setinden örnek görüntüler alınarak:

```text
Gerçek Sınıf
Tahmin Edilen Sınıf
Confidence
```

birlikte gösterilir.

Çıktı:

```text
outputs/
└── clip_zero_shot_predictions.png
```

---

# 2. Text-to-Image Search

Zero-shot classification sonrasında:

```bash
python 02_text_to_image_search.py
```

çalıştırılır.

Bu dosyada doğal dil sorgusu kullanılarak görüntü araması yapılır.

Örneğin:

```text
"a photo of a dog"
```

sorgusu kullanılabilir.

Akış:

```text
Text Query
    ↓
CLIP Text Encoder
    ↓
Text Embedding
    ↕
Image Embeddings
    ↓
Cosine Similarity
    ↓
Top 5 Images
```

---

# Text Query

Arama sorgusu `config.py` içerisinde:

```python
TEXT_QUERY = "a photo of a dog"
```

olarak tanımlanmaktadır.

Farklı sorgular kullanılabilir.

Örneğin:

```text
a photo of an airplane

a photo of a horse

a vehicle

an animal

a ship
```

---

# Benzer Görüntülerin Bulunması

Text embedding bütün görüntü embedding'leri ile karşılaştırılır.

Örnek terminal çıktısı:

```text
========== SEARCH RESULTS ==========

Query: a photo of a dog

1. Class: dog   | Similarity: 0.9124
2. Class: dog   | Similarity: 0.8841
3. Class: dog   | Similarity: 0.8512
4. Class: dog   | Similarity: 0.8237
5. Class: horse | Similarity: 0.7185
```

En yüksek similarity değerine sahip görüntüler doğal dil sorgusuna en yakın sonuçlar olarak seçilir.

---

# Arama Sonuçlarının Görselleştirilmesi

Bulunan en yakın görüntüler yan yana gösterilir.

```text
Query: "a photo of a dog"

#1        #2        #3        #4        #5

Dog       Dog       Dog       Dog       Horse
0.91      0.88      0.85      0.82      0.71
```

Çıktı:

```text
outputs/
└── clip_text_to_image_search.png
```

---

# Projenin Çalıştırılması

Dosyalar sırasıyla çalıştırılabilir.

## 1. Zero-Shot Classification

```bash
python 01_zero_shot_classification.py
```

## 2. Text-to-Image Search

```bash
python 02_text_to_image_search.py
```

Tam akış:

```text
01_zero_shot_classification.py
        ↓
STL-10 Test Images
        ↓
CLIP
        ↓
Zero-Shot Classification
        ↓
Accuracy
        ↓
Classification Report
        ↓
Confusion Matrix
        ↓
Sample Predictions


02_text_to_image_search.py
        ↓
Natural Language Query
        ↓
CLIP Text Embedding
        ↓
Image-Text Similarity
        ↓
Top 5 Images
```

---

# Proje Çıktıları

Proje tamamlandığında:

```text
clip-project/
│
├── outputs/
│   ├── clip_confusion_matrix.png
│   ├── clip_zero_shot_predictions.png
│   └── clip_text_to_image_search.png
│
├── requirements.txt
├── config.py
├── 01_zero_shot_classification.py
└── 02_text_to_image_search.py
```

---

# Proje Akışının Özeti

```text
STL-10
   ↓
Pretrained CLIP ViT-B/32
   ↓
Training YOK
   ↓
   ┌─────────────────────┐
   │                     │
   ↓                     ↓
Image Encoder        Text Encoder
   ↓                     ↓
Image Embedding      Text Embedding
   │                     │
   └──── Similarity ─────┘
             ↓
      ┌──────┴──────┐
      ↓             ↓
Zero-Shot       Text-to-Image
Classification  Search
      ↓             ↓
Accuracy        Top 5 Images
```

---

# Projenin Temel Çıktısı

Bu projede model STL-10 veri seti üzerinde yeniden eğitilmemektedir.

CLIP'in pretrained image ve text representation'ları doğrudan kullanılmaktadır.

Projenin temel çıktıları:

```text
Zero-Shot Classification
        +
Zero-Shot Accuracy
        +
Confusion Matrix
        +
Text-to-Image Search
```

olacaktır.