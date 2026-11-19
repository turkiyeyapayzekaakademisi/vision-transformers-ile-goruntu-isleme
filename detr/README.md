# DETR ile Transformer Tabanlı Nesne Tespiti

Bu projede **pretrained DETR (DEtection TRansformer)** modeli kullanılarak gerçek bir görüntü üzerinde nesne tespiti gerçekleştirilecektir.

Projede model eğitimi veya fine-tuning yapılmayacaktır. COCO veri seti üzerinde önceden eğitilmiş:

```text
facebook/detr-resnet-50
```

modeli kullanılacaktır.

Proje kapsamında:

- Görüntü üzerinde nesne tespiti
- Sınıf ve bounding box tahminlerinin alınması
- Object Query çıktılarının incelenmesi
- Confidence Threshold değerlerinin karşılaştırılması
- Detection sonuçlarının görselleştirilmesi

işlemleri gerçekleştirilecektir.

---

# Proje Akışı

| Aşama | Ne Yapacağız? |
|---|---|
| **Proje Altyapısı ve Girdi Görüntüsünün Hazırlanması** | Gerekli kütüphaneler kurulacak ve `input.jpg` görüntüsü hazırlanacaktır. |
| **Pretrained DETR Modelinin Hazırlanması** | COCO üzerinde pretrained `facebook/detr-resnet-50` modeli yüklenecektir. |
| **DETR ile Object Detection** | Görüntü üzerinde sınıf ve bounding box tahminleri oluşturulacaktır. |
| **Object Query Çıktılarının İncelenmesi** | DETR tarafından üretilen 100 object query analiz edilecektir. |
| **Confidence Threshold Karşılaştırması** | `0.50`, `0.70` ve `0.90` threshold değerlerinin detection sonuçlarına etkisi incelenecektir. |
| **Detection Sonuçlarının Görselleştirilmesi** | Bounding box, sınıf ve confidence bilgileri görüntü üzerinde gösterilecektir. |

---

# Kullanılan Model

Projede:

```text
facebook/detr-resnet-50
```

modeli kullanılmaktadır.

Temel proje ayarları:

```text
Backbone        : ResNet-50
Object Queries  : 100
Training        : Yok
Fine-Tuning     : Yok
Device          : CPU
```

Model tamamen pretrained olarak kullanılmaktadır.

```text
Input Image
    ↓
Pretrained DETR
    ↓
100 Object Query
    ↓
Class Prediction
    +
Bounding Box Prediction
    ↓
Confidence Threshold
    ↓
Final Detections
```

---

# Girdi Görüntüsü

Projede:

```text
input.jpg
```

dosyası kullanılmaktadır.

Görüntü içerisinde object detection sonucunu rahat gözlemleyebilmek için birden fazla nesne bulunmaktadır.

Örneğin:

```text
person
dog
cat
bicycle
car
bus
backpack
```

gibi COCO veri setinde bulunan nesneler tercih edilmiştir.

---

# Proje Yapısı

```text
detr-project/
│
├── input.jpg
├── outputs/
│
├── requirements.txt
├── config.py
└── detr_object_detection.py
```

Dosyaların görevleri:

```text
requirements.txt
→ Projede kullanılan Python kütüphanelerini içerir.

config.py
→ Model, confidence threshold ve dosya ayarlarını içerir.

input.jpg
→ Object detection yapılacak girdi görüntüsüdür.

detr_object_detection.py
→ Pretrained DETR modelini yükler, inference gerçekleştirir,
  object query'leri analiz eder ve sonuçları görselleştirir.
```

---

# Virtual Environment Oluşturma

Proje klasörüne girin:

```bash
cd detr-project
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
transformers>=5.0.0
timm
Pillow
matplotlib
```

Kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

---

# Proje Parametreleri

Temel ayarlar `config.py` içerisinde bulunmaktadır.

```text
MODEL_NAME             = facebook/detr-resnet-50

INPUT_IMAGE            = input.jpg

CONFIDENCE_THRESHOLD   = 0.70

THRESHOLDS             = [0.50, 0.70, 0.90]

TOP_QUERIES            = 15

OUTPUT_DIR             = outputs
```

---

# Pretrained DETR Modelinin Hazırlanması

Model:

```text
facebook/detr-resnet-50
```

olarak yüklenmektedir.

Projede model eğitilmediği için bütün parametreler freeze edilmektedir.

```text
DETR
↓
FROZEN
↓
Training Yok
```

Terminalde:

```text
Eğitilebilir Parametre: 0
Training: YOK
```

bilgileri görülür.

---

# Model Girdisinin Hazırlanması

`input.jpg` görüntüsü modelin image processor'ı ile hazırlanır.

```text
input.jpg
    ↓
Image Processor
    ↓
Pixel Tensor
    ↓
DETR
```

Model görüntüyü kendi beklediği formata dönüştürerek inference işlemini gerçekleştirir.

---

# DETR Inference

Projeyi çalıştırmak için:

```bash
python detr_object_detection.py
```

komutu kullanılır.

Ana akış:

```text
input.jpg
    ↓
ResNet-50 Backbone
    ↓
Feature Map
    ↓
Transformer Encoder
    ↓
Transformer Decoder
    ↓
100 Object Query
    ↓
Class + Bounding Box
    ↓
Final Detection
```

---

# Ham DETR Çıktıları

Inference sonrasında model iki temel çıktı üretmektedir.

```text
Class Logits
Bounding Boxes
```

Örnek terminal çıktısı:

```text
========== RAW DETR OUTPUT ==========

Class Logits: (1, 100, ...)
Bounding Boxes: (1, 100, 4)
Object Query Sayısı: 100
```

Buradaki:

```text
100
```

değeri modelin kullandığı object query sayısını temsil eder.

---

# Object Query Analizi

DETR toplam:

```text
100 Object Query
```

üretmektedir.

Her query için:

```text
Class
Class Confidence
Objectness
No-Object Probability
Bounding Box
```

bilgileri incelenmektedir.

Örnek:

```text
Query 72
Class: person
Class Confidence: 0.9971
Objectness: 0.9992
No-Object: 0.0008
```

En güçlü query'ler terminalde gösterilir.

Ayrıca sonuçlar:

```text
outputs/detr_query_analysis.txt
```

dosyasına kaydedilir.

---

# Object Query Objectness Grafiği

Modelin ürettiği 100 object query için:

```text
1 - P(No Object)
```

değeri hesaplanmaktadır.

Bu değer query'nin gerçek bir nesne içerme ihtimalini temsil etmek için kullanılmaktadır.

Çıktı:

```text
outputs/detr_query_objectness.png
```

dosyasına kaydedilir.

---

# Final Object Detection

Ana detection işleminde:

```text
CONFIDENCE_THRESHOLD = 0.70
```

kullanılmaktadır.

Threshold değerinin üzerinde kalan detection sonuçları görüntü üzerine çizilir.

Her detection için:

```text
Sınıf
Confidence
Bounding Box
```

bilgileri gösterilir.

Örnek:

```text
person | Confidence: 0.9971

dog    | Confidence: 0.9632

car    | Confidence: 0.9518
```

Final çıktı:

```text
outputs/detr_detection.png
```

dosyasına kaydedilir.

---

# Confidence Threshold Karşılaştırması

Projede üç farklı threshold değeri kullanılmaktadır.

```text
0.50
0.70
0.90
```

Aynı DETR çıktısı bu üç threshold ile tekrar değerlendirilir.

```text
Threshold 0.50
↓
Daha fazla detection

Threshold 0.70
↓
Daha seçici detection

Threshold 0.90
↓
Yalnızca yüksek confidence değerleri
```

Sonuçlar yan yana gösterilir.

Çıktı:

```text
outputs/detr_threshold_comparison.png
```

dosyasına kaydedilir.

---

# Detection Görseli

Final görsel üzerinde:

```text
Bounding Box
Class Name
Confidence Score
```

birlikte gösterilmektedir.

Örneğin:

```text
┌───────────────────────┐
│                       │
│   ┌──────────────┐    │
│   │    person    │    │
│   └──────────────┘    │
│        0.99           │
│                       │
│       ┌──────┐        │
│       │ dog  │        │
│       └──────┘        │
│        0.96           │
│                       │
└───────────────────────┘
```

---

# Projenin Çalıştırılması

Proje klasöründe:

```bash
python detr_object_detection.py
```

komutunu çalıştırın.

Tam akış:

```text
input.jpg
    ↓
Pretrained DETR
    ↓
Inference
    ↓
100 Object Query
    ↓
Query Analysis
    ↓
Confidence Threshold
    ↓
Final Detections
    ↓
Bounding Box Visualization
```

---

# Proje Çıktıları

Proje tamamlandığında:

```text
detr-project/
│
├── input.jpg
│
├── outputs/
│   ├── detr_detection.png
│   ├── detr_threshold_comparison.png
│   ├── detr_query_objectness.png
│   └── detr_query_analysis.txt
│
├── requirements.txt
├── config.py
└── detr_object_detection.py
```

---

# Çıktı Dosyaları

## `detr_detection.png`

Ana object detection sonucunu içerir.

```text
Image
+
Bounding Boxes
+
Classes
+
Confidence Scores
```

---

## `detr_threshold_comparison.png`

Üç farklı confidence threshold sonucunu yan yana gösterir.

```text
0.50 | 0.70 | 0.90
```

---

## `detr_query_objectness.png`

100 object query'nin objectness skorlarını gösterir.

---

## `detr_query_analysis.txt`

En güçlü query'lerin:

```text
Query ID
Class
Confidence
Objectness
No-Object Probability
Bounding Box
```

bilgilerini içerir.

---

# Proje Akışının Özeti

```text
Input Image
     ↓
Pretrained DETR
     ↓
ResNet-50 Backbone
     ↓
Transformer Encoder
     ↓
Transformer Decoder
     ↓
100 Object Queries
     ↓
Class + Bounding Box
     ↓
Confidence Threshold
     ↓
Final Object Detection
     ↓
Bounding Box Visualization
```

---

# Projenin Temel Çıktısı

Bu projede herhangi bir training veya fine-tuning yapılmamaktadır.

Pretrained DETR modeli doğrudan object detection amacıyla kullanılmaktadır.

Projenin temel çıktıları:

```text
Object Detection
       +
Object Query Analysis
       +
Confidence Threshold Comparison
       +
Bounding Box Visualization
```

olacaktır.