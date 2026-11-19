# Deformable DETR ile Nesne Tespiti

Bu projede **pretrained Deformable DETR** modeli kullanılarak gerçek bir görüntü üzerinde nesne tespiti gerçekleştirilmektedir.

Projede model eğitimi veya fine-tuning yapılmaz. Amaç, Deformable DETR'nin nesne tespiti akışını ve **reference point** yapısını uygulamalı olarak incelemektir.

Kullanılan model:

```text
SenseTime/deformable-detr-with-box-refine
```

---

# Proje Akışı

```text
input.jpg
    ↓
Pretrained Deformable DETR
    ↓
Multi-Scale Feature Maps
    ↓
Reference Points
    ↓
Deformable Attention
    ↓
Object Queries
    ↓
Class + Bounding Box
    ↓
Final Detection
```

Projede iki temel çıktı oluşturulur:

1. Nesne tespiti sonucu
2. Reference point görselleştirmesi

---

# Proje Yapısı

```text
deformable-detr-project/
│
├── input.jpg
├── outputs/
│
├── requirements.txt
├── config.py
└── deformable_detr.py
```

Dosyaların görevleri:

```text
requirements.txt
→ Projede kullanılan Python kütüphanelerini içerir.

config.py
→ Model, threshold ve dosya ayarlarını içerir.

input.jpg
→ Nesne tespiti yapılacak girdi görüntüsüdür.

deformable_detr.py
→ Modeli yükler, inference gerçekleştirir,
  detection sonuçlarını ve reference point'leri görselleştirir.
```

---

# Virtual Environment Oluşturma

Proje klasörüne girin:

```bash
cd deformable-detr-project
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

# Kullanılan Model

Projede:

```text
SenseTime/deformable-detr-with-box-refine
```

modeli kullanılmaktadır.

Temel özellikler:

```text
Backbone          : ResNet-50
Feature Levels    : 4
Object Queries    : 300
Box Refinement    : True
Training          : Yok
Fine-Tuning       : Yok
Device            : CPU
```

Model COCO üzerinde pretrained olarak kullanılmaktadır.

---

# Girdi Görüntüsü

Proje klasöründe:

```text
input.jpg
```

isimli bir görüntü bulunmalıdır.

Görüntü içerisinde insan, araç, hayvan veya günlük nesneler gibi birden fazla nesnenin bulunması detection sonucunu daha anlaşılır hale getirir.

---

# Proje Parametreleri

Temel ayarlar `config.py` dosyasında bulunmaktadır.

```python
MODEL_NAME = "SenseTime/deformable-detr-with-box-refine"

INPUT_IMAGE = "input.jpg"

CONFIDENCE_THRESHOLD = 0.70

REFERENCE_POINT_COUNT = 10

OUTPUT_DIR = "outputs"
```

---

# Modelin Hazırlanması

Pretrained image processor ve Deformable DETR modeli yüklenir.

```text
input.jpg
    ↓
Image Processor
    ↓
Pixel Tensor
    ↓
Deformable DETR
```

Model yalnızca inference amacıyla kullanılır.

```text
Training: Yok
Fine-Tuning: Yok
```

---

# Deformable DETR ile Nesne Tespiti

Görüntü modelden geçirildikten sonra her object query için sınıf ve bounding box tahminleri üretilir.

Ham model çıktıları daha sonra gerçek görüntü koordinatlarına dönüştürülür.

```text
Model Output
    ↓
Class Scores
    +
Bounding Boxes
    ↓
Confidence Threshold
    ↓
Final Detections
```

Projede kullanılan confidence threshold:

```text
0.70
```

Bu değerin altında kalan tahminler final detection sonucuna dahil edilmez.

---

# Detection Sonuçlarının Görselleştirilmesi

Her detection için:

```text
Class Name
Confidence Score
Bounding Box
```

bilgileri görüntü üzerine çizilir.

Örnek terminal çıktısı:

```text
========== DETECTIONS ==========

Detection Sayısı: 5

1. person | Confidence: 0.9951 | Box: [120.4, 80.3, 310.5, 520.7]
2. dog    | Confidence: 0.9724 | Box: [410.2, 340.8, 610.5, 570.1]
3. car    | Confidence: 0.9418 | Box: [650.7, 250.5, 1040.6, 510.2]
```

Detection görseli:

```text
outputs/deformable_detr_detection.png
```

dosyasına kaydedilir.

---

# Reference Point Analizi

Deformable DETR'de object query'ler bütün feature map üzerinde tam attention uygulamak yerine belirli **reference point** bölgeleri çevresindeki sınırlı sayıdaki sampling point'e odaklanır.

Projede en güçlü object query'ler seçilir ve bu query'lerin:

```text
Initial Reference Points
        ↓
Final Reference Points
```

konumları görüntü üzerinde gösterilir.

Bu sayede decoder katmanları boyunca reference point'lerin nasıl refine edildiği gözlemlenebilir.

---

# Reference Point Görselleştirmesi

Reference point çıktısı iki aşamayı yan yana gösterir:

```text
Initial Reference Points

Final Reference Points
```

Final görselde başlangıç ve final noktaları arasındaki değişim çizgilerle gösterilir.

Çıktı:

```text
outputs/deformable_detr_reference_points.png
```

dosyasına kaydedilir.

---

# Projenin Çalıştırılması

Proje klasöründe:

```bash
python deformable_detr.py
```

komutunu çalıştırın.

Program sırasıyla:

```text
1. input.jpg görüntüsünü yükler
2. Pretrained Deformable DETR modelini yükler
3. Görüntüyü modele uygun formata dönüştürür
4. Inference gerçekleştirir
5. Detection sonuçlarını oluşturur
6. Bounding box'ları görselleştirir
7. Reference point'leri görselleştirir
```

---

# Beklenen Terminal Çıktısı

```text
Device: CPU

========== MODEL ==========
Model: SenseTime/deformable-detr-with-box-refine
Feature Levels: 4
Object Queries: 300
Sampling Points: 4
Box Refinement: True

========== DETECTIONS ==========
Detection Sayısı: 5

1. person | Confidence: 0.9951
2. dog    | Confidence: 0.9724
3. car    | Confidence: 0.9418

========== PROJE TAMAMLANDI ==========
Detection: outputs/deformable_detr_detection.png
Reference Points: outputs/deformable_detr_reference_points.png
```

---

# Proje Çıktıları

Proje tamamlandığında klasör yapısı:

```text
deformable-detr-project/
│
├── input.jpg
│
├── outputs/
│   ├── deformable_detr_detection.png
│   └── deformable_detr_reference_points.png
│
├── requirements.txt
├── config.py
└── deformable_detr.py
```

---

# Çıktı Dosyaları

## `deformable_detr_detection.png`

Pretrained Deformable DETR modelinin nesne tespiti sonucunu içerir.

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

## `deformable_detr_reference_points.png`

Seçilen object query'lerin başlangıç ve final reference point konumlarını gösterir.

```text
Initial Reference Points
          ↓
Reference Point Refinement
          ↓
Final Reference Points
```

---

# Projenin Özeti

Bu projede herhangi bir training veya fine-tuning yapılmamaktadır.

Pretrained Deformable DETR modeli kullanılarak:

```text
Object Detection
        +
Multi-Scale Features
        +
Reference Points
        +
Bounding Box Prediction
```

akışı incelenmektedir.

Projenin temel amacı, yalnızca detection sonucunu görmek değil, Deformable DETR'nin ayırt edici yapılarından biri olan **reference point tabanlı deformable attention yaklaşımını** uygulamalı olarak gözlemlemektir.
