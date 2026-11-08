# DeiT ve Knowledge Distillation ile Görüntü Sınıflandırma

Bu projede **Data-Efficient Image Transformer (DeiT)** ve **Knowledge Distillation** yaklaşımını gerçek bir görüntü sınıflandırma problemi üzerinde uygulayacağız.

Projenin temel amacı, aynı veri seti üzerinde üç farklı modeli eğiterek sonuçlarını karşılaştırmaktır:

1. **ResNet50 Teacher Model**
2. **Normal DeiT-Tiny Model**
3. **Knowledge Distillation ile eğitilmiş Distilled DeiT-Tiny Model**

Bu sayede bir Vision Transformer modelinin yalnızca gerçek etiketlerden öğrenmesi ile güçlü bir teacher modelden ek bilgi alarak öğrenmesi arasındaki farkı gözlemleyeceğiz.

---

# 1. Proje Tanımı

Projede **Oxford-IIIT Pet** veri setindeki kedi ve köpek ırklarını sınıflandıracağız.

İlk olarak ImageNet üzerinde önceden eğitilmiş bir **ResNet50** modelini Oxford-IIIT Pet veri setine fine-tune edeceğiz.

Bu model daha sonra **Teacher Model** olarak kullanılacaktır.

Ardından aynı veri seti üzerinde iki farklı DeiT modeli eğiteceğiz.

İlk model:

```text
DeiT-Tiny
```

yalnızca gerçek sınıf etiketlerini kullanarak öğrenir.

İkinci model:

```text
Distilled DeiT-Tiny
```

hem gerçek sınıf etiketlerinden hem de ResNet50 teacher modelinin tahminlerinden öğrenir.

Eğitim sonunda üç model aynı test veri setinde karşılaştırılacaktır.

---

| Aşama | Ne Yapacağız? | Dosyalar |
|---|---|---|
| **Proje Altyapısı ve Veri Setinin Hazırlanması** | Sanal ortamı kuracağız, gerekli kütüphaneleri yükleyeceğiz ve Oxford-IIIT Pet veri setini train/validation/test olarak hazırlayacağız. | `requirements.txt`, `config.py`, `data_utils.py`, `01_dataset.py` |
| **ResNet50 Teacher Modelinin Eğitilmesi** | ImageNet pretrained ResNet50 modelini 37 sınıfa uyarlayıp teacher model olarak fine-tune edeceğiz. | `train_utils.py`, `02_train_resnet_teacher.py` |
| **DeiT-Tiny Modelinin Eğitilmesi** | Aynı veri setinde normal DeiT-Tiny modelini yalnızca gerçek etiketlerle eğiteceğiz. | `03_train_deit.py` |
| **Knowledge Distillation ile Distilled DeiT Eğitimi** | ResNet50 teacher çıktıları ve gerçek etiketleri birlikte kullanarak Distilled DeiT-Tiny modelini eğiteceğiz. | `04_train_distilled_deit.py` |
| **Modellerin Karşılaştırılması** | ResNet50, normal DeiT-Tiny ve Distilled DeiT-Tiny modellerini aynı test setinde karşılaştıracağız. | `05_compare_models.py` |

# 2. Proje Blok Şeması

```text
                       Oxford-IIIT Pet
                           37 Sınıf
                              │
              ┌───────────────┼─────────────────┐
              │               │                 │
              ▼               ▼                 ▼
          ResNet50         DeiT-Tiny      Distilled DeiT
              │               │                 │
              │               │                 │
      Gerçek Etiketler   Gerçek Etiketler       │
              │               │                 │
              ▼               ▼                 │
          Fine-Tune        Fine-Tune             │
              │                                 │
              │                                 │
              └──────── Teacher Bilgisi ────────┤
                                                │
                                     Gerçek Etiketler
                                                +
                                       Teacher Çıktısı
                                                │
                                                ▼
                                      Knowledge Distillation
                                                │
                                                ▼
                                      Distilled DeiT-Tiny


                         TEST SET
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
      ResNet50          DeiT-Tiny       Distilled DeiT
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                  Accuracy Karşılaştırması
```

Knowledge Distillation bölümünün temel yapısı ise şöyledir:

```text
                         IMAGE
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
             ResNet50          Distilled DeiT
             Teacher               Student
                │                     │
                │              ┌──────┴──────┐
                │              ▼             ▼
                │            [CLS]         [DIST]
                │              │             │
                │              ▼             │
                │       Gerçek Etiket        │
                │                            │
                └──── Teacher Bilgisi ──────►│
```

`[CLS]` token gerçek sınıf etiketlerinden öğrenirken `[DIST]` token teacher modelin ürettiği bilgiyi öğrenir.

---

# 3. Kullanılan Veri Seti

Projede **Oxford-IIIT Pet Dataset** kullanılmaktadır.

Veri seti farklı kedi ve köpek ırklarından oluşan gerçek görüntüler içerir.

Temel özellikleri:

```text
Toplam sınıf sayısı : 37
Görüntü türü        : Kedi ve köpek fotoğrafları
Problem             : Multi-Class Image Classification
Training kaynağı    : trainval
Test kaynağı        : test
Görüntü boyutu      : 224x224
```

Veri setinde birbirine görsel olarak benzeyen birçok hayvan ırkı bulunmaktadır.

Örneğin:

```text
British Shorthair
Russian Blue
Persian
Egyptian Mau
```

gibi sınıfların birbirine benzemesi Knowledge Distillation açısından faydalıdır.

Teacher model yalnızca doğru sınıfı değil, diğer sınıflara verdiği olasılıkları da student modele aktarabilir.

Örneğin:

```text
Gerçek sınıf:

British Shorthair
```

Teacher modelin tahmini:

```text
British Shorthair    %72
Russian Blue         %15
Persian               %8
Diğer                 %5
```

Student model böylece yalnızca doğru etiketi değil, sınıflar arasındaki benzerlik bilgisini de öğrenebilir.

---

# 4. Kullanılan Modeller

## ResNet50

ResNet50 projede **Teacher Model** olarak kullanılmaktadır.

```text
ImageNet Pretrained ResNet50
            │
            ▼
Oxford-IIIT Pet
            │
            ▼
      Fine-Tuning
            │
            ▼
     Teacher Model
```

ResNet50'nin ImageNet için kullanılan 1000 sınıflı classification katmanı kaldırılarak yerine 37 sınıflı yeni bir classification katmanı eklenir.

---

## DeiT-Tiny

DeiT-Tiny küçük ve hızlı bir Vision Transformer modelidir.

Projede normal DeiT modeli yalnızca gerçek etiketlerden öğrenmektedir.

```text
Image
  │
  ▼
DeiT-Tiny
  │
  ▼
Prediction
  │
  ▼
Gerçek Label
  │
  ▼
CrossEntropy Loss
```

Bu model Knowledge Distillation sonucunu değerlendirmek için **baseline model** olarak kullanılacaktır.

---

## Distilled DeiT-Tiny

Distilled DeiT-Tiny Knowledge Distillation için tasarlanmış DeiT modelidir.

Model içerisinde iki farklı classification token bulunmaktadır:

```text
[CLS]

[DIST]
```

`CLS` gerçek etiketlerden öğrenir.

`DIST` ise teacher model tarafından üretilen bilgilerden öğrenir.

Toplam loss:

```text
Total Loss
    =
Classification Loss
    +
Distillation Loss
```

Projede iki loss değeri eşit ağırlıkta kullanılmaktadır.

```text
ALPHA = 0.5
```

Yani:

```text
%50 Gerçek Label Bilgisi

+

%50 Teacher Bilgisi
```

---

# 5. Proje Yapısı

```text
deit_distillation/
│
├── data/
├── models/
├── outputs/
│
├── requirements.txt
├── config.py
├── data_utils.py
├── train_utils.py
│
├── 01_dataset.py
├── 02_train_resnet_teacher.py
├── 03_train_deit.py
├── 04_train_distilled_deit.py
└── 05_compare_models.py
```

Dosyaların görevleri:

```text
config.py
→ Projede kullanılan tüm eğitim parametrelerini içerir.

data_utils.py
→ Veri setini indirir ve Train / Validation / Test DataLoader'larını oluşturur.

train_utils.py
→ Ortak training ve evaluation fonksiyonlarını içerir.

01_dataset.py
→ Veri setini indirir, kontrol eder ve örnek görüntüleri gösterir.

02_train_resnet_teacher.py
→ ResNet50 Teacher modelini eğitir.

03_train_deit.py
→ Normal DeiT-Tiny modelini eğitir.

04_train_distilled_deit.py
→ ResNet50 Teacher kullanarak Distilled DeiT-Tiny modelini eğitir.

05_compare_models.py
→ Üç modeli aynı test veri setinde karşılaştırır.
```

---

# 6. Virtual Environment Oluşturma

Proje klasörünü oluşturun:

```bash
mkdir deit_distillation
```

Klasöre girin:

```bash
cd deit_distillation
```

Virtual environment oluşturun:

```bash
python -m venv .venv
```

Windows üzerinde aktifleştirin:

```bash
.venv\Scripts\activate
```

Linux veya macOS:

```bash
source .venv/bin/activate
```

pip'i güncelleyin:

```bash
python -m pip install --upgrade pip
```

---

# 7. Requirements

`requirements.txt`:

```txt
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

GPU kontrolü:

```bash
python -c "import torch; print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CUDA aktifse örnek çıktı:

```text
CUDA: True
GPU: NVIDIA ...
```

---

# 8. Proje Parametreleri

Temel eğitim parametreleri `config.py` içerisinde tutulmaktadır.

```text
IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 15

TEACHER_LR = 1e-4

DEIT_LR = 5e-5

DISTILL_LR = 5e-5

WEIGHT_DECAY = 1e-4

TEMPERATURE = 2.0

ALPHA = 0.5
```

### IMAGE_SIZE

```text
224x224
```

ResNet50 ve DeiT modellerinde standart giriş boyutu olarak kullanılmaktadır.

### BATCH_SIZE

```text
32
```

Her training adımında modele gönderilecek görüntü sayısını belirler.

### TEMPERATURE

```text
2.0
```

Teacher modelin probability dağılımını yumuşatarak sınıflar arasındaki bilginin student modele aktarılmasını kolaylaştırır.

### ALPHA

```text
0.5
```

Gerçek etiket loss'u ile teacher distillation loss'u arasındaki dengeyi belirler.

---

# 9. Adım 1 — Veri Setinin Hazırlanması

Çalıştırın:

```bash
python 01_dataset.py
```

Program:

```text
Oxford-IIIT Pet veri setini indirir.

Train / Validation / Test verilerini oluşturur.

Training görüntülerine augmentation uygular.

Örnek görüntüleri ekranda gösterir.

Dataset bilgilerini terminale yazdırır.
```

Beklenen görüntü Tensor yapısı:

```text
[32, 3, 224, 224]
```

Burada:

```text
32  → Batch Size

3   → RGB kanalları

224 → Görüntü yüksekliği

224 → Görüntü genişliği
```

---

# 10. Adım 2 — ResNet50 Teacher Eğitimi

Çalıştırın:

```bash
python 02_train_resnet_teacher.py
```

Akış:

```text
ImageNet Pretrained ResNet50
            │
            ▼
Classification Head
1000 sınıf → 37 sınıf
            │
            ▼
Oxford-IIIT Pet
            │
            ▼
Fine-Tuning
            │
            ▼
resnet50_teacher.pt
```

Eğitim sırasında terminalde:

```text
Train Loss

Train Accuracy

Validation Loss

Validation Accuracy
```

değerleri gösterilir.

En yüksek validation accuracy değerine sahip model:

```text
models/resnet50_teacher.pt
```

dosyasına kaydedilir.

---

# 11. Adım 3 — Normal DeiT-Tiny Eğitimi

Çalıştırın:

```bash
python 03_train_deit.py
```

Bu aşamada herhangi bir teacher kullanılmaz.

Model yalnızca gerçek label bilgisiyle eğitilir.

```text
Oxford-IIIT Pet
       │
       ▼
   DeiT-Tiny
       │
       ▼
CrossEntropy Loss
       │
       ▼
Gerçek Label
```

En iyi model:

```text
models/deit_tiny.pt
```

dosyasına kaydedilir.

Bu model daha sonra Knowledge Distillation sonucuyla karşılaştırılacaktır.

---

# 12. Adım 4 — Knowledge Distillation

Çalıştırın:

```bash
python 04_train_distilled_deit.py
```

Bu aşamada daha önce eğittiğimiz ResNet50 teacher modeli yüklenir.

Teacher ağırlıkları artık değiştirilmez.

```text
requires_grad = False
```

Teacher yalnızca student modele bilgi üretir.

Her görüntü aynı anda:

```text
ResNet50 Teacher

ve

Distilled DeiT Student
```

modellerine gönderilir.

Teacher:

```text
37 sınıf için logits
```

üretir.

Student ise:

```text
CLS Head

DIST Head
```

olmak üzere iki çıktı üretir.

`CLS Head` gerçek label bilgisinden öğrenir.

`DIST Head` ResNet50 teacher çıktılarından öğrenir.

---

# 13. Knowledge Distillation Loss

Gerçek etiket için:

```text
Classification Loss
=
CrossEntropy(
    Student CLS Prediction,
    Ground Truth
)
```

Teacher bilgisi için:

```text
Distillation Loss
=
KL Divergence(
    Student DIST Prediction,
    Teacher Prediction
)
```

Toplam loss:

```text
Total Loss
=
ALPHA × Classification Loss
+
(1 - ALPHA) × Distillation Loss
```

Projede:

```text
ALPHA = 0.5
```

olduğu için:

```text
Total Loss
=
%50 Gerçek Label Loss

+

%50 Teacher Loss
```

kullanılmaktadır.

---

# 14. Temperature Mantığı

Teacher'ın normal tahmini:

```text
British Shorthair  %97
Russian Blue        %2
Persian             %1
```

çok keskin olabilir.

Temperature uygulandığında örneğin:

```text
British Shorthair  %70
Russian Blue       %20
Persian            %10
```

gibi daha yumuşak bir dağılım elde edilebilir.

Bu sayede student model yalnızca doğru sınıfı değil, diğer sınıfların doğru sınıfa ne kadar benzediğini de öğrenebilir.

---

# 15. Adım 5 — Modellerin Karşılaştırılması

Tüm modeller eğitildikten sonra:

```bash
python 05_compare_models.py
```

çalıştırılır.

Üç model aynı test veri setinde değerlendirilir:

```text
ResNet50 Teacher

DeiT-Tiny

Distilled DeiT-Tiny
```

Örnek terminal çıktısı:

```text
================ MODEL KARŞILAŞTIRMA ================

ResNet50 Teacher    : XX.XX%

DeiT-Tiny           : XX.XX%

Distilled DeiT-Tiny : XX.XX%

Distillation farkı  : +X.XX puan

=====================================================
```

Sonuçların önceden belirlenmiş olması beklenmez.

Knowledge Distillation'ın etkisi gerçek test sonuçları üzerinden gözlemlenir.

---

# 16. Projenin Çalıştırılma Sırası

Projeyi ilk kez çalıştırırken aşağıdaki sırayı takip edin.

### 1. Veri setini hazırlayın

```bash
python 01_dataset.py
```

### 2. ResNet50 Teacher modelini eğitin

```bash
python 02_train_resnet_teacher.py
```

### 3. Normal DeiT-Tiny modelini eğitin

```bash
python 03_train_deit.py
```

### 4. Knowledge Distillation ile DeiT modelini eğitin

```bash
python 04_train_distilled_deit.py
```

### 5. Modelleri karşılaştırın

```bash
python 05_compare_models.py
```

---

# 17. Eğitim Akışının Özeti

```text
Oxford-IIIT Pet
      │
      ▼
ResNet50 Teacher
      │
      ▼
Fine-Tuning
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
Teacher Bilgisi         Normal DeiT
      │                      │
      ▼                      ▼
Distilled DeiT          Label Learning
      │                      │
      └──────────┬───────────┘
                 ▼
             Test Set
                 │
                 ▼
       Accuracy Karşılaştırması
```

Bu proje sonunda aşağıdaki kavramlar uygulamalı olarak öğrenilmiş olacaktır:

```text
Vision Transformer

Data-Efficient Image Transformer

Transfer Learning

Fine-Tuning

Teacher / Student Model

Knowledge Distillation

Distillation Token

Temperature

KL Divergence

Cross Entropy

Train / Validation / Test

Image Classification

Model Comparison
```

---

# 18. Projenin Temel Amacı

Bu projenin temel sorusu şudur:

> Aynı DeiT modeli güçlü bir teacher modelden ek bilgi aldığında, yalnızca gerçek etiketlerden öğrenen DeiT modeline göre nasıl bir sonuç üretir?

Normal DeiT:

```text
Veri
+
Gerçek Etiket
```

kullanırken Distilled DeiT:

```text
Veri
+
Gerçek Etiket
+
Teacher Bilgisi
```

kullanır.

Bu sayede **Knowledge Distillation yaklaşımının Vision Transformer modellerinde nasıl kullanıldığını** gerçek bir görüntü sınıflandırma problemi üzerinde gözlemleyebiliriz.
