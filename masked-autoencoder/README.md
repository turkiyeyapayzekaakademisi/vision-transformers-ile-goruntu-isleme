# Masked Autoencoder (MAE) ile Görüntü Reconstruction

Bu projede **pretrained Masked Autoencoder (MAE)** modeli kullanılarak self-supervised learning yaklaşımı uygulanacaktır.

Projede STL-10 veri setindeki görüntüler kullanılacak ancak **sınıf etiketleri kullanılmayacaktır**. Görüntülerin büyük bir bölümü maskelenecek ve MAE modelinin eksik patch'leri yeniden oluşturması sağlanacaktır.

Pretrained MAE modeli STL-10 görüntüleri üzerinde **3 epoch self-supervised olarak eğitime devam ettirilecek**, ardından kaydedilen model kullanılarak reconstruction sonuçları görselleştirilecektir.

---

# Proje Akışı

| Aşama | Ne Yapacağız? | Dosya |
|---|---|---|
| **Proje Altyapısı ve Veri Setinin Hazırlanması** | Gerekli kütüphaneler kurulacak ve STL-10 train görüntüleri hazırlanacaktır. | `train_mae.py` |
| **Pretrained MAE Modelinin Hazırlanması** | Önceden eğitilmiş `facebook/vit-mae-base` modeli yüklenecektir. | `train_mae.py` |
| **Görüntülerin Maskelenmesi** | Görüntüler patch'lere ayrılacak ve patch'lerin %75'i rastgele maskelenecektir. | `train_mae.py` |
| **MAE Modelinin Eğitilmesi** | Model eksik patch'leri tahmin edecek ve reconstruction loss üzerinden 3 epoch eğitilecektir. | `train_mae.py` |
| **Eğitilmiş Modelin Yüklenmesi** | Validation loss açısından en iyi model tekrar yüklenecektir. | `reconstruct_mae.py` |
| **Reconstruction** | Maskelenmiş görüntülerdeki eksik patch'ler model tarafından tahmin edilecektir. | `reconstruct_mae.py` |
| **Sonuçların Görselleştirilmesi** | Original, Masked ve Reconstructed görüntüler yan yana gösterilecektir. | `reconstruct_mae.py` |

---

# Kullanılan Veri Seti

Projede **STL-10** veri setinin `train` bölümü kullanılmaktadır.

```text
Train Görüntü Sayısı : 5.000
Görüntü Boyutu       : 96x96
Sınıf Sayısı         : 10
Kullanılan Label     : Yok
Model Giriş Boyutu   : 224x224
```

STL-10 görüntülerinde sınıf bilgileri bulunmasına rağmen MAE eğitimi sırasında bu etiketler kullanılmamaktadır.

Kod içerisinde yalnızca görüntüler alınır:

```python
image = self.dataset[index]["image"].convert("RGB")
```

Böylece eğitim tamamen **self-supervised** olarak gerçekleştirilir.

CPU üzerinde daha hızlı deneme yapabilmek için başlangıçta:

```python
TRAIN_SAMPLES = 500
```

kullanılmaktadır.

İstenirse:

```python
TRAIN_SAMPLES = 1000
```

veya veri setinin tamamı için:

```python
TRAIN_SAMPLES = 5000
```

olarak değiştirilebilir.

---

# Kullanılan Model

Projede pretrained:

```text
facebook/vit-mae-base
```

modeli kullanılmaktadır.

Temel proje ayarları:

```text
Image Size     : 224x224
Patch Size     : 16x16
Toplam Patch   : 196
Mask Ratio     : %75
Görünen Patch  : 49
Maskeli Patch  : 147
```

Model daha önce eğitilmiş ağırlıklarla başlatılır ve STL-10 görüntüleri üzerinde self-supervised eğitimine devam eder.

---

# Proje Yapısı

```text
mae-project/
│
├── models/
├── outputs/
│
├── requirements.txt
├── config.py
├── train_mae.py
└── reconstruct_mae.py
```

Dosyaların görevleri:

```text
requirements.txt
→ Projede kullanılan Python kütüphanelerini içerir.

config.py
→ Model, veri seti, masking ve training parametrelerini içerir.

train_mae.py
→ Veri setini hazırlar, pretrained MAE modelini yükler,
  self-supervised eğitimi gerçekleştirir ve modeli kaydeder.

reconstruct_mae.py
→ Eğitilmiş modeli yükler ve Original / Masked /
  Reconstructed görüntülerini oluşturur.
```

---

# Virtual Environment Oluşturma

Proje klasörüne girin:

```bash
cd mae-project
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

`requirements.txt` dosyası:

```text
torch
torchvision
transformers
datasets
Pillow
tqdm
matplotlib
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

---

# Proje Parametreleri

Temel ayarlar `config.py` içerisinde bulunmaktadır.

```text
MODEL_NAME             = facebook/vit-mae-base

TRAIN_SAMPLES           = 500
VAL_RATIO               = 0.10

IMAGE_SIZE              = 224
PATCH_SIZE              = 16
MASK_RATIO              = 0.75

BATCH_SIZE              = 2
EPOCHS                   = 3

LEARNING_RATE           = 1e-5
WEIGHT_DECAY            = 0.05

RECONSTRUCTION_SAMPLES  = 5
```

Proje CPU üzerinde çalışacak şekilde hazırlanmıştır.

---

# 1. MAE Modelinin Eğitilmesi

Training işlemini başlatmak için:

```bash
python train_mae.py
```

çalıştırılır.

Training akışı:

```text
STL-10 Görüntüleri
        ↓
Train / Validation
        ↓
224x224 Görüntü
        ↓
16x16 Patch'ler
        ↓
%75 Random Masking
        ↓
Pretrained MAE
        ↓
Encoder
        ↓
Decoder
        ↓
Eksik Patch Tahmini
        ↓
Reconstruction Loss
        ↓
Backpropagation
        ↓
3 Epoch
        ↓
En İyi Modeli Kaydet
```

---

# Self-Supervised Training

Bu projede klasik classification eğitimindeki gibi:

```text
Görüntü → Label → Loss
```

akışı kullanılmamaktadır.

Bunun yerine:

```text
Görüntü
   ↓
%75 Mask
   ↓
MAE
   ↓
Eksik Patch Tahmini
   ↓
Reconstruction Loss
```

akışı kullanılmaktadır.

Model training sırasında herhangi bir sınıf etiketi görmez.

Her epoch sonunda:

```text
Train Reconstruction Loss
Validation Reconstruction Loss
```

değerleri terminalde gösterilir.

Örnek:

```text
Epoch 1/3 | Train Reconstruction Loss: 0.3842 | Val Reconstruction Loss: 0.3517

Epoch 2/3 | Train Reconstruction Loss: 0.3014 | Val Reconstruction Loss: 0.2895

Epoch 3/3 | Train Reconstruction Loss: 0.2671 | Val Reconstruction Loss: 0.2583
```

Validation reconstruction loss azaldığında model otomatik olarak kaydedilir.

---

# Kaydedilen Model

En iyi model:

```text
models/
└── mae_stl10/
```

klasörüne kaydedilir.

Örnek yapı:

```text
models/
└── mae_stl10/
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

---

# Training Loss Grafiği

Training ve validation reconstruction loss değerleri grafik olarak kaydedilir.

Çıktı:

```text
outputs/
└── mae_training_loss.png
```

Bu grafik modelin eğitim süresince reconstruction hatasının nasıl değiştiğini takip etmek için kullanılabilir.

---

# 2. Reconstruction İşlemi

Model eğitildikten sonra:

```bash
python reconstruct_mae.py
```

çalıştırılır.

Bu dosyada tekrar training yapılmaz.

Akış:

```text
Kaydedilmiş MAE
       ↓
STL-10 Görüntüsü
       ↓
224x224
       ↓
%75 Mask
       ↓
MAE Reconstruction
       ↓
Original
Masked
Reconstructed
```

---

# Original Görüntü

Modele verilen orijinal görüntüdür.

```text
Original
```

Bu görüntü MAE'nin beklediği `224x224` boyutuna dönüştürülür.

---

# Masked Görüntü

Görüntü:

```text
14 x 14 = 196
```

patch'e ayrılır.

Patch'lerin yaklaşık:

```text
%75 → Masked
%25 → Visible
```

olacak şekilde rastgele bir kısmı gizlenir.

Yani model görüntünün yalnızca yaklaşık:

```text
49 patch
```

bölümünü görür.

---

# Reconstructed Görüntü

Model maskelenen patch'leri tahmin eder.

Final görselde:

```text
Görünen Patch'ler
        +
Modelin Tahmin Ettiği Patch'ler
        ↓
Reconstructed Image
```

oluşturulur.

Bu sayede modelin eksik görüntü bölgelerini ne kadar başarılı şekilde tamamladığı gözlemlenebilir.

---

# Tek Görüntü Reconstruction

İlk çıktı tek bir görüntü üzerinde:

```text
Original | Masked | Reconstructed
```

şeklinde oluşturulur.

Dosya:

```text
outputs/mae_single_reconstruction.png
```

---

# Birden Fazla Reconstruction

Modelin farklı görüntülerdeki davranışını görebilmek için birden fazla görüntü de birlikte gösterilir.

```text
Original       Masked       Reconstructed
------------------------------------------------
Image 1        Image 1      Image 1
Image 2        Image 2      Image 2
Image 3        Image 3      Image 3
Image 4        Image 4      Image 4
Image 5        Image 5      Image 5
```

Çıktı:

```text
outputs/mae_multiple_reconstructions.png
```

---

# Projenin Çalıştırılması

İlk olarak modeli eğitin:

```bash
python train_mae.py
```

Daha sonra reconstruction işlemini çalıştırın:

```bash
python reconstruct_mae.py
```

Çalıştırma sırası:

```text
1. train_mae.py
       ↓
2. models/mae_stl10
       ↓
3. reconstruct_mae.py
       ↓
4. Reconstruction Görselleri
```

---

# Proje Çıktıları

Proje tamamlandığında temel dosya yapısı:

```text
mae-project/
│
├── models/
│   └── mae_stl10/
│       ├── config.json
│       ├── model.safetensors
│       └── preprocessor_config.json
│
├── outputs/
│   ├── mae_training_loss.png
│   ├── mae_single_reconstruction.png
│   └── mae_multiple_reconstructions.png
│
├── requirements.txt
├── config.py
├── train_mae.py
└── reconstruct_mae.py
```

---

# Proje Akışının Özeti

```text
STL-10
5.000 Train Görüntüsü
        ↓
Label Kullanma
        ↓
İlk 500 Görüntüyü Seç
        ↓
Train / Validation
        ↓
224x224
        ↓
16x16 Patch
        ↓
196 Patch
        ↓
%75 Mask
        ↓
Pretrained ViT-MAE
        ↓
Reconstruction Loss
        ↓
3 Epoch Self-Supervised Training
        ↓
En İyi Modeli Kaydet
        ↓
Reconstruction
        ↓
Original | Masked | Reconstructed
```

---

# Projenin Temel Çıktısı

Bu projede classification accuracy veya confusion matrix kullanılmamaktadır.

Projenin temel çıktıları:

```text
Reconstruction Loss
        +
Original / Masked / Reconstructed Görselleri
```

olacaktır.

Modelin eğitiminde herhangi bir sınıf etiketi kullanılmadan, görüntünün maskelenmiş bölümleri tahmin edilerek self-supervised eğitim gerçekleştirilmektedir.