# ============================================================
# MODEL AYARLARI
# ============================================================

# ImageNet üzerinde MAE yöntemiyle pretrained ViT-Base modelini kullanıyoruz.
MODEL_NAME = "facebook/vit-mae-base"

# Eğitilmiş MAE modelinin kaydedileceği klasördür.
MODEL_SAVE_DIR = "models/mae_stl10"

# Eğitim grafikleri ve reconstruction sonuçlarının kaydedileceği klasördür.
OUTPUT_DIR = "outputs"


# ============================================================
# VERİ SETİ AYARLARI
# ============================================================

# Hugging Face üzerinde bulunan STL-10 veri setini kullanıyoruz.
DATASET_NAME = "mteb/stl10"

# CPU üzerinde hızlı bir eğitim göstermek için ilk 500 görüntüyü kullanıyoruz.
TRAIN_SAMPLES = 500

# Verinin yüzde 10'unu validation için ayırıyoruz.
VAL_RATIO = 0.10

# Her training adımında modele aynı anda verilecek görüntü sayısıdır.
BATCH_SIZE = 2

# Windows multiprocessing problemi yaşamamak için worker sayısını 0 kullanıyoruz.
NUM_WORKERS = 0

# Veri ayrımının ve rastgele işlemlerin tekrar üretilebilir olmasını sağlar.
SEED = 42


# ============================================================
# MAE AYARLARI
# ============================================================

# MAE modelinin kullandığı giriş görüntü boyutudur.
IMAGE_SIZE = 224

# Görüntüler 16x16 boyutunda patch'lere ayrılır.
PATCH_SIZE = 16

# Patch'lerin yüzde 75'ini modelden gizliyoruz.
MASK_RATIO = 0.75


# ============================================================
# EĞİTİM AYARLARI
# ============================================================

# Modelin veri setini üç kez görmesini sağlıyoruz.
EPOCHS = 3

# Pretrained modeli bozmadan eğitime devam etmek için düşük learning rate kullanıyoruz.
LEARNING_RATE = 1e-5

# Model ağırlıklarına regularization uygular.
WEIGHT_DECAY = 0.05


# ============================================================
# RECONSTRUCTION AYARLARI
# ============================================================

# Final görselde kaç farklı STL-10 görüntüsü göstereceğimizi belirler.
RECONSTRUCTION_SAMPLES = 5