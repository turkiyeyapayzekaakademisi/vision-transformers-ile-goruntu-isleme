# ============================================================
# KLASÖR AYARLARI
# ============================================================

# Oxford-IIIT Pet veri setinin indirileceği klasör.
DATA_DIR = "data"

# Eğitilen Swin-Tiny modelinin kaydedileceği klasör.
MODEL_DIR = "models"

# Confusion matrix, tahmin ve feature map görsellerinin kaydedileceği klasör.
OUTPUT_DIR = "outputs"


# ============================================================
# MODEL AYARLARI
# ============================================================

# ImageNet-1k üzerinde pretrained Swin-Tiny modelinin timm içerisindeki adıdır.
MODEL_NAME = "swin_tiny_patch4_window7_224.ms_in1k"

# Oxford-IIIT Pet veri setinde toplam 37 farklı kedi ve köpek ırkı bulunur.
NUM_CLASSES = 37

# Swin-Tiny modelinin kullanacağı giriş görüntü boyutudur.
IMAGE_SIZE = 224


# ============================================================
# VERİ SETİ AYARLARI
# ============================================================

# CPU üzerinde çalışacağımız için batch size değerini 16 kullanıyoruz.
BATCH_SIZE = 16

# Windows multiprocessing problemi yaşamamak için DataLoader worker sayısını 0 yapıyoruz.
NUM_WORKERS = 0

# Train-validation ayrımının her çalıştırmada aynı olması için sabit seed kullanıyoruz.
SEED = 42

# Trainval veri setinin yüzde 20'sini validation için ayırıyoruz.
VAL_RATIO = 0.20


# ============================================================
# EĞİTİM AYARLARI
# ============================================================

# Modelin training veri setinin tamamını kaç kez göreceğini belirler.
EPOCHS = 3

# Pretrained Swin-Tiny modelini fine-tune etmek için düşük learning rate kullanıyoruz.
LEARNING_RATE = 5e-5

# Overfitting'i azaltmak için model ağırlıklarına regularization uygular.
WEIGHT_DECAY = 1e-4


# ============================================================
# ÇALIŞTIRMA AYARI
# ============================================================

# True olduğunda model eğitilir.
# False olduğunda daha önce kaydedilmiş model doğrudan yüklenir.
TRAIN_MODEL = True