# ============================================================
# MODEL AYARLARI
# ============================================================

# Meta tarafından yayınlanan en küçük DINOv3 ViT modellerinden birini kullanıyoruz.
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

# DINOv3 backbone tamamen frozen kullanılacaktır.
FREEZE_BACKBONE = True


# ============================================================
# VERİ SETİ AYARLARI
# ============================================================

# Hugging Face üzerinde bulunan STL-10 veri setini kullanıyoruz.
DATASET_NAME = "mteb/stl10"

# CPU üzerinde hızlı çalışmak için 1000 training görüntüsü kullanıyoruz.
TRAIN_SAMPLES = 1000

# Final değerlendirme için 500 test görüntüsü kullanıyoruz.
TEST_SAMPLES = 500

# Görüntüler DINOv3 tarafından 224x224 boyutunda hazırlanacaktır.
IMAGE_SIZE = 224

# Feature extraction sırasında aynı anda işlenecek görüntü sayısıdır.
BATCH_SIZE = 8

# Windows multiprocessing problemi yaşamamak için worker sayısını 0 kullanıyoruz.
NUM_WORKERS = 0

# Rastgele işlemlerin her çalıştırmada aynı sonucu vermesini sağlar.
SEED = 42


# ============================================================
# STL-10 SINIFLARI
# ============================================================

# STL-10 veri setindeki 10 sınıfın isimleridir.
CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

# STL-10 veri setindeki toplam sınıf sayısıdır.
NUM_CLASSES = 10


# ============================================================
# LINEAR PROBE AYARLARI
# ============================================================

# Feature verisinin yüzde 20'sini validation için kullanıyoruz.
VAL_RATIO = 0.20

# Sadece linear classifier eğitildiği için 20 epoch hızlı şekilde çalışacaktır.
LINEAR_EPOCHS = 20

# Linear classifier için learning rate değeridir.
LINEAR_LR = 1e-3

# Linear classifier için regularization değeridir.
WEIGHT_DECAY = 1e-4

# Embedding training sırasında kullanılacak batch büyüklüğüdür.
LINEAR_BATCH_SIZE = 64


# ============================================================
# BENZER GÖRSEL ARAMA AYARLARI
# ============================================================

# Benzer görsel aramasında kullanılacak query görüntüsünün indeksidir.
QUERY_INDEX = 0

# Query görüntüsüne en yakın kaç görüntünün gösterileceğini belirler.
TOP_K = 5


# ============================================================
# DOSYA YOLLARI
# ============================================================

# Feature dosyalarının saklanacağı klasördür.
FEATURE_DIR = "features"

# Eğitilmiş linear probe modelinin saklanacağı klasördür.
MODEL_DIR = "models"

# Grafik ve analiz sonuçlarının saklanacağı klasördür.
OUTPUT_DIR = "outputs"

# Training embedding'lerinin kaydedileceği dosyadır.
TRAIN_FEATURE_PATH = "features/train_features.pt"

# Test embedding'lerinin kaydedileceği dosyadır.
TEST_FEATURE_PATH = "features/test_features.pt"

# Linear classifier'ın kaydedileceği dosyadır.
LINEAR_MODEL_PATH = "models/dinov3_linear_probe.pt"