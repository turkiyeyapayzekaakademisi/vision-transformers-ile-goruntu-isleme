# ============================================================
# PROJE AYARLARI
# ============================================================

# Oxford-IIIT Pet veri setinin indirileceği klasör.
DATA_DIR = "data"

# Eğitilen modellerin ağırlıklarının kaydedileceği klasör.
MODEL_DIR = "models"

# Grafik ve örnek görüntü gibi çıktıların kaydedileceği klasör.
OUTPUT_DIR = "outputs"


# ============================================================
# VERİ SETİ AYARLARI
# ============================================================

# Oxford-IIIT Pet veri setinde 37 farklı kedi ve köpek ırkı bulunur.
NUM_CLASSES = 37

# ResNet50 ve DeiT modelleri için görüntüleri 224x224 boyutunda kullanacağız.
IMAGE_SIZE = 224

# Her training adımında modele aynı anda gönderilecek görüntü sayısıdır.
BATCH_SIZE = 32

# DataLoader'ın verileri paralel hazırlamak için kullanacağı worker sayısıdır.
NUM_WORKERS = 2

# Train-validation ayrımının her çalıştırmada aynı olmasını sağlar.
SEED = 42

# Trainval veri setinin yüzde 20'sini validation için ayırıyoruz.
VAL_RATIO = 0.20


# ============================================================
# GENEL EĞİTİM AYARLARI
# ============================================================

# Her model veri setinin tamamını 15 kez görecektir.
EPOCHS = 1

# ResNet50 teacher modelini fine-tune ederken kullanılacak learning rate.
TEACHER_LR = 1e-4

# Normal DeiT-Tiny modelini fine-tune ederken kullanılacak learning rate.
DEIT_LR = 5e-5

# Knowledge Distillation ile eğitilen DeiT-Tiny için kullanılacak learning rate.
DISTILL_LR = 5e-5

# Büyük ağırlıkları cezalandırarak overfitting'i azaltmaya yardımcı olur.
WEIGHT_DECAY = 1e-4


# ============================================================
# KNOWLEDGE DISTILLATION AYARLARI
# ============================================================

# Teacher modelin probability dağılımını yumuşatarak öğrenciye daha fazla bilgi aktarır.
TEMPERATURE = 2.0

# Gerçek label loss ile teacher loss arasındaki dengeyi belirler.
ALPHA = 0.5