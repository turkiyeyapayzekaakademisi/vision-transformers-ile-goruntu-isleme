# ============================================================
# MODEL AYARLARI
# ============================================================

# OpenAI tarafından yayınlanan pretrained CLIP ViT-B/32 modelini kullanıyoruz.
MODEL_NAME = "openai/clip-vit-base-patch32"


# ============================================================
# VERİ SETİ AYARLARI
# ============================================================

# Hugging Face üzerinde bulunan STL-10 veri setini kullanıyoruz.
DATASET_NAME = "mteb/stl10"

# CPU üzerinde hızlı çalışmak için test setinden 500 görüntü kullanıyoruz.
TEST_SAMPLES = 500

# Aynı anda CLIP modeline gönderilecek görüntü sayısını belirliyoruz.
BATCH_SIZE = 8

# Windows multiprocessing problemi yaşamamak için worker kullanmıyoruz.
NUM_WORKERS = 0

# Veri setinin her çalıştırmada aynı şekilde karıştırılmasını sağlar.
SEED = 42


# ============================================================
# STL-10 SINIFLARI
# ============================================================

# STL-10 veri setindeki 10 sınıfı doğru indeks sırasıyla tanımlıyoruz.
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

# Veri setindeki toplam sınıf sayısını belirtiyoruz.
NUM_CLASSES = 10


# ============================================================
# ZERO-SHOT PROMPT'LARI
# ============================================================

# Her STL-10 sınıfını CLIP'e doğal dil cümlesi olarak veriyoruz.
CLASS_PROMPTS = [
    "a photo of an airplane",
    "a photo of a bird",
    "a photo of a car",
    "a photo of a cat",
    "a photo of a deer",
    "a photo of a dog",
    "a photo of a horse",
    "a photo of a monkey",
    "a photo of a ship",
    "a photo of a truck",
]


# ============================================================
# TEXT-TO-IMAGE SEARCH AYARLARI
# ============================================================

# Görsel aramasında kullanılacak doğal dil sorgusunu belirliyoruz.
TEXT_QUERY = "a photo of a dog"

# Metin sorgusuna en yakın kaç görüntünün gösterileceğini belirliyoruz.
TOP_K = 5


# ============================================================
# DOSYA AYARLARI
# ============================================================

# Tüm görsel çıktıların kaydedileceği klasörü belirliyoruz.
OUTPUT_DIR = "outputs"