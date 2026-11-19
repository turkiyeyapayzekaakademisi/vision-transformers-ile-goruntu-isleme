# ============================================================
# MODEL AYARLARI
# ============================================================

# COCO veri seti üzerinde pretrained DETR ResNet-50 modelini kullanıyoruz.
MODEL_NAME = "facebook/detr-resnet-50"


# ============================================================
# GÖRÜNTÜ AYARLARI
# ============================================================

# Object detection yapılacak görüntünün dosya yoludur.
INPUT_IMAGE = "input.png"


# ============================================================
# DETECTION AYARLARI
# ============================================================

# Ana detection çıktısında minimum yüzde 70 confidence kullanıyoruz.
CONFIDENCE_THRESHOLD = 0.70

# Farklı threshold değerlerinin etkisini karşılaştıracağız.
THRESHOLDS = [0.50, 0.70, 0.90]

# En güçlü kaç object query'nin terminalde gösterileceğini belirler.
TOP_QUERIES = 15


# ============================================================
# ÇIKTI AYARLARI
# ============================================================

# Tüm görsel ve analiz çıktılarının kaydedileceği klasördür.
OUTPUT_DIR = "outputs"