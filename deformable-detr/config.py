# ============================================================
# MODEL AYARLARI
# ============================================================

# COCO üzerinde pretrained Deformable DETR modelini kullanıyoruz.
MODEL_NAME = "SenseTime/deformable-detr-with-box-refine"


# ============================================================
# GÖRÜNTÜ AYARLARI
# ============================================================

# Object detection yapılacak görüntünün dosya yoludur.
INPUT_IMAGE = "input.png"


# ============================================================
# DETECTION AYARLARI
# ============================================================

# Minimum detection confidence değeridir.
CONFIDENCE_THRESHOLD = 0.70

# Reference point görselinde gösterilecek detection sayısıdır.
REFERENCE_POINT_COUNT = 10


# ============================================================
# ÇIKTI AYARLARI
# ============================================================

# Çıktı dosyalarının kaydedileceği klasördür.
OUTPUT_DIR = "outputs"