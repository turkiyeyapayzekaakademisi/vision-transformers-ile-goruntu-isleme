# ============================================================
# 1. KÜTÜPHANELER
# ============================================================

# PyTorch işlemleri için kullanılır.
import torch

# Görüntüyü okumak için kullanılır.
from PIL import Image

# Pretrained Deformable DETR modelini kullanmak için kullanılır.
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

# Sonuçları görselleştirmek için kullanılır.
import matplotlib.pyplot as plt

# Bounding box çizmek için kullanılır.
from matplotlib.patches import Rectangle

# Dosya ve klasör işlemleri için kullanılır.
from pathlib import Path

# Proje ayarlarını config.py dosyasından alıyoruz.
from config import MODEL_NAME, INPUT_IMAGE, CONFIDENCE_THRESHOLD, REFERENCE_POINT_COUNT, OUTPUT_DIR


# ============================================================
# 2. DETECTION GÖRSELİ
# ============================================================

def draw_detections(image, result, model):

    # Detection sonucunu göstermek için figure oluşturuyoruz.
    figure, axis = plt.subplots(figsize=(12, 8))

    # Orijinal görüntüyü gösteriyoruz.
    axis.imshow(image)

    # Tespit edilen bütün nesneleri dolaşıyoruz.
    for score, label, box in zip(
        result["scores"],
        result["labels"],
        result["boxes"],
    ):

        # Bounding box koordinatlarını alıyoruz.
        x_min, y_min, x_max, y_max = box.tolist()

        # Bounding box genişliğini hesaplıyoruz.
        width = x_max - x_min

        # Bounding box yüksekliğini hesaplıyoruz.
        height = y_max - y_min

        # Tahmin edilen sınıf ismini alıyoruz.
        class_name = model.config.id2label[label.item()]

        # Bounding box oluşturuyoruz.
        rectangle = Rectangle(
            (x_min, y_min),
            width,
            height,
            fill=False,
            linewidth=2.5,
            edgecolor="red",
        )

        # Bounding box'ı görüntüye ekliyoruz.
        axis.add_patch(rectangle)

        # Sınıf ve confidence bilgisini ekliyoruz.
        axis.text(
            x_min,
            max(0, y_min - 5),
            f"{class_name} {score.item():.2f}",
            fontsize=10,
            backgroundcolor="white",
        )

    # Grafik başlığını ekliyoruz.
    axis.set_title("Deformable DETR Object Detection")

    # Koordinat eksenlerini kapatıyoruz.
    axis.axis("off")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Detection sonucunu kaydediyoruz.
    plt.savefig(
        f"{OUTPUT_DIR}/deformable_detr_detection.png",
        dpi=200,
        bbox_inches="tight",
    )

    # Grafik penceresini kapatıyoruz.
    plt.close()


# ============================================================
# 3. REFERENCE POINT GÖRSELİ
# ============================================================

def draw_reference_points(image, outputs, result):

    # Final detection skorlarına göre en güçlü sonuçları seçiyoruz.
    detection_count = min(
        REFERENCE_POINT_COUNT,
        len(result["scores"]),
    )

    # Detection yoksa reference point görselini oluşturmuyoruz.
    if detection_count == 0:
        print("Reference point için detection bulunamadı.")
        return

    # Ham logits değerlerini probability değerlerine dönüştürüyoruz.
    probabilities = outputs.logits.softmax(dim=-1)[0]

    # No-object sınıfını çıkartıyoruz.
    object_probabilities = probabilities[:, :-1]

    # Her query için en yüksek sınıf skorunu alıyoruz.
    query_scores = object_probabilities.max(dim=-1).values

    # En güçlü query indekslerini seçiyoruz.
    query_indices = torch.argsort(
        query_scores,
        descending=True,
    )[:detection_count]

    # Başlangıç reference point'lerini alıyoruz.
    initial_points = outputs.init_reference_points[0].detach().cpu()

    # Decoder katmanlarındaki reference point'leri alıyoruz.
    decoder_points = outputs.intermediate_reference_points[0].detach().cpu()

    # Son decoder katmanını alıyoruz.
    final_points = decoder_points[-1]

    # İlgili query'lerin başlangıç noktalarını seçiyoruz.
    initial_points = initial_points[query_indices.cpu()]

    # İlgili query'lerin final noktalarını seçiyoruz.
    final_points = final_points[query_indices.cpu()]

    # Reference point'lerin merkez koordinatlarını alıyoruz.
    initial_x = initial_points[:, 0] * image.width
    initial_y = initial_points[:, 1] * image.height

    # Final reference point merkez koordinatlarını alıyoruz.
    final_x = final_points[:, 0] * image.width
    final_y = final_points[:, 1] * image.height

    # İki aşamayı yan yana göstermek için figure oluşturuyoruz.
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
    )

    # --------------------------------------------------------
    # INITIAL REFERENCE POINTS
    # --------------------------------------------------------

    # Orijinal görüntüyü gösteriyoruz.
    axes[0].imshow(image)

    # Başlangıç reference point'lerini çiziyoruz.
    axes[0].scatter(
        initial_x,
        initial_y,
        s=80,
        marker="x",
        linewidths=2,
    )

    # Grafik başlığını ekliyoruz.
    axes[0].set_title("Initial Reference Points")

    # Koordinat eksenlerini kapatıyoruz.
    axes[0].axis("off")


    # --------------------------------------------------------
    # FINAL REFERENCE POINTS
    # --------------------------------------------------------

    # Orijinal görüntüyü gösteriyoruz.
    axes[1].imshow(image)

    # Final reference point'leri çiziyoruz.
    axes[1].scatter(
        final_x,
        final_y,
        s=80,
        marker="x",
        linewidths=2,
    )

    # Initial ve final noktaları bir çizgiyle birleştiriyoruz.
    for x1, y1, x2, y2 in zip(
        initial_x,
        initial_y,
        final_x,
        final_y,
    ):

        # Reference point değişimini çiziyoruz.
        axes[1].plot(
            [x1, x2],
            [y1, y2],
            linewidth=1.5,
        )

    # Grafik başlığını ekliyoruz.
    axes[1].set_title("Final Reference Points")

    # Koordinat eksenlerini kapatıyoruz.
    axes[1].axis("off")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Reference point görselini kaydediyoruz.
    plt.savefig(
        f"{OUTPUT_DIR}/deformable_detr_reference_points.png",
        dpi=200,
        bbox_inches="tight",
    )

    # Grafik penceresini kapatıyoruz.
    plt.close()


# ============================================================
# 4. ANA PROGRAM
# ============================================================

def main():

    # Bu projede yalnızca CPU kullanıyoruz.
    print("Device: CPU")

    # Output klasörünü oluşturuyoruz.
    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    # Girdi görüntüsünün varlığını kontrol ediyoruz.
    if not Path(INPUT_IMAGE).exists():

        # Dosya bulunamadığında hata oluşturuyoruz.
        raise FileNotFoundError(
            f"{INPUT_IMAGE} bulunamadı."
        )


    # ========================================================
    # 5. GÖRÜNTÜYÜ YÜKLE
    # ========================================================

    # Girdi görüntüsünü RGB formatında açıyoruz.
    image = Image.open(
        INPUT_IMAGE
    ).convert("RGB")


    # ========================================================
    # 6. MODELİ YÜKLE
    # ========================================================

    # Deformable DETR image processor'ını yüklüyoruz.
    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    # Pretrained Deformable DETR modelini yüklüyoruz.
    model = DeformableDetrForObjectDetection.from_pretrained(
        MODEL_NAME
    )

    # Modeli CPU üzerinde kullanıyoruz.
    model = model.to("cpu")

    # Modeli inference moduna geçiriyoruz.
    model.eval()

    # Model bilgilerini terminalde gösteriyoruz.
    print("\n========== MODEL ==========")

    # Model adını gösteriyoruz.
    print("Model:", MODEL_NAME)

    # Feature level sayısını gösteriyoruz.
    print(
        "Feature Levels:",
        model.config.num_feature_levels,
    )

    # Object query sayısını gösteriyoruz.
    print(
        "Object Queries:",
        model.config.num_queries,
    )

    # Sampling point sayısını gösteriyoruz.
    print(
        "Sampling Points:",
        model.config.decoder_n_points,
    )

    # Box refinement bilgisini gösteriyoruz.
    print(
        "Box Refinement:",
        model.config.with_box_refine,
    )


    # ========================================================
    # 7. GÖRÜNTÜYÜ HAZIRLA
    # ========================================================

    # Görüntüyü modelin beklediği Tensor formatına dönüştürüyoruz.
    inputs = processor(
        images=image,
        return_tensors="pt",
    )


    # ========================================================
    # 8. INFERENCE
    # ========================================================

    # Gradient hesaplamadan inference gerçekleştiriyoruz.
    with torch.inference_mode():

        # Görüntüyü Deformable DETR modelinden geçiriyoruz.
        outputs = model(
            **inputs
        )


    # ========================================================
    # 9. DETECTION SONUÇLARI
    # ========================================================

    # Orijinal görüntü boyutunu hazırlıyoruz.
    target_sizes = torch.tensor(
        [[image.height, image.width]]
    )

    # Ham çıktıları gerçek bounding box koordinatlarına dönüştürüyoruz.
    result = processor.post_process_object_detection(
        outputs,
        threshold=CONFIDENCE_THRESHOLD,
        target_sizes=target_sizes,
    )[0]

    # Detection başlığını terminalde gösteriyoruz.
    print("\n========== DETECTIONS ==========")

    # Toplam detection sayısını gösteriyoruz.
    print(
        "Detection Sayısı:",
        len(result["scores"]),
    )

    # Detection sonuçlarını terminalde gösteriyoruz.
    for index, (score, label, box) in enumerate(
        zip(
            result["scores"],
            result["labels"],
            result["boxes"],
        ),
        start=1,
    ):

        # Tahmin edilen sınıf ismini alıyoruz.
        class_name = model.config.id2label[
            label.item()
        ]

        # Bounding box koordinatlarını alıyoruz.
        coordinates = [
            round(value, 1)
            for value in box.tolist()
        ]

        # Sonucu terminale yazdırıyoruz.
        print(
            f"{index}. "
            f"{class_name} | "
            f"Confidence: {score.item():.4f} | "
            f"Box: {coordinates}"
        )


    # ========================================================
    # 10. DETECTION GÖRSELİ
    # ========================================================

    # Detection sonuçlarını görüntü üzerine çiziyoruz.
    draw_detections(
        image,
        result,
        model,
    )


    # ========================================================
    # 11. REFERENCE POINT GÖRSELİ
    # ========================================================

    # Reference point değişimini görüntü üzerinde gösteriyoruz.
    draw_reference_points(
        image,
        outputs,
        result,
    )


    # ========================================================
    # 12. PROGRAM SONU
    # ========================================================

    # Projenin tamamlandığını terminalde gösteriyoruz.
    print("\n========== PROJE TAMAMLANDI ==========")

    # Detection görselinin yolunu gösteriyoruz.
    print(
        f"Detection: "
        f"{OUTPUT_DIR}/deformable_detr_detection.png"
    )

    # Reference point görselinin yolunu gösteriyoruz.
    print(
        f"Reference Points: "
        f"{OUTPUT_DIR}/deformable_detr_reference_points.png"
    )


# ============================================================
# 13. WINDOWS İÇİN GÜVENLİ BAŞLANGIÇ
# ============================================================

# Program doğrudan çalıştırıldığında main fonksiyonunu çalıştırıyoruz.
if __name__ == "__main__":

    # Projeyi başlatıyoruz.
    main()