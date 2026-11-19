# ============================================================
# 1. KÜTÜPHANELER
# ============================================================

# PyTorch işlemleri için kullanılır.
import torch

# Görüntüyü okumak için kullanılır.
from PIL import Image

# Pretrained DETR modelini ve image processor'ı kullanmak için kullanılır.
from transformers import AutoImageProcessor, DetrForObjectDetection

# Detection sonuçlarını çizmek için kullanılır.
import matplotlib.pyplot as plt

# Bounding box çizmek için kullanılır.
from matplotlib.patches import Rectangle

# Klasör ve dosya işlemleri için kullanılır.
from pathlib import Path

# Proje ayarlarını config.py dosyasından alıyoruz.
from config import MODEL_NAME, INPUT_IMAGE, CONFIDENCE_THRESHOLD, THRESHOLDS, TOP_QUERIES, OUTPUT_DIR


# ============================================================
# 2. LABEL İSMİNİ ALMA FONKSİYONU
# ============================================================

def get_label_name(model, label_id):

    # Model config içerisindeki COCO sınıf ismini alıyoruz.
    return model.config.id2label.get(int(label_id), f"class_{label_id}")


# ============================================================
# 3. DETECTION SONUÇLARINI ÇİZME FONKSİYONU
# ============================================================

def draw_detection(image, result, model, title, save_path):

    # Görüntüyü göstermek için figure oluşturuyoruz.
    figure, axis = plt.subplots(figsize=(12, 8))

    # Orijinal görüntüyü gösteriyoruz.
    axis.imshow(image)

    # Tespit edilen her nesneyi dolaşıyoruz.
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):

        # Bounding box koordinatlarını Python sayılarına dönüştürüyoruz.
        x_min, y_min, x_max, y_max = box.tolist()

        # Bounding box genişliğini hesaplıyoruz.
        width = x_max - x_min

        # Bounding box yüksekliğini hesaplıyoruz.
        height = y_max - y_min

        # Sınıf ismini model config içerisinden alıyoruz.
        class_name = get_label_name(model, label.item())

        # Bounding box çiziyoruz.
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

        # Sınıf ve confidence bilgisini bounding box üzerine yazıyoruz.
        axis.text(
            x_min,
            max(0, y_min - 5),
            f"{class_name} {score.item():.2f}",
            fontsize=10,
            backgroundcolor="white",
        )

    # Grafik başlığını ekliyoruz.
    axis.set_title(title)

    # Eksenleri kapatıyoruz.
    axis.axis("off")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Detection görselini kaydediyoruz.
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    # Grafik penceresini kapatıyoruz.
    plt.close()


# ============================================================
# 4. OBJECT QUERY ANALİZİ
# ============================================================

def analyze_queries(outputs, model):

    # Her query'nin sınıf skorlarını olasılığa dönüştürüyoruz.
    probabilities = outputs.logits.softmax(dim=-1)[0]

    # Son sınıf DETR içerisindeki no-object sınıfıdır.
    no_object_probabilities = probabilities[:, -1]

    # No-object dışındaki gerçek nesne sınıflarını alıyoruz.
    object_class_probabilities = probabilities[:, :-1]

    # Her query için en yüksek nesne sınıf skorunu hesaplıyoruz.
    best_class_scores, best_class_ids = object_class_probabilities.max(dim=-1)

    # Bir query'nin nesne içerme olasılığını hesaplıyoruz.
    objectness = 1.0 - no_object_probabilities

    # En yüksek objectness değerine sahip query'leri sıralıyoruz.
    top_query_indices = torch.argsort(objectness, descending=True)[:TOP_QUERIES]

    # Query analiz sonuçlarını saklamak için metin listesi oluşturuyoruz.
    lines = []

    # Başlık bilgisini ekliyoruz.
    lines.append("DETR OBJECT QUERY ANALİZİ")

    # Ayırıcı ekliyoruz.
    lines.append("=" * 80)

    # Toplam query sayısını ekliyoruz.
    lines.append(f"Toplam Object Query: {outputs.logits.shape[1]}")

    # Query logits boyutunu ekliyoruz.
    lines.append(f"Logits Shape: {tuple(outputs.logits.shape)}")

    # Bounding box Tensor boyutunu ekliyoruz.
    lines.append(f"Bounding Box Shape: {tuple(outputs.pred_boxes.shape)}")

    # Boş satır ekliyoruz.
    lines.append("")

    # En güçlü object query'leri dolaşıyoruz.
    for rank, query_index in enumerate(top_query_indices.tolist(), start=1):

        # Query'nin en olası sınıf indeksini alıyoruz.
        class_id = best_class_ids[query_index].item()

        # Query'nin sınıf ismini alıyoruz.
        class_name = get_label_name(model, class_id)

        # Query'nin objectness skorunu alıyoruz.
        query_objectness = objectness[query_index].item()

        # Query'nin en güçlü sınıf confidence değerini alıyoruz.
        class_score = best_class_scores[query_index].item()

        # Query'nin no-object olasılığını alıyoruz.
        no_object_score = no_object_probabilities[query_index].item()

        # Query'nin normalize bounding box tahminini alıyoruz.
        center_x, center_y, width, height = outputs.pred_boxes[0, query_index].tolist()

        # Query sonucunu terminal ve dosya için hazırlıyoruz.
        line = (
            f"{rank:02d}. Query {query_index:03d} | "
            f"Class: {class_name:<15} | "
            f"Class Confidence: {class_score:.4f} | "
            f"Objectness: {query_objectness:.4f} | "
            f"No-Object: {no_object_score:.4f} | "
            f"Box(cx,cy,w,h): "
            f"({center_x:.3f}, {center_y:.3f}, {width:.3f}, {height:.3f})"
        )

        # Query sonucunu listeye ekliyoruz.
        lines.append(line)

    # Query analizini terminalde gösteriyoruz.
    print("\n========== OBJECT QUERY ANALİZİ ==========")

    # Query satırlarını terminale yazdırıyoruz.
    for line in lines:
        print(line)

    # Query analizini txt dosyasına kaydediyoruz.
    with open(f"{OUTPUT_DIR}/detr_query_analysis.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


    # ========================================================
    # OBJECTNESS GRAFİĞİ
    # ========================================================

    # Bütün 100 query'nin objectness skorlarını CPU'ya taşıyoruz.
    query_scores = objectness.detach().cpu()

    # Query indekslerini oluşturuyoruz.
    query_indices = range(len(query_scores))

    # Objectness grafiği oluşturuyoruz.
    plt.figure(figsize=(14, 5))

    # Her query için objectness skorunu bar grafiği olarak çiziyoruz.
    plt.bar(query_indices, query_scores.numpy())

    # Grafiğe başlık ekliyoruz.
    plt.title("DETR Object Query Objectness Scores")

    # X eksenini isimlendiriyoruz.
    plt.xlabel("Object Query")

    # Y eksenini isimlendiriyoruz.
    plt.ylabel("1 - P(No Object)")

    # Y eksenini 0-1 aralığında tutuyoruz.
    plt.ylim(0, 1)

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Object query grafiğini kaydediyoruz.
    plt.savefig(f"{OUTPUT_DIR}/detr_query_objectness.png", dpi=200)

    # Grafik penceresini kapatıyoruz.
    plt.close()


# ============================================================
# 5. THRESHOLD KARŞILAŞTIRMA FONKSİYONU
# ============================================================

def compare_thresholds(image, outputs, processor, model):

    # Görüntünün orijinal height ve width değerlerini alıyoruz.
    target_sizes = torch.tensor([[image.height, image.width]])

    # Üç farklı threshold sonucunu yan yana gösterecek figure oluşturuyoruz.
    figure, axes = plt.subplots(1, len(THRESHOLDS), figsize=(7 * len(THRESHOLDS), 7))

    # Tek threshold kullanılırsa axes yapısını listeye çeviriyoruz.
    if len(THRESHOLDS) == 1:
        axes = [axes]

    # Bütün threshold değerlerini dolaşıyoruz.
    for axis, threshold in zip(axes, THRESHOLDS):

        # Ham DETR çıktılarını gerçek bounding box koordinatlarına dönüştürüyoruz.
        result = processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )[0]

        # Orijinal görüntüyü gösteriyoruz.
        axis.imshow(image)

        # Threshold üzerinde kalan detection sonuçlarını dolaşıyoruz.
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):

            # Bounding box koordinatlarını alıyoruz.
            x_min, y_min, x_max, y_max = box.tolist()

            # Bounding box boyutlarını hesaplıyoruz.
            width = x_max - x_min
            height = y_max - y_min

            # Detection sınıfını alıyoruz.
            class_name = get_label_name(model, label.item())

            # Bounding box oluşturuyoruz.
            rectangle = Rectangle(
                (x_min, y_min),
                width,
                height,
                fill=False,
                linewidth=2,
                edgecolor="red",
            )

            # Bounding box'ı görüntüye ekliyoruz.
            axis.add_patch(rectangle)

            # Detection sınıfı ve skorunu yazıyoruz.
            axis.text(
                x_min,
                max(0, y_min - 4),
                f"{class_name} {score.item():.2f}",
                fontsize=8,
                backgroundcolor="white",
            )

        # Threshold ve detection sayısını başlıkta gösteriyoruz.
        axis.set_title(
            f"Threshold = {threshold:.2f}\n"
            f"Detection = {len(result['scores'])}"
        )

        # Koordinat eksenlerini kapatıyoruz.
        axis.axis("off")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Threshold karşılaştırma görselini kaydediyoruz.
    plt.savefig(
        f"{OUTPUT_DIR}/detr_threshold_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    # Grafik penceresini kapatıyoruz.
    plt.close()


# ============================================================
# 6. ANA PROGRAM
# ============================================================

def main():

    # Bu projede yalnızca CPU kullanıyoruz.
    print("Device: CPU")

    # Output klasörü yoksa oluşturuyoruz.
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Girdi görüntüsünün gerçekten var olup olmadığını kontrol ediyoruz.
    if not Path(INPUT_IMAGE).exists():
        raise FileNotFoundError(
            f"{INPUT_IMAGE} bulunamadı. Proje klasörüne input.jpg isimli bir görüntü ekleyin."
        )


    # ========================================================
    # 7. GÖRÜNTÜYÜ YÜKLE
    # ========================================================

    # Detection yapılacak görüntüyü RGB formatında açıyoruz.
    image = Image.open(INPUT_IMAGE).convert("RGB")

    # Görüntü bilgilerini terminalde gösteriyoruz.
    print("\n========== IMAGE ==========")

    # Görüntü dosyasını gösteriyoruz.
    print("Dosya:", INPUT_IMAGE)

    # Orijinal görüntü boyutunu gösteriyoruz.
    print("Boyut:", image.size)


    # ========================================================
    # 8. PRETRAINED DETR MODELİ
    # ========================================================

    # DETR için gerekli image processor'ı PIL backend ile yüklüyoruz.
    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        backend="pil",
    )

    # COCO üzerinde pretrained DETR ResNet-50 modelini yüklüyoruz.
    model = DetrForObjectDetection.from_pretrained(MODEL_NAME)

    # Modeli CPU üzerinde kullanıyoruz.
    model = model.to("cpu")

    # Modeli inference moduna geçiriyoruz.
    model.eval()

    # Model parametrelerinin tamamını freeze ediyoruz.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Modelin toplam parametre sayısını hesaplıyoruz.
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    # Eğitilebilir parametre sayısını hesaplıyoruz.
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # Model bilgilerini terminalde gösteriyoruz.
    print("\n========== DETR MODEL ==========")

    # Model adını gösteriyoruz.
    print("Model:", MODEL_NAME)

    # Backbone bilgisini gösteriyoruz.
    print("Backbone: ResNet-50")

    # Transformer encoder katman sayısını gösteriyoruz.
    print("Encoder Layers:", model.config.encoder_layers)

    # Transformer decoder katman sayısını gösteriyoruz.
    print("Decoder Layers:", model.config.decoder_layers)

    # Object query sayısını gösteriyoruz.
    print("Object Queries:", model.config.num_queries)

    # Toplam parametre sayısını gösteriyoruz.
    print(f"Toplam Parametre: {total_parameters:,}")

    # Training yapılmadığı için eğitilebilir parametre sayısını gösteriyoruz.
    print(f"Eğitilebilir Parametre: {trainable_parameters:,}")

    # Training yapılmadığını gösteriyoruz.
    print("Training: YOK")


    # ========================================================
    # 9. GÖRÜNTÜYÜ MODEL İÇİN HAZIRLA
    # ========================================================

    # Görüntüyü DETR modelinin beklediği Tensor formatına dönüştürüyoruz.
    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    # Model giriş Tensor boyutunu terminalde gösteriyoruz.
    print("\n========== MODEL INPUT ==========")

    # Pixel Tensor boyutunu gösteriyoruz.
    print("Pixel Values:", tuple(inputs["pixel_values"].shape))


    # ========================================================
    # 10. DETR INFERENCE
    # ========================================================

    # Gradient hesaplamadan object detection inference yapıyoruz.
    with torch.inference_mode():

        # Görüntüyü DETR modelinden geçiriyoruz.
        outputs = model(**inputs)

    # Ham model çıktılarını terminalde gösteriyoruz.
    print("\n========== RAW DETR OUTPUT ==========")

    # 100 query'nin class logits Tensor boyutunu gösteriyoruz.
    print("Class Logits:", tuple(outputs.logits.shape))

    # 100 query'nin bounding box Tensor boyutunu gösteriyoruz.
    print("Bounding Boxes:", tuple(outputs.pred_boxes.shape))

    # Object query sayısını tekrar gösteriyoruz.
    print("Object Query Sayısı:", outputs.logits.shape[1])


    # ========================================================
    # 11. OBJECT QUERY ANALİZİ
    # ========================================================

    # Modelin oluşturduğu object query'leri inceliyoruz.
    analyze_queries(
        outputs,
        model,
    )


    # ========================================================
    # 12. FINAL OBJECT DETECTION
    # ========================================================

    # Görüntünün orijinal height ve width değerlerini Tensor olarak hazırlıyoruz.
    target_sizes = torch.tensor([[image.height, image.width]])

    # Ham model çıktılarını final detection sonuçlarına dönüştürüyoruz.
    result = processor.post_process_object_detection(
        outputs,
        threshold=CONFIDENCE_THRESHOLD,
        target_sizes=target_sizes,
    )[0]

    # Detection sonuç başlığını terminalde gösteriyoruz.
    print("\n========== FINAL DETECTIONS ==========")

    # Kullanılan confidence threshold değerini gösteriyoruz.
    print(f"Threshold: {CONFIDENCE_THRESHOLD:.2f}")

    # Toplam tespit sayısını gösteriyoruz.
    print("Detection Sayısı:", len(result["scores"]))

    # Tespit edilen bütün nesneleri terminalde gösteriyoruz.
    for index, (score, label, box) in enumerate(
        zip(result["scores"], result["labels"], result["boxes"]),
        start=1,
    ):

        # Sınıf ismini alıyoruz.
        class_name = get_label_name(model, label.item())

        # Bounding box koordinatlarını alıyoruz.
        x_min, y_min, x_max, y_max = box.tolist()

        # Detection sonucunu terminale yazdırıyoruz.
        print(
            f"{index}. "
            f"{class_name} | "
            f"Confidence: {score.item():.4f} | "
            f"Box: "
            f"({x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f})"
        )


    # ========================================================
    # 13. FINAL DETECTION GÖRSELİ
    # ========================================================

    # Final detection sonuçlarını görüntü üzerine çiziyoruz.
    draw_detection(
        image=image,
        result=result,
        model=model,
        title=f"DETR Object Detection - Threshold {CONFIDENCE_THRESHOLD:.2f}",
        save_path=f"{OUTPUT_DIR}/detr_detection.png",
    )


    # ========================================================
    # 14. THRESHOLD KARŞILAŞTIRMASI
    # ========================================================

    # Aynı model çıktısını farklı threshold değerleriyle karşılaştırıyoruz.
    compare_thresholds(
        image=image,
        outputs=outputs,
        processor=processor,
        model=model,
    )


    # ========================================================
    # 15. PROGRAM SONU
    # ========================================================

    # Projenin tamamlandığını terminalde gösteriyoruz.
    print("\n========== PROJE TAMAMLANDI ==========")

    # Final detection görselinin yolunu gösteriyoruz.
    print(f"Detection: {OUTPUT_DIR}/detr_detection.png")

    # Threshold karşılaştırma görselinin yolunu gösteriyoruz.
    print(f"Threshold Comparison: {OUTPUT_DIR}/detr_threshold_comparison.png")

    # Query objectness grafiğinin yolunu gösteriyoruz.
    print(f"Query Objectness: {OUTPUT_DIR}/detr_query_objectness.png")

    # Query analiz dosyasının yolunu gösteriyoruz.
    print(f"Query Analysis: {OUTPUT_DIR}/detr_query_analysis.txt")


# ============================================================
# 16. WINDOWS İÇİN GÜVENLİ PROGRAM BAŞLANGICI
# ============================================================

# Program doğrudan çalıştırıldığında ana fonksiyonu çalıştırıyoruz.
if __name__ == "__main__":

    # DETR projesini başlatıyoruz.
    main()