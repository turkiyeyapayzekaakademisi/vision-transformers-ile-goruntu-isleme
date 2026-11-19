# ============================================================
# 1. KÜTÜPHANELER
# ============================================================

# PyTorch işlemleri için kullanılır.
import torch

# STL-10 veri setini Hugging Face üzerinden yüklemek için kullanılır.
from datasets import load_dataset

# Pretrained CLIP modelini ve processor'ı yüklemek için kullanılır.
from transformers import CLIPModel, AutoProcessor

# Batch işlemleri sırasında progress bar göstermek için kullanılır.
from tqdm import tqdm

# Classification metriklerini hesaplamak için kullanılır.
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Grafik ve örnek tahmin görsellerini oluşturmak için kullanılır.
import matplotlib.pyplot as plt

# Output klasörünü oluşturmak için kullanılır.
from pathlib import Path

# Proje parametrelerini config.py dosyasından alıyoruz.
from config import MODEL_NAME, DATASET_NAME, TEST_SAMPLES, BATCH_SIZE, SEED, CLASS_NAMES, CLASS_PROMPTS, NUM_CLASSES, OUTPUT_DIR


# ============================================================
# 2. ANA PROGRAM
# ============================================================

def main():

    # Random işlemlerin her çalıştırmada aynı sonucu vermesini sağlıyoruz.
    torch.manual_seed(SEED)

    # Bu projede yalnızca CPU kullanıyoruz.
    print("Device: CPU")

    # Output klasörü yoksa oluşturuyoruz.
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


    # ========================================================
    # 3. STL-10 TEST VERİ SETİ
    # ========================================================

    # STL-10 veri setinin resmi test bölümünü yüklüyoruz.
    dataset = load_dataset(DATASET_NAME, split="test")

    # Veri setindeki sınıfların dengeli şekilde dağılması için önce karıştırıyoruz.
    dataset = dataset.shuffle(seed=SEED)

    # CPU üzerinde hızlı çalışmak için belirlediğimiz sayıda görüntü seçiyoruz.
    dataset = dataset.select(range(min(TEST_SAMPLES, len(dataset))))

    # Veri seti bilgilerini terminalde gösteriyoruz.
    print("\n========== DATASET ==========")

    # Kullanılan test görüntüsü sayısını gösteriyoruz.
    print("Test Görüntüsü:", len(dataset))

    # STL-10 sınıf sayısını gösteriyoruz.
    print("Sınıf Sayısı:", NUM_CLASSES)


    # ========================================================
    # 4. PRETRAINED CLIP MODELİ
    # ========================================================

    # CLIP'in görüntü ve metinleri hazırlayan processor'ını yüklüyoruz.
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Pretrained CLIP ViT-B/32 modelini yüklüyoruz.
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Modeli CPU üzerinde kullanıyoruz.
    model = model.to("cpu")

    # Modeli evaluation moduna geçiriyoruz.
    model.eval()

    # CLIP modelinin tüm parametrelerini freeze ediyoruz.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Modelin toplam parametre sayısını hesaplıyoruz.
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    # Eğitilebilir parametre sayısını hesaplıyoruz.
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    # Model bilgilerini terminalde gösteriyoruz.
    print("\n========== CLIP ==========")

    # Kullanılan model adını gösteriyoruz.
    print("Model:", MODEL_NAME)

    # Toplam parametre sayısını gösteriyoruz.
    print(f"Toplam Parametre: {total_parameters:,}")

    # Eğitim yapılmadığı için bu değerin sıfır olmasını bekliyoruz.
    print(f"Eğitilebilir Parametre: {trainable_parameters:,}")

    # Modelin çalışma biçimini gösteriyoruz.
    print("Training: YOK")

    # Classification biçimini gösteriyoruz.
    print("Classification: ZERO-SHOT")


    # ========================================================
    # 5. ZERO-SHOT CLASSIFICATION
    # ========================================================

    # Gerçek sınıf indekslerini saklamak için liste oluşturuyoruz.
    all_labels = []

    # CLIP tahminlerini saklamak için liste oluşturuyoruz.
    all_predictions = []

    # Tahmin confidence değerlerini saklamak için liste oluşturuyoruz.
    all_confidences = []

    # Örnek görüntüleri daha sonra göstermek için liste oluşturuyoruz.
    sample_images = []

    # Örnek gerçek sınıfları saklamak için liste oluşturuyoruz.
    sample_labels = []

    # Örnek tahminleri saklamak için liste oluşturuyoruz.
    sample_predictions = []

    # Örnek confidence değerlerini saklamak için liste oluşturuyoruz.
    sample_confidences = []

    # Zero-shot classification başlangıcını gösteriyoruz.
    print("\n========== ZERO-SHOT CLASSIFICATION ==========")

    # Test veri setini batch'ler halinde dolaşıyoruz.
    for start_index in tqdm(range(0, len(dataset), BATCH_SIZE), desc="CLIP Zero-Shot"):

        # Mevcut batch'in son indeksini hesaplıyoruz.
        end_index = min(start_index + BATCH_SIZE, len(dataset))

        # Batch içerisindeki veri seti indekslerini oluşturuyoruz.
        batch_indices = range(start_index, end_index)

        # Batch içerisindeki PIL görüntülerini alıyoruz.
        images = [dataset[index]["image"].convert("RGB") for index in batch_indices]

        # Batch içerisindeki gerçek sınıf indekslerini alıyoruz.
        labels = torch.tensor([int(dataset[index]["label"]) for index in batch_indices], dtype=torch.long)

        # Görüntüleri ve 10 sınıf prompt'unu CLIP formatına dönüştürüyoruz.
        inputs = processor(text=CLASS_PROMPTS, images=images, return_tensors="pt", padding=True)

        # Gradient hesaplamadan CLIP modelini çalıştırıyoruz.
        with torch.inference_mode():

            # CLIP görüntüler ile sınıf metinleri arasındaki similarity skorlarını hesaplıyor.
            outputs = model(**inputs)

        # Her görüntünün 10 farklı text prompt ile similarity skorunu alıyoruz.
        logits_per_image = outputs.logits_per_image

        # Similarity skorlarını olasılık dağılımına dönüştürüyoruz.
        probabilities = logits_per_image.softmax(dim=1)

        # Her görüntü için en yüksek similarity değerine sahip sınıfı seçiyoruz.
        predictions = probabilities.argmax(dim=1)

        # Tahmin edilen sınıfın confidence değerini alıyoruz.
        confidences = probabilities.max(dim=1).values

        # Gerçek sınıfları listeye ekliyoruz.
        all_labels.extend(labels.tolist())

        # Model tahminlerini listeye ekliyoruz.
        all_predictions.extend(predictions.tolist())

        # Confidence değerlerini listeye ekliyoruz.
        all_confidences.extend(confidences.tolist())

        # Görselleştirme için en fazla 8 örnek saklıyoruz.
        if len(sample_images) < 8:

            # Kaç görüntü daha saklanabileceğini hesaplıyoruz.
            remaining = 8 - len(sample_images)

            # Örnek görüntüleri listeye ekliyoruz.
            sample_images.extend(images[:remaining])

            # Örnek gerçek sınıfları listeye ekliyoruz.
            sample_labels.extend(labels[:remaining].tolist())

            # Örnek tahminleri listeye ekliyoruz.
            sample_predictions.extend(predictions[:remaining].tolist())

            # Örnek confidence değerlerini listeye ekliyoruz.
            sample_confidences.extend(confidences[:remaining].tolist())


    # ========================================================
    # 6. TEST ACCURACY
    # ========================================================

    # Doğru tahmin edilen görüntü sayısını hesaplıyoruz.
    correct = sum(prediction == label for prediction, label in zip(all_predictions, all_labels))

    # Zero-shot test accuracy değerini hesaplıyoruz.
    accuracy = correct / len(all_labels)

    # Sonucu terminalde gösteriyoruz.
    print("\n========== TEST RESULT ==========")

    # Zero-shot accuracy değerini yüzde olarak gösteriyoruz.
    print(f"Zero-Shot Accuracy: {accuracy * 100:.2f}%")

    # Doğru tahmin sayısını gösteriyoruz.
    print(f"Doğru Tahmin: {correct}/{len(all_labels)}")


    # ========================================================
    # 7. CLASSIFICATION REPORT
    # ========================================================

    # Her sınıf için precision, recall ve F1-score değerlerini gösteriyoruz.
    print("\n========== CLASSIFICATION REPORT ==========")

    # Classification report'u terminale yazdırıyoruz.
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            digits=3,
            zero_division=0,
        )
    )


    # ========================================================
    # 8. CONFUSION MATRIX
    # ========================================================

    # Gerçek ve tahmin edilen sınıflardan confusion matrix hesaplıyoruz.
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    # Confusion matrix için figure oluşturuyoruz.
    figure, axis = plt.subplots(figsize=(10, 10))

    # Confusion matrix display oluşturuyoruz.
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    # Confusion matrix'i çiziyoruz.
    display.plot(
        ax=axis,
        xticks_rotation=45,
    )

    # Grafiğe zero-shot accuracy değerini ekliyoruz.
    plt.title(f"CLIP Zero-Shot Classification - Accuracy: {accuracy * 100:.2f}%")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Confusion matrix görselini kaydediyoruz.
    plt.savefig(f"{OUTPUT_DIR}/clip_confusion_matrix.png", dpi=200)

    # Grafik penceresini kapatıyoruz.
    plt.close()


    # ========================================================
    # 9. ÖRNEK ZERO-SHOT TAHMİNLER
    # ========================================================

    # Sekiz örnek görüntü için 2x4 grafik oluşturuyoruz.
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))

    # Sekiz örneğin tamamını dolaşıyoruz.
    for index, axis in enumerate(axes.flatten()):

        # STL-10 görüntüsünü gösteriyoruz.
        axis.imshow(sample_images[index])

        # Gerçek sınıfı alıyoruz.
        true_name = CLASS_NAMES[sample_labels[index]]

        # CLIP tarafından tahmin edilen sınıfı alıyoruz.
        predicted_name = CLASS_NAMES[sample_predictions[index]]

        # Confidence değerini yüzdeye dönüştürüyoruz.
        confidence = sample_confidences[index] * 100

        # Gerçek sınıf, tahmin ve confidence değerini gösteriyoruz.
        axis.set_title(
            f"Gerçek: {true_name}\n"
            f"Tahmin: {predicted_name}\n"
            f"Confidence: {confidence:.1f}%"
        )

        # Koordinat eksenlerini kapatıyoruz.
        axis.axis("off")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Örnek zero-shot tahminleri kaydediyoruz.
    plt.savefig(f"{OUTPUT_DIR}/clip_zero_shot_predictions.png", dpi=200)

    # Grafik penceresini kapatıyoruz.
    plt.close()


    # ========================================================
    # 10. PROGRAM SONU
    # ========================================================

    # Zero-shot classification işleminin tamamlandığını gösteriyoruz.
    print("\nZero-shot classification tamamlandı.")

    # Confusion matrix dosyasını gösteriyoruz.
    print(f"Confusion Matrix: {OUTPUT_DIR}/clip_confusion_matrix.png")

    # Örnek tahmin dosyasını gösteriyoruz.
    print(f"Örnek Tahminler: {OUTPUT_DIR}/clip_zero_shot_predictions.png")


# ============================================================
# 11. WINDOWS İÇİN GÜVENLİ BAŞLANGIÇ
# ============================================================

# Program doğrudan çalıştırıldığında main fonksiyonunu başlatıyoruz.
if __name__ == "__main__":

    # Zero-shot classification akışını çalıştırıyoruz.
    main()