# ============================================================
# 1. KÜTÜPHANELER
# ============================================================

# PyTorch işlemleri için kullanılır.
import torch

# Linear classifier oluşturmak için kullanılır.
from torch import nn

# Feature vektörlerini normalize etmek için kullanılır.
import torch.nn.functional as F

# STL-10 test görüntülerini tekrar yüklemek için kullanılır.
from datasets import load_dataset

# Classification sonuçlarını hesaplamak için kullanılır.
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Embedding'leri iki boyuta indirmek için kullanılır.
from sklearn.decomposition import PCA

# Grafik ve görüntü oluşturmak için kullanılır.
import matplotlib.pyplot as plt

# Proje ayarlarını config.py dosyasından alıyoruz.
from config import DATASET_NAME, TEST_SAMPLES, TEST_FEATURE_PATH, LINEAR_MODEL_PATH, OUTPUT_DIR, CLASS_NAMES, NUM_CLASSES, QUERY_INDEX, TOP_K, SEED


# ============================================================
# 2. ANA PROGRAM
# ============================================================

def main():

    # DINOv3 test embedding'lerini yüklüyoruz.
    data = torch.load(TEST_FEATURE_PATH, map_location="cpu", weights_only=False)

    # Test feature Tensor'unu alıyoruz.
    test_features = data["features"].float()

    # Test label Tensor'unu alıyoruz.
    test_labels = data["labels"].long()

    # Feature vektörlerini L2 normalize ediyoruz.
    normalized_features = F.normalize(test_features, p=2, dim=1)

    # Kaydedilmiş linear probe modelini yüklüyoruz.
    checkpoint = torch.load(LINEAR_MODEL_PATH, map_location="cpu", weights_only=False)

    # Training sırasında kullanılan embedding boyutunu alıyoruz.
    embedding_dim = checkpoint["embedding_dim"]

    # Linear classifier mimarisini tekrar oluşturuyoruz.
    classifier = nn.Linear(embedding_dim, NUM_CLASSES)

    # Eğitilmiş classifier ağırlıklarını yüklüyoruz.
    classifier.load_state_dict(checkpoint["state_dict"])

    # Classifier'ı evaluation moduna geçiriyoruz.
    classifier.eval()


    # ========================================================
    # 3. TEST CLASSIFICATION
    # ========================================================

    # Gradient hesaplamadan tahmin yapıyoruz.
    with torch.inference_mode():

        # Test embedding'lerini classifier'a gönderiyoruz.
        outputs = classifier(normalized_features)

        # En yüksek skorlu sınıfı prediction olarak seçiyoruz.
        predictions = outputs.argmax(dim=1)

    # Doğru tahmin edilen görüntü sayısını hesaplıyoruz.
    correct = (predictions == test_labels).sum().item()

    # Test accuracy değerini hesaplıyoruz.
    test_accuracy = correct / len(test_labels)

    # Final test sonucunu gösteriyoruz.
    print("========== TEST ==========")

    # Test accuracy değerini gösteriyoruz.
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Her sınıf için precision, recall ve F1-score değerlerini gösteriyoruz.
    print("\n========== CLASSIFICATION REPORT ==========")

    # Classification report'u terminale yazdırıyoruz.
    print(
        classification_report(
            test_labels.numpy(),
            predictions.numpy(),
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            digits=3,
            zero_division=0,
        )
    )


    # ========================================================
    # 4. CONFUSION MATRIX
    # ========================================================

    # Confusion matrix değerlerini hesaplıyoruz.
    matrix = confusion_matrix(
        test_labels.numpy(),
        predictions.numpy(),
        labels=list(range(NUM_CLASSES)),
    )

    # Confusion matrix figure oluşturuyoruz.
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
        cmap="Blues",
    )

    # Grafiğe başlık ekliyoruz.
    plt.title(f"DINOv3 Linear Probe - Test Accuracy: {test_accuracy * 100:.2f}%")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Confusion matrix görselini kaydediyoruz.
    plt.savefig(f"{OUTPUT_DIR}/dinov3_confusion_matrix.png", dpi=200)

    # Grafik penceresini kapatıyoruz.
    plt.close()


    # ========================================================
    # 5. PCA FEATURE SPACE
    # ========================================================

    # 384 boyutlu DINOv3 embedding'lerini iki boyuta indiriyoruz.
    pca = PCA(n_components=2, random_state=SEED)

    # PCA dönüşümünü hesaplıyoruz.
    features_2d = pca.fit_transform(normalized_features.numpy())

    # PCA scatter plot oluşturuyoruz.
    plt.figure(figsize=(10, 8))

    # Her STL-10 sınıfını ayrı ayrı çiziyoruz.
    for class_index, class_name in enumerate(CLASS_NAMES):

        # Mevcut sınıfa ait görüntülerin indekslerini seçiyoruz.
        class_mask = test_labels.numpy() == class_index

        # Mevcut sınıfın PCA noktalarını çiziyoruz.
        plt.scatter(
            features_2d[class_mask, 0],
            features_2d[class_mask, 1],
            label=class_name,
            alpha=0.7,
        )

    # Grafiğe başlık ekliyoruz.
    plt.title("DINOv3 Feature Space - PCA")

    # X eksenini isimlendiriyoruz.
    plt.xlabel("PCA 1")

    # Y eksenini isimlendiriyoruz.
    plt.ylabel("PCA 2")

    # Sınıf açıklamalarını gösteriyoruz.
    plt.legend()

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # PCA görselini kaydediyoruz.
    plt.savefig(f"{OUTPUT_DIR}/dinov3_pca.png", dpi=200)

    # Grafik penceresini kapatıyoruz.
    plt.close()


    # ========================================================
    # 6. COSINE SIMILARITY
    # ========================================================

    # Query olarak kullanılacak embedding'i seçiyoruz.
    query_feature = normalized_features[QUERY_INDEX]

    # Query ile bütün test embedding'leri arasındaki cosine similarity değerlerini hesaplıyoruz.
    similarities = normalized_features @ query_feature

    # Query görüntüsünün kendisini sonuçlardan çıkarmak için similarity değerini düşürüyoruz.
    similarities[QUERY_INDEX] = -1.0

    # En yüksek similarity değerine sahip görüntüleri seçiyoruz.
    top_indices = torch.topk(similarities, k=TOP_K).indices

    # STL-10 test split'ini tekrar yüklüyoruz.
    test_raw = load_dataset(DATASET_NAME, split="test")

    # Feature extraction sırasında kullandığımız aynı sıralamayı oluşturuyoruz.
    test_raw = test_raw.shuffle(seed=SEED)

    # Feature extraction sırasında kullandığımız test örneklerini seçiyoruz.
    test_raw = test_raw.select(range(min(TEST_SAMPLES, len(test_raw))))

    # Query görüntüsünü alıyoruz.
    query_image = test_raw[QUERY_INDEX]["image"].convert("RGB")

    # Query görüntüsünün sınıfını alıyoruz.
    query_label = int(test_labels[QUERY_INDEX])

    # Query ve benzer görüntüler için grafik oluşturuyoruz.
    figure, axes = plt.subplots(1, TOP_K + 1, figsize=(3 * (TOP_K + 1), 4))

    # Query görüntüsünü ilk sütunda gösteriyoruz.
    axes[0].imshow(query_image)

    # Query görüntüsünün sınıfını gösteriyoruz.
    axes[0].set_title(f"QUERY\n{CLASS_NAMES[query_label]}")

    # Query görüntüsünün eksenlerini kapatıyoruz.
    axes[0].axis("off")

    # En yakın görüntüleri sırayla gösteriyoruz.
    for position, dataset_index in enumerate(top_indices.tolist(), start=1):

        # Benzer görüntüyü veri setinden alıyoruz.
        image = test_raw[dataset_index]["image"].convert("RGB")

        # Benzer görüntünün gerçek sınıfını alıyoruz.
        label = int(test_labels[dataset_index])

        # Query ile similarity değerini alıyoruz.
        similarity = similarities[dataset_index].item()

        # Benzer görüntüyü gösteriyoruz.
        axes[position].imshow(image)

        # Görüntü sınıfını ve similarity değerini gösteriyoruz.
        axes[position].set_title(
            f"{CLASS_NAMES[label]}\n"
            f"Similarity: {similarity:.3f}"
        )

        # Koordinat eksenlerini kapatıyoruz.
        axes[position].axis("off")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Benzer görüntü sonuçlarını kaydediyoruz.
    plt.savefig(f"{OUTPUT_DIR}/dinov3_similar_images.png", dpi=200, bbox_inches="tight")

    # Grafik penceresini kapatıyoruz.
    plt.close()


    # ========================================================
    # 7. PROGRAM SONU
    # ========================================================

    # Analizin tamamlandığını gösteriyoruz.
    print("\n========== ANALİZ TAMAMLANDI ==========")

    # Confusion matrix dosyasını gösteriyoruz.
    print(f"Confusion Matrix: {OUTPUT_DIR}/dinov3_confusion_matrix.png")

    # PCA görselini gösteriyoruz.
    print(f"PCA: {OUTPUT_DIR}/dinov3_pca.png")

    # Similar image search görselini gösteriyoruz.
    print(f"Similar Images: {OUTPUT_DIR}/dinov3_similar_images.png")


# ============================================================
# 8. WINDOWS İÇİN GÜVENLİ BAŞLANGIÇ
# ============================================================

# Program doğrudan çalıştırıldığında main fonksiyonunu başlatıyoruz.
if __name__ == "__main__":

    # Analiz akışını çalıştırıyoruz.
    main()