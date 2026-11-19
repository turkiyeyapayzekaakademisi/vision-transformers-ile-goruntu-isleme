# ============================================================
# 1. KÜTÜPHANELER
# ============================================================

# PyTorch işlemleri için kullanılır.
import torch

# Linear classifier ve loss fonksiyonu için kullanılır.
from torch import nn

# Feature Tensor'larını batch'lere ayırmak için kullanılır.
from torch.utils.data import TensorDataset, DataLoader

# Feature vektörlerini normalize etmek için kullanılır.
import torch.nn.functional as F

# Stratified train-validation ayrımı için kullanılır.
from sklearn.model_selection import train_test_split

# Sayısal indeks işlemleri için kullanılır.
import numpy as np

# Proje ayarlarını config.py dosyasından alıyoruz.
from config import TRAIN_FEATURE_PATH, LINEAR_MODEL_PATH, NUM_CLASSES, VAL_RATIO, LINEAR_EPOCHS, LINEAR_LR, WEIGHT_DECAY, LINEAR_BATCH_SIZE, SEED


# ============================================================
# 2. DEĞERLENDİRME FONKSİYONU
# ============================================================

# Validation sırasında gradient hesaplamıyoruz.
@torch.inference_mode()
def evaluate(model, loader, criterion):

    # Modeli evaluation moduna geçiriyoruz.
    model.eval()

    # Toplam loss değerini saklıyoruz.
    total_loss = 0.0

    # Doğru tahmin edilen örnek sayısını saklıyoruz.
    total_correct = 0

    # Toplam örnek sayısını saklıyoruz.
    total_samples = 0

    # Validation batch'lerini dolaşıyoruz.
    for features, labels in loader:

        # Linear classifier tahminlerini hesaplıyoruz.
        outputs = model(features)

        # Classification loss değerini hesaplıyoruz.
        loss = criterion(outputs, labels)

        # Batch loss değerini topluyoruz.
        total_loss += loss.item() * labels.size(0)

        # En yüksek skorlu sınıfı tahmin olarak seçiyoruz.
        predictions = outputs.argmax(dim=1)

        # Doğru tahminleri topluyoruz.
        total_correct += (predictions == labels).sum().item()

        # Toplam örnek sayısını artırıyoruz.
        total_samples += labels.size(0)

    # Ortalama validation loss değerini hesaplıyoruz.
    average_loss = total_loss / total_samples

    # Validation accuracy değerini hesaplıyoruz.
    accuracy = total_correct / total_samples

    # Loss ve accuracy değerlerini döndürüyoruz.
    return average_loss, accuracy


# ============================================================
# 3. ANA PROGRAM
# ============================================================

def main():

    # Random işlemlerin tekrar üretilebilir olmasını sağlıyoruz.
    torch.manual_seed(SEED)

    # DINOv3 tarafından oluşturulan training feature dosyasını yüklüyoruz.
    data = torch.load(TRAIN_FEATURE_PATH, map_location="cpu", weights_only=False)

    # Görüntü embedding'lerini alıyoruz.
    features = data["features"].float()

    # Görüntü sınıflarını alıyoruz.
    labels = data["labels"].long()

    # Feature vektörlerini L2 normalize ediyoruz.
    features = F.normalize(features, p=2, dim=1)

    # Embedding boyutunu otomatik olarak alıyoruz.
    embedding_dim = features.shape[1]

    # Veri setindeki tüm indeksleri oluşturuyoruz.
    indices = np.arange(len(labels))

    # Sınıf dağılımını koruyarak train-validation ayrımı yapıyoruz.
    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=labels.numpy(),
    )

    # Training feature Tensor'larını seçiyoruz.
    train_features = features[train_indices]

    # Training label Tensor'larını seçiyoruz.
    train_labels = labels[train_indices]

    # Validation feature Tensor'larını seçiyoruz.
    val_features = features[val_indices]

    # Validation label Tensor'larını seçiyoruz.
    val_labels = labels[val_indices]

    # Training TensorDataset oluşturuyoruz.
    train_dataset = TensorDataset(train_features, train_labels)

    # Validation TensorDataset oluşturuyoruz.
    val_dataset = TensorDataset(val_features, val_labels)

    # Training DataLoader oluşturuyoruz.
    train_loader = DataLoader(train_dataset, batch_size=LINEAR_BATCH_SIZE, shuffle=True)

    # Validation DataLoader oluşturuyoruz.
    val_loader = DataLoader(val_dataset, batch_size=LINEAR_BATCH_SIZE, shuffle=False)

    # Veri bilgilerini gösteriyoruz.
    print("========== FEATURES ==========")

    # Embedding boyutunu gösteriyoruz.
    print("Embedding Dimension:", embedding_dim)

    # Training örnek sayısını gösteriyoruz.
    print("Train:", len(train_dataset))

    # Validation örnek sayısını gösteriyoruz.
    print("Validation:", len(val_dataset))


    # ========================================================
    # 4. LINEAR PROBE
    # ========================================================

    # DINOv3 embedding'lerini 10 sınıfa dönüştüren tek linear layer oluşturuyoruz.
    classifier = nn.Linear(embedding_dim, NUM_CLASSES)

    # Multi-class classification için CrossEntropyLoss kullanıyoruz.
    criterion = nn.CrossEntropyLoss()

    # Yalnızca linear classifier parametrelerini optimize ediyoruz.
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LINEAR_LR, weight_decay=WEIGHT_DECAY)

    # En yüksek validation accuracy değerini saklıyoruz.
    best_val_accuracy = 0.0

    # Eğitim başlangıcını gösteriyoruz.
    print("\n========== LINEAR PROBE TRAINING ==========")

    # Linear classifier'ı belirlenen epoch sayısı kadar eğitiyoruz.
    for epoch in range(LINEAR_EPOCHS):

        # Classifier'ı training moduna geçiriyoruz.
        classifier.train()

        # Training loss değerini saklıyoruz.
        total_loss = 0.0

        # Doğru tahmin sayısını saklıyoruz.
        total_correct = 0

        # Toplam training örneğini saklıyoruz.
        total_samples = 0

        # Training batch'lerini dolaşıyoruz.
        for batch_features, batch_labels in train_loader:

            # Önceki gradient değerlerini temizliyoruz.
            optimizer.zero_grad()

            # Linear classifier tahminlerini hesaplıyoruz.
            outputs = classifier(batch_features)

            # Classification loss değerini hesaplıyoruz.
            loss = criterion(outputs, batch_labels)

            # Gradient değerlerini hesaplıyoruz.
            loss.backward()

            # Sadece linear classifier ağırlıklarını güncelliyoruz.
            optimizer.step()

            # Training loss değerlerini topluyoruz.
            total_loss += loss.item() * batch_labels.size(0)

            # Tahmin edilen sınıfları belirliyoruz.
            predictions = outputs.argmax(dim=1)

            # Doğru tahminleri topluyoruz.
            total_correct += (predictions == batch_labels).sum().item()

            # Toplam örnek sayısını artırıyoruz.
            total_samples += batch_labels.size(0)

        # Ortalama training loss hesaplıyoruz.
        train_loss = total_loss / total_samples

        # Training accuracy hesaplıyoruz.
        train_accuracy = total_correct / total_samples

        # Validation sonuçlarını hesaplıyoruz.
        val_loss, val_accuracy = evaluate(classifier, val_loader, criterion)

        # Epoch sonuçlarını terminalde gösteriyoruz.
        print(
            f"Epoch {epoch + 1}/{LINEAR_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy * 100:.2f}%"
        )

        # Validation accuracy şimdiye kadarki en iyi değerse modeli kaydediyoruz.
        if val_accuracy > best_val_accuracy:

            # En iyi validation accuracy değerini güncelliyoruz.
            best_val_accuracy = val_accuracy

            # Linear classifier bilgilerini kaydediyoruz.
            torch.save(
                {
                    "state_dict": classifier.state_dict(),
                    "embedding_dim": embedding_dim,
                    "num_classes": NUM_CLASSES,
                },
                LINEAR_MODEL_PATH,
            )

            # Yeni en iyi model bilgisini gösteriyoruz.
            print("Yeni en iyi Linear Probe kaydedildi.")

    # Final validation sonucunu gösteriyoruz.
    print(f"\nBest Validation Accuracy: {best_val_accuracy * 100:.2f}%")

    # Kaydedilen classifier yolunu gösteriyoruz.
    print("Linear Probe:", LINEAR_MODEL_PATH)


# ============================================================
# 5. WINDOWS İÇİN GÜVENLİ BAŞLANGIÇ
# ============================================================

# Program doğrudan çalıştırıldığında main fonksiyonunu başlatıyoruz.
if __name__ == "__main__":

    # Linear probe eğitimini başlatıyoruz.
    main()