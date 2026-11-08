# PyTorch işlemleri için torch kullanıyoruz.
import torch

# Loss fonksiyonları için nn modülünü kullanıyoruz.
from torch import nn

# Model dosyalarını kaydetmek için Path kullanıyoruz.
from pathlib import Path

# Eğitim sırasında progress bar göstermek için tqdm kullanıyoruz.
from tqdm import tqdm


# Gradient hesaplamadan validation veya test yapmak için inference_mode kullanıyoruz.
@torch.inference_mode()
def evaluate(model, loader, device):

    # Modeli evaluation moduna geçiriyoruz.
    model.eval()

    # Multi-class classification için CrossEntropyLoss kullanıyoruz.
    criterion = nn.CrossEntropyLoss()

    # Toplam loss değerini tutuyoruz.
    total_loss = 0.0

    # Doğru tahmin edilen örnek sayısını tutuyoruz.
    total_correct = 0

    # Toplam örnek sayısını tutuyoruz.
    total_samples = 0


    # DataLoader içerisindeki tüm batch'leri dolaşıyoruz.
    for images, labels in loader:

        # Görüntüleri GPU varsa GPU'ya taşıyoruz.
        images = images.to(device)

        # Etiketleri GPU varsa GPU'ya taşıyoruz.
        labels = labels.to(device)

        # Görüntüleri modele gönderip tahminleri alıyoruz.
        outputs = model(images)

        # Tahmin ile gerçek label arasındaki loss değerini hesaplıyoruz.
        loss = criterion(outputs, labels)

        # Batch loss değerini örnek sayısıyla ağırlıklandırarak topluyoruz.
        total_loss += loss.item() * labels.size(0)

        # En yüksek skora sahip sınıfı model tahmini olarak seçiyoruz.
        predictions = outputs.argmax(dim=1)

        # Doğru tahminlerin sayısını hesaplıyoruz.
        total_correct += (predictions == labels).sum().item()

        # İşlenen toplam örnek sayısını güncelliyoruz.
        total_samples += labels.size(0)


    # Tüm veri setinin ortalama loss değerini hesaplıyoruz.
    average_loss = total_loss / total_samples

    # Accuracy değerini hesaplıyoruz.
    accuracy = total_correct / total_samples

    # Loss ve accuracy değerlerini geri döndürüyoruz.
    return average_loss, accuracy


def train_supervised(model, train_loader, val_loader, device, epochs, lr, weight_decay, save_path):

    # Gerçek class label'larıyla eğitim için CrossEntropyLoss kullanıyoruz.
    criterion = nn.CrossEntropyLoss()

    # AdamW optimizer ile model ağırlıklarını güncelliyoruz.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Şimdiye kadar görülen en iyi validation accuracy değerini tutuyoruz.
    best_accuracy = 0.0


    # Modeli belirlenen epoch sayısı kadar eğitiyoruz.
    for epoch in range(epochs):

        # Modeli training moduna geçiriyoruz.
        model.train()

        # Epoch boyunca toplam loss değerini tutuyoruz.
        running_loss = 0.0

        # Epoch boyunca doğru tahmin edilen örnek sayısını tutuyoruz.
        running_correct = 0

        # Epoch boyunca işlenen toplam örnek sayısını tutuyoruz.
        running_samples = 0


        # Training DataLoader içerisindeki batch'leri işliyoruz.
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):

            # Görüntüleri kullanılacak cihaza taşıyoruz.
            images = images.to(device)

            # Etiketleri kullanılacak cihaza taşıyoruz.
            labels = labels.to(device)

            # Önceki batch'ten kalan gradient değerlerini temizliyoruz.
            optimizer.zero_grad()

            # Görüntüleri modelden geçiriyoruz.
            outputs = model(images)

            # Model tahminleri ile gerçek label arasındaki loss'u hesaplıyoruz.
            loss = criterion(outputs, labels)

            # Loss üzerinden gradient hesaplıyoruz.
            loss.backward()

            # Gradient değerlerini kullanarak model ağırlıklarını güncelliyoruz.
            optimizer.step()


            # Batch loss değerini örnek sayısıyla ağırlıklandırarak topluyoruz.
            running_loss += loss.item() * labels.size(0)

            # En yüksek skora sahip sınıfı model tahmini olarak alıyoruz.
            predictions = outputs.argmax(dim=1)

            # Doğru tahmin edilen örneklerin sayısını topluyoruz.
            running_correct += (predictions == labels).sum().item()

            # İşlenen toplam örnek sayısını güncelliyoruz.
            running_samples += labels.size(0)


        # Epoch sonundaki ortalama training loss değerini hesaplıyoruz.
        train_loss = running_loss / running_samples

        # Epoch sonundaki training accuracy değerini hesaplıyoruz.
        train_accuracy = running_correct / running_samples


        # Modeli validation veri setinde değerlendiriyoruz.
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device,
        )


        # Epoch sonuçlarını terminale yazdırıyoruz.
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )


        # Validation accuracy önceki en iyi sonuçtan yüksekse modeli kaydediyoruz.
        if val_accuracy > best_accuracy:

            # En iyi accuracy değerini güncelliyoruz.
            best_accuracy = val_accuracy

            # Model klasörü yoksa oluşturuyoruz.
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Model ağırlıklarını kaydediyoruz.
            torch.save(model.state_dict(), save_path)

            # Yeni en iyi modelin kaydedildiğini gösteriyoruz.
            print("Yeni en iyi model kaydedildi.")


    # Eğitim sonunda elde edilen en iyi validation accuracy değerini gösteriyoruz.
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")