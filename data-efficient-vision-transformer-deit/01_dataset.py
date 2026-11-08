# Dosya ve klasör işlemleri için Path kullanıyoruz.
from pathlib import Path

# PyTorch kullanıyoruz.
import torch

# Görüntüleri grid şeklinde oluşturmak için torchvision utils kullanıyoruz.
from torchvision.utils import make_grid

# Görüntüyü ekrana çizmek için matplotlib kullanıyoruz.
import matplotlib.pyplot as plt

# Hazırladığımız DataLoader fonksiyonunu içe aktarıyoruz.
from data_utils import get_loaders, IMAGENET_MEAN, IMAGENET_STD

# Çıktı klasörünü config dosyasından alıyoruz.
from config import OUTPUT_DIR


# Program doğrudan çalıştırıldığında aşağıdaki kodlar çalışır.
if __name__ == "__main__":

    # Train, validation ve test DataLoader'larını oluşturuyoruz.
    train_loader, val_loader, test_loader = get_loaders()

    # İlk training batch'ini alıyoruz.
    images, labels = next(iter(train_loader))

    # Train veri setindeki toplam örnek sayısını gösteriyoruz.
    print("Train örnek sayısı:", len(train_loader.dataset))

    # Validation veri setindeki toplam örnek sayısını gösteriyoruz.
    print("Validation örnek sayısı:", len(val_loader.dataset))

    # Test veri setindeki toplam örnek sayısını gösteriyoruz.
    print("Test örnek sayısı:", len(test_loader.dataset))

    # Batch içerisindeki görüntü Tensor boyutunu gösteriyoruz.
    print("Image batch shape:", images.shape)

    # Batch içerisindeki label Tensor boyutunu gösteriyoruz.
    print("Label batch shape:", labels.shape)

    # Batch'teki en küçük sınıf indeksini gösteriyoruz.
    print("Minimum label:", labels.min().item())

    # Batch'teki en büyük sınıf indeksini gösteriyoruz.
    print("Maximum label:", labels.max().item())


    # Normalization işlemini geri almak için mean değerlerini Tensor'a dönüştürüyoruz.
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)

    # Normalization işlemini geri almak için std değerlerini Tensor'a dönüştürüyoruz.
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    # İlk 8 görüntünün normalization işlemini geri alıyoruz.
    sample_images = images[:8] * std + mean

    # Piksel değerlerinin 0 ile 1 arasında kalmasını sağlıyoruz.
    sample_images = sample_images.clamp(0, 1)

    # İlk 8 görüntüyü tek bir grid haline getiriyoruz.
    grid = make_grid(sample_images, nrow=4)


    # Çıktı klasörünü oluşturuyoruz.
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Yeni bir matplotlib figure oluşturuyoruz.
    plt.figure(figsize=(10, 6))

    # Tensor kanal sırasını matplotlib'in beklediği formata çeviriyoruz.
    plt.imshow(grid.permute(1, 2, 0))

    # Görüntünün etrafındaki koordinat eksenlerini kaldırıyoruz.
    plt.axis("off")

    # Görsel başlığını belirliyoruz.
    plt.title("Oxford-IIIT Pet Örnek Görüntüler")

    # Görseli dosyaya kaydediyoruz.
    plt.savefig(f"{OUTPUT_DIR}/dataset_samples.png", bbox_inches="tight")

    # Görseli ekranda gösteriyoruz.
    plt.show()