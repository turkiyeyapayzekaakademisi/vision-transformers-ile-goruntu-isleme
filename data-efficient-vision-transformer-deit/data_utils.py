# PyTorch işlemleri için torch kütüphanesini içe aktarıyoruz.
import torch

# Hazır veri setleri ve görüntü dönüşümleri için torchvision kullanıyoruz.
from torchvision import datasets, transforms

# Veri setlerinin belirli indekslerini seçmek ve batch oluşturmak için kullanıyoruz.
from torch.utils.data import DataLoader, Subset

# Proje ayarlarını config.py dosyasından alıyoruz.
from config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, SEED, VAL_RATIO


# ImageNet üzerinde pretrained modeller bu ortalama değerlerle eğitildiği için aynı normalization değerlerini kullanıyoruz.
IMAGENET_MEAN = (0.485, 0.456, 0.406)

# ImageNet üzerinde kullanılan standart RGB kanal standart sapma değerleridir.
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_loaders():

    # Eğitim görüntülerine augmentation uygulayarak veri çeşitliliğini artırıyoruz.
    train_transform = transforms.Compose([

        # Görüntünün farklı bölgelerini rastgele kırpar ve 224x224 boyutuna getirir.
        transforms.RandomResizedCrop(IMAGE_SIZE),

        # Görüntüyü yüzde 50 ihtimalle yatay çevirerek augmentation yapar.
        transforms.RandomHorizontalFlip(),

        # Renk, kontrast ve benzeri rastgele dönüşümler uygulayarak veri çeşitliliğini artırır.
        transforms.RandAugment(num_ops=2, magnitude=9),

        # PIL görüntüsünü PyTorch Tensor formatına dönüştürür.
        transforms.ToTensor(),

        # Görüntüyü pretrained ImageNet modellerinin beklediği dağılıma getirir.
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


    # Validation ve test sırasında rastgele augmentation kullanmıyoruz.
    eval_transform = transforms.Compose([

        # Görüntünün kısa kenarını 256 piksele getirir.
        transforms.Resize(256),

        # Görüntünün merkezinden 224x224 alan alır.
        transforms.CenterCrop(IMAGE_SIZE),

        # PIL görüntüsünü Tensor formatına dönüştürür.
        transforms.ToTensor(),

        # ImageNet normalization uygular.
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


    # Oxford-IIIT Pet trainval bölümünü training augmentation ile yüklüyoruz.
    train_source = datasets.OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=True,
    )


    # Aynı trainval verisini validation dönüşümleriyle tekrar tanımlıyoruz.
    val_source = datasets.OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=eval_transform,
        download=True,
    )


    # Veri setinin resmi test bölümünü yalnızca final değerlendirme için kullanıyoruz.
    test_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR,
        split="test",
        target_types="category",
        transform=eval_transform,
        download=True,
    )


    # Her çalıştırmada aynı rastgele sıralamayı oluşturmak için sabit seed kullanıyoruz.
    generator = torch.Generator().manual_seed(SEED)

    # Trainval veri setindeki tüm indeksleri rastgele sıralıyoruz.
    indices = torch.randperm(len(train_source), generator=generator).tolist()

    # Trainval verisinin yüzde 20'sini validation için ayırıyoruz.
    val_size = int(len(indices) * VAL_RATIO)

    # İlk yüzde 20'lik indeksleri validation olarak kullanıyoruz.
    val_indices = indices[:val_size]

    # Geri kalan indeksleri training olarak kullanıyoruz.
    train_indices = indices[val_size:]


    # Training veri setini seçilen training indeksleriyle oluşturuyoruz.
    train_dataset = Subset(train_source, train_indices)

    # Validation veri setini aynı veri havuzunun validation indeksleriyle oluşturuyoruz.
    val_dataset = Subset(val_source, val_indices)


    # Training sırasında verileri karıştırarak batch'ler oluşturuyoruz.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


    # Validation sırasında sıralamanın önemi olmadığı için shuffle kullanmıyoruz.
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


    # Test setini yalnızca eğitim tamamlandıktan sonra kullanacağız.
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


    # Üç DataLoader'ı çağıran programa geri döndürüyoruz.
    return train_loader, val_loader, test_loader