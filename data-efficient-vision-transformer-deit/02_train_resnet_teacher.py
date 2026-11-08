# PyTorch kullanıyoruz.
import torch

# ResNet50'nin son classification katmanını değiştirmek için nn kullanıyoruz.
from torch import nn

# Hazır ResNet50 modelini torchvision'dan alıyoruz.
from torchvision import models

# Veri setimizi yüklemek için hazırladığımız fonksiyonu kullanıyoruz.
from data_utils import get_loaders

# Ortak supervised eğitim fonksiyonumuzu kullanıyoruz.
from train_utils import train_supervised

# Proje parametrelerini config dosyasından alıyoruz.
from config import NUM_CLASSES, EPOCHS, TEACHER_LR, WEIGHT_DECAY, MODEL_DIR


# Program doğrudan çalıştırıldığında eğitim başlar.
if __name__ == "__main__":

    # CUDA varsa GPU, yoksa CPU kullanıyoruz.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training ve validation DataLoader'larını oluşturuyoruz.
    train_loader, val_loader, _ = get_loaders()


    # ImageNet üzerinde pretrained ResNet50 modelini yüklüyoruz.
    teacher = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # ResNet50'nin orijinal 1000 sınıflı ImageNet classifier giriş boyutunu alıyoruz.
    input_features = teacher.fc.in_features

    # Son classification katmanını Oxford-IIIT Pet'in 37 sınıfına göre değiştiriyoruz.
    teacher.fc = nn.Linear(input_features, NUM_CLASSES)

    # Modeli GPU varsa GPU'ya taşıyoruz.
    teacher = teacher.to(device)


    # Kullanılan cihazı ekranda gösteriyoruz.
    print("Device:", device)

    # Eğitilecek modelin rolünü gösteriyoruz.
    print("Model: ResNet50 Teacher")


    # ResNet50 teacher modelini Oxford-IIIT Pet veri setinde fine-tune ediyoruz.
    train_supervised(
        model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        lr=TEACHER_LR,
        weight_decay=WEIGHT_DECAY,
        save_path=f"{MODEL_DIR}/resnet50_teacher.pt",
    )