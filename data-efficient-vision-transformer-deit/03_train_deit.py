# PyTorch kullanıyoruz.
import torch

# DeiT modelini timm kütüphanesinden oluşturacağız.
import timm

# Veri setimizi yüklemek için hazırladığımız fonksiyonu kullanıyoruz.
from data_utils import get_loaders

# Ortak supervised training fonksiyonumuzu kullanıyoruz.
from train_utils import train_supervised

# Proje ayarlarını config dosyasından alıyoruz.
from config import NUM_CLASSES, EPOCHS, DEIT_LR, WEIGHT_DECAY, MODEL_DIR


# Program doğrudan çalıştırıldığında aşağıdaki kod çalışır.
if __name__ == "__main__":

    # CUDA varsa GPU, yoksa CPU seçiyoruz.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training ve validation DataLoader'larını oluşturuyoruz.
    train_loader, val_loader, _ = get_loaders()


    # ImageNet pretrained DeiT-Tiny modelini yüklüyoruz.
    model = timm.create_model(
        "deit_tiny_patch16_224.fb_in1k",
        pretrained=True,
        num_classes=NUM_CLASSES,
    )

    # Modeli GPU varsa GPU'ya taşıyoruz.
    model = model.to(device)


    # Kullanılan cihazı gösteriyoruz.
    print("Device:", device)

    # Eğitilen modeli gösteriyoruz.
    print("Model: Normal DeiT-Tiny")

    # Bu model teacher kullanmadan yalnızca gerçek label'lardan öğrenecektir.
    print("Training: Sadece gerçek label")


    # DeiT-Tiny modelini supervised şekilde fine-tune ediyoruz.
    train_supervised(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        lr=DEIT_LR,
        weight_decay=WEIGHT_DECAY,
        save_path=f"{MODEL_DIR}/deit_tiny.pt",
    )