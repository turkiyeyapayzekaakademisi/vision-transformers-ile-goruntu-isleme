# PyTorch kullanıyoruz.
import torch

# DeiT modellerini oluşturmak için timm kullanıyoruz.
import timm

# ResNet50 classifier katmanını değiştirmek için nn kullanıyoruz.
from torch import nn

# ResNet50 mimarisini torchvision'dan alıyoruz.
from torchvision import models

# Test DataLoader'ını oluşturmak için veri yardımcı fonksiyonumuzu kullanıyoruz.
from data_utils import get_loaders

# Modelleri test etmek için ortak evaluate fonksiyonumuzu kullanıyoruz.
from train_utils import evaluate

# Proje ayarlarını config dosyasından alıyoruz.
from config import NUM_CLASSES, MODEL_DIR


# Program doğrudan çalıştırıldığında karşılaştırma yapılır.
if __name__ == "__main__":

    # CUDA varsa GPU, yoksa CPU kullanıyoruz.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Yalnızca test DataLoader'ına ihtiyacımız olduğu için ilk ikisini kullanmıyoruz.
    _, _, test_loader = get_loaders()


    # ========================================================
    # RESNET50 TEACHER
    # ========================================================

    # ResNet50 mimarisini oluşturuyoruz.
    teacher = models.resnet50(weights=None)

    # Son classification katmanının giriş feature sayısını alıyoruz.
    teacher_features = teacher.fc.in_features

    # Son classifier'ı 37 sınıfa göre değiştiriyoruz.
    teacher.fc = nn.Linear(teacher_features, NUM_CLASSES)

    # Eğittiğimiz teacher ağırlıklarını yüklüyoruz.
    teacher.load_state_dict(
        torch.load(
            f"{MODEL_DIR}/resnet50_teacher.pt",
            map_location=device,
            weights_only=True,
        )
    )

    # Teacher modelini kullanılacak cihaza taşıyoruz.
    teacher = teacher.to(device)


    # ========================================================
    # NORMAL DEIT
    # ========================================================

    # Normal DeiT-Tiny mimarisini 37 sınıflı olarak oluşturuyoruz.
    deit = timm.create_model(
        "deit_tiny_patch16_224.fb_in1k",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )

    # Daha önce eğittiğimiz normal DeiT ağırlıklarını yüklüyoruz.
    deit.load_state_dict(
        torch.load(
            f"{MODEL_DIR}/deit_tiny.pt",
            map_location=device,
            weights_only=True,
        )
    )

    # DeiT modelini kullanılacak cihaza taşıyoruz.
    deit = deit.to(device)


    # ========================================================
    # DISTILLED DEIT
    # ========================================================

    # Distillation token içeren DeiT-Tiny mimarisini oluşturuyoruz.
    distilled_deit = timm.create_model(
        "deit_tiny_distilled_patch16_224.fb_in1k",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )

    # Knowledge Distillation ile eğittiğimiz ağırlıkları yüklüyoruz.
    distilled_deit.load_state_dict(
        torch.load(
            f"{MODEL_DIR}/deit_tiny_distilled.pt",
            map_location=device,
            weights_only=True,
        )
    )

    # Distilled DeiT modelini kullanılacak cihaza taşıyoruz.
    distilled_deit = distilled_deit.to(device)


    # ========================================================
    # TEST
    # ========================================================

    # Teacher modelini test veri setinde değerlendiriyoruz.
    teacher_loss, teacher_accuracy = evaluate(
        teacher,
        test_loader,
        device,
    )

    # Normal DeiT modelini aynı test veri setinde değerlendiriyoruz.
    deit_loss, deit_accuracy = evaluate(
        deit,
        test_loader,
        device,
    )

    # Distilled DeiT modelini aynı test veri setinde değerlendiriyoruz.
    distilled_loss, distilled_accuracy = evaluate(
        distilled_deit,
        test_loader,
        device,
    )


    # ========================================================
    # SONUÇLAR
    # ========================================================

    # Sonuç başlığını ekrana yazdırıyoruz.
    print("\n================ MODEL KARŞILAŞTIRMA ================")

    # ResNet50 teacher modelinin test accuracy değerini gösteriyoruz.
    print(f"ResNet50 Teacher    : {teacher_accuracy * 100:.2f}%")

    # Normal DeiT-Tiny modelinin test accuracy değerini gösteriyoruz.
    print(f"DeiT-Tiny           : {deit_accuracy * 100:.2f}%")

    # Knowledge Distillation ile eğitilen DeiT modelinin accuracy değerini gösteriyoruz.
    print(f"Distilled DeiT-Tiny : {distilled_accuracy * 100:.2f}%")

    # Normal DeiT ile Distilled DeiT arasındaki accuracy farkını hesaplıyoruz.
    improvement = (distilled_accuracy - deit_accuracy) * 100

    # Knowledge Distillation'ın normal DeiT'e göre oluşturduğu farkı gösteriyoruz.
    print(f"Distillation farkı  : {improvement:+.2f} puan")

    # Sonuç bölümünü kapatıyoruz.
    print("=====================================================\n")