# PyTorch kullanıyoruz.
import torch

# Knowledge Distillation loss işlemlerinde softmax kullanmak için functional modülünü alıyoruz.
import torch.nn.functional as F

# Distilled DeiT modelini oluşturmak için timm kullanıyoruz.
import timm

# Classification ve KL divergence loss fonksiyonları için nn kullanıyoruz.
from torch import nn

# Teacher olarak pretrained ResNet50 mimarisini kullanacağız.
from torchvision import models

# Training progress bar göstermek için tqdm kullanıyoruz.
from tqdm import tqdm

# Model klasörünü oluşturmak için Path kullanıyoruz.
from pathlib import Path

# Veri seti DataLoader'larımızı içe aktarıyoruz.
from data_utils import get_loaders

# Validation sırasında ortak evaluate fonksiyonumuzu kullanıyoruz.
from train_utils import evaluate

# Tüm proje parametrelerini config dosyasından alıyoruz.
from config import (
    NUM_CLASSES,
    EPOCHS,
    DISTILL_LR,
    WEIGHT_DECAY,
    TEMPERATURE,
    ALPHA,
    MODEL_DIR,
)


# Program doğrudan çalıştırıldığında Knowledge Distillation eğitimi başlar.
if __name__ == "__main__":

    # CUDA varsa GPU, yoksa CPU kullanıyoruz.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training ve validation DataLoader'larını oluşturuyoruz.
    train_loader, val_loader, _ = get_loaders()


    # ========================================================
    # TEACHER MODEL
    # ========================================================

    # ResNet50 mimarisini oluşturuyoruz fakat ImageNet ağırlıklarını tekrar indirmiyoruz.
    teacher = models.resnet50(weights=None)

    # ResNet50 classifier giriş feature sayısını alıyoruz.
    teacher_features = teacher.fc.in_features

    # Teacher classifier'ını Oxford-IIIT Pet'in 37 sınıfına göre oluşturuyoruz.
    teacher.fc = nn.Linear(teacher_features, NUM_CLASSES)

    # Daha önce eğittiğimiz teacher model ağırlıklarını dosyadan yüklüyoruz.
    teacher.load_state_dict(
        torch.load(
            f"{MODEL_DIR}/resnet50_teacher.pt",
            map_location=device,
            weights_only=True,
        )
    )

    # Teacher modelini kullanılacak cihaza taşıyoruz.
    teacher = teacher.to(device)

    # Teacher modelini inference moduna geçiriyoruz çünkü artık teacher'ı eğitmeyeceğiz.
    teacher.eval()


    # Teacher modelindeki tüm parametrelerin gradient hesaplamasını kapatıyoruz.
    for parameter in teacher.parameters():

        # Teacher sabit kalacak ve yalnızca student'a bilgi sağlayacaktır.
        parameter.requires_grad = False


    # ========================================================
    # STUDENT MODEL
    # ========================================================

    # Distillation token içeren ImageNet pretrained DeiT-Tiny Distilled modelini yüklüyoruz.
    student = timm.create_model(
        "deit_tiny_distilled_patch16_224.fb_in1k",
        pretrained=True,
        num_classes=NUM_CLASSES,
    )

    # Eğitim sırasında CLS head ve DIST head sonuçlarını ayrı almak için distilled training modunu açıyoruz.
    student.set_distilled_training(True)

    # Student modelini kullanılacak cihaza taşıyoruz.
    student = student.to(device)


    # ========================================================
    # OPTIMIZER VE LOSS FONKSİYONLARI
    # ========================================================

    # Student model ağırlıklarını AdamW optimizer ile güncelleyeceğiz.
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=DISTILL_LR,
        weight_decay=WEIGHT_DECAY,
    )

    # CLS token çıktısını gerçek label ile karşılaştırmak için CrossEntropyLoss kullanıyoruz.
    classification_loss_function = nn.CrossEntropyLoss()

    # DIST token çıktısını teacher probability dağılımıyla karşılaştırmak için KL Divergence kullanıyoruz.
    distillation_loss_function = nn.KLDivLoss(reduction="batchmean")

    # En yüksek validation accuracy değerini tutuyoruz.
    best_accuracy = 0.0


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    # Student modelini belirlenen epoch sayısı kadar eğitiyoruz.
    for epoch in range(EPOCHS):

        # Student modelini training moduna geçiriyoruz.
        student.train()

        # Epoch toplam loss değerini tutuyoruz.
        running_loss = 0.0


        # Training veri setindeki bütün batch'leri işliyoruz.
        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        ):

            # Görüntüleri GPU varsa GPU'ya taşıyoruz.
            images = images.to(device)

            # Gerçek label'ları GPU varsa GPU'ya taşıyoruz.
            labels = labels.to(device)


            # Teacher model için gradient hesaplamaya ihtiyacımız olmadığı için no_grad kullanıyoruz.
            with torch.no_grad():

                # Aynı görüntüleri teacher ResNet50 modeline gönderiyoruz.
                teacher_logits = teacher(images)


            # Önceki batch'ten kalan student gradient değerlerini temizliyoruz.
            optimizer.zero_grad()


            # Distilled DeiT eğitim modunda iki ayrı çıktı üretir.
            class_logits, distill_logits = student(images)


            # ====================================================
            # 1. GERÇEK LABEL LOSS
            # ====================================================

            # CLS token çıktısını gerçek sınıf etiketleriyle karşılaştırıyoruz.
            classification_loss = classification_loss_function(
                class_logits,
                labels,
            )


            # ====================================================
            # 2. TEACHER DISTILLATION LOSS
            # ====================================================

            # Student DIST çıktısını temperature ile yumuşatıp log probability haline getiriyoruz.
            student_soft_predictions = F.log_softmax(
                distill_logits / TEMPERATURE,
                dim=1,
            )

            # Teacher çıktısını temperature ile yumuşatılmış probability dağılımına dönüştürüyoruz.
            teacher_soft_predictions = F.softmax(
                teacher_logits / TEMPERATURE,
                dim=1,
            )

            # Student ile teacher probability dağılımları arasındaki farkı hesaplıyoruz.
            distillation_loss = distillation_loss_function(
                student_soft_predictions,
                teacher_soft_predictions,
            )

            # Temperature ölçeklemesinin gradient büyüklüğüne etkisini dengelemek için T² ile çarpıyoruz.
            distillation_loss = distillation_loss * (TEMPERATURE ** 2)


            # ====================================================
            # TOPLAM LOSS
            # ====================================================

            # Gerçek label loss ve teacher loss'u ALPHA değeriyle birleştiriyoruz.
            loss = (
                ALPHA * classification_loss
                + (1.0 - ALPHA) * distillation_loss
            )


            # Toplam loss üzerinden student model gradientlerini hesaplıyoruz.
            loss.backward()

            # Student model ağırlıklarını güncelliyoruz.
            optimizer.step()

            # Batch loss değerini epoch toplamına ekliyoruz.
            running_loss += loss.item()


        # ========================================================
        # VALIDATION
        # ========================================================

        # Student eval moduna geçtiğinde iki DeiT head'inin ortalaması final tahmin olarak kullanılır.
        val_loss, val_accuracy = evaluate(
            student,
            val_loader,
            device,
        )

        # Epoch ortalama training loss değerini hesaplıyoruz.
        train_loss = running_loss / len(train_loader)


        # Epoch sonuçlarını ekrana yazdırıyoruz.
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )


        # Validation accuracy şimdiye kadarki en yüksek değerden iyiyse modeli kaydediyoruz.
        if val_accuracy > best_accuracy:

            # En iyi accuracy değerini güncelliyoruz.
            best_accuracy = val_accuracy

            # Models klasörü yoksa oluşturuyoruz.
            Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

            # Distilled DeiT modelinin ağırlıklarını kaydediyoruz.
            torch.save(
                student.state_dict(),
                f"{MODEL_DIR}/deit_tiny_distilled.pt",
            )

            # Kullanıcıya yeni en iyi modelin kaydedildiğini bildiriyoruz.
            print("Yeni en iyi Distilled DeiT modeli kaydedildi.")


    # Eğitim sonunda en iyi validation accuracy değerini gösteriyoruz.
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")