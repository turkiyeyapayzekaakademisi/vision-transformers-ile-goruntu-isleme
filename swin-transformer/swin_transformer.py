# ============================================================
# 1. KÜTÜPHANELER
# ============================================================

# PyTorch işlemleri için kullanılır.
import torch

# Classification loss fonksiyonu için kullanılır.
from torch import nn

# Feature map boyutlarını büyütmek için kullanılır.
import torch.nn.functional as F

# Oxford-IIIT Pet veri seti ve görüntü dönüşümleri için kullanılır.
from torchvision import datasets, transforms

# DataLoader ve veri setinin belirli bölümlerini seçmek için kullanılır.
from torch.utils.data import DataLoader, Subset

# Pretrained Swin Transformer modelini kullanmak için timm kullanılır.
import timm

# Eğitim sırasında progress bar göstermek için kullanılır.
from tqdm import tqdm

# Grafik ve görüntü çizmek için kullanılır.
import matplotlib.pyplot as plt

# Classification report ve confusion matrix hesaplamak için kullanılır.
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Klasör oluşturmak için kullanılır.
from pathlib import Path

# Proje ayarlarını config.py dosyasından alıyoruz.
from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, MODEL_NAME, NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, SEED, VAL_RATIO, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, TRAIN_MODEL


# ============================================================
# 2. CPU AYARI
# ============================================================

# Bu projede yalnızca CPU kullanıyoruz.
DEVICE = torch.device("cpu")

# Kullanılan cihazı terminalde gösteriyoruz.
print("Device: CPU")


# ============================================================
# 3. KLASÖRLER
# ============================================================

# Model klasörü yoksa oluşturuyoruz.
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# Output klasörü yoksa oluşturuyoruz.
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. IMAGENET NORMALIZATION
# ============================================================

# ImageNet pretrained modellerde kullanılan RGB kanal ortalamalarıdır.
IMAGENET_MEAN = (0.485, 0.456, 0.406)

# ImageNet pretrained modellerde kullanılan RGB kanal standart sapmalarıdır.
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# 5. TRAINING TRANSFORM
# ============================================================

# Training görüntülerine data augmentation uyguluyoruz.
train_transform = transforms.Compose([

    # Görüntünün farklı bölgelerini rastgele seçerek 224x224 boyutuna getirir.
    transforms.RandomResizedCrop(IMAGE_SIZE),

    # Görüntüyü yüzde 50 ihtimalle yatay çevirerek veri çeşitliliğini artırır.
    transforms.RandomHorizontalFlip(),

    # Görüntüye otomatik olarak farklı augmentation işlemleri uygular.
    transforms.RandAugment(num_ops=2, magnitude=9),

    # PIL görüntüsünü PyTorch Tensor formatına dönüştürür.
    transforms.ToTensor(),

    # Görüntüyü pretrained ImageNet modelinin beklediği dağılıma getirir.
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# 6. VALIDATION VE TEST TRANSFORM
# ============================================================

# Validation ve test sırasında rastgele augmentation uygulamıyoruz.
eval_transform = transforms.Compose([

    # Görüntünün kısa kenarını 256 piksele getirir.
    transforms.Resize(256),

    # Görüntünün merkezinden 224x224 alan alır.
    transforms.CenterCrop(IMAGE_SIZE),

    # Görüntüyü Tensor formatına dönüştürür.
    transforms.ToTensor(),

    # ImageNet normalization uygular.
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# 7. VERİ SETİNİ HAZIRLAYAN FONKSİYON
# ============================================================

def prepare_data():

    # Oxford-IIIT Pet trainval veri setini training augmentation ile yüklüyoruz.
    train_source = datasets.OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=True,
    )

    # Aynı trainval veri setini validation dönüşümleri ile tekrar tanımlıyoruz.
    val_source = datasets.OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=eval_transform,
        download=True,
    )

    # Veri setinin resmi test bölümünü yüklüyoruz.
    test_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR,
        split="test",
        target_types="category",
        transform=eval_transform,
        download=True,
    )

    # Her çalıştırmada aynı train-validation ayrımının oluşmasını sağlar.
    generator = torch.Generator().manual_seed(SEED)

    # Trainval içerisindeki tüm indeksleri rastgele sıralıyoruz.
    indices = torch.randperm(len(train_source), generator=generator).tolist()

    # Validation için ayrılacak görüntü sayısını hesaplıyoruz.
    val_size = int(len(indices) * VAL_RATIO)

    # İlk yüzde 20'lik kısmı validation için ayırıyoruz.
    val_indices = indices[:val_size]

    # Geri kalan görüntüleri training için kullanıyoruz.
    train_indices = indices[val_size:]

    # Training veri setini oluşturuyoruz.
    train_dataset = Subset(train_source, train_indices)

    # Validation veri setini oluşturuyoruz.
    val_dataset = Subset(val_source, val_indices)

    # CPU ve Windows için num_workers=0 kullanarak training DataLoader oluşturuyoruz.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # Validation DataLoader oluşturuyoruz.
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Test DataLoader oluşturuyoruz.
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Veri setindeki 37 sınıfın isimlerini alıyoruz.
    class_names = test_dataset.classes

    # Veri seti bilgilerini terminalde gösteriyoruz.
    print("\n========== DATASET ==========")

    # Training örnek sayısını gösteriyoruz.
    print("Train:", len(train_dataset))

    # Validation örnek sayısını gösteriyoruz.
    print("Validation:", len(val_dataset))

    # Test örnek sayısını gösteriyoruz.
    print("Test:", len(test_dataset))

    # Sınıf sayısını gösteriyoruz.
    print("Sınıf Sayısı:", len(class_names))

    # Oluşturulan DataLoader'ları geri döndürüyoruz.
    return train_loader, val_loader, test_loader, class_names


# ============================================================
# 8. SWIN-TINY MODELİNİ HAZIRLAYAN FONKSİYON
# ============================================================

def prepare_model(pretrained=True):

    # ImageNet pretrained Swin-Tiny modelini oluşturuyoruz.
    # num_classes=37 ile orijinal 1000 sınıflı classifier'ı değiştiriyoruz.
    model = timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
    )

    # Modeli CPU üzerinde çalıştırıyoruz.
    model = model.to(DEVICE)

    # Modelin toplam parametre sayısını hesaplıyoruz.
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    # Eğitim sırasında güncellenecek parametre sayısını hesaplıyoruz.
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    # Model bilgilerini terminalde gösteriyoruz.
    print("\n========== MODEL ==========")

    # Kullanılan modeli gösteriyoruz.
    print("Model:", MODEL_NAME)

    # Toplam parametre sayısını gösteriyoruz.
    print(f"Toplam Parametre: {total_parameters:,}")

    # Eğitilebilir parametre sayısını gösteriyoruz.
    print(f"Eğitilebilir Parametre: {trainable_parameters:,}")

    # Modelin çıktı sınıf sayısını gösteriyoruz.
    print("Çıkış Sınıf Sayısı:", NUM_CLASSES)

    # Hazırlanan modeli geri döndürüyoruz.
    return model


# ============================================================
# 9. MODEL DEĞERLENDİRME FONKSİYONU
# ============================================================

# Validation ve test sırasında gradient hesaplamıyoruz.
@torch.inference_mode()
def evaluate(model, loader, criterion):

    # Modeli evaluation moduna geçiriyoruz.
    model.eval()

    # Toplam loss değerini tutuyoruz.
    total_loss = 0.0

    # Doğru tahmin sayısını tutuyoruz.
    total_correct = 0

    # Toplam örnek sayısını tutuyoruz.
    total_samples = 0

    # DataLoader içerisindeki tüm batch'leri dolaşıyoruz.
    for images, labels in loader:

        # Görüntüleri CPU üzerinde kullanıyoruz.
        images = images.to(DEVICE)

        # Label'ları CPU üzerinde kullanıyoruz.
        labels = labels.to(DEVICE)

        # Görüntüleri modele gönderiyoruz.
        outputs = model(images)

        # Loss değerini hesaplıyoruz.
        loss = criterion(outputs, labels)

        # Batch loss değerini görüntü sayısıyla ağırlıklandırıyoruz.
        total_loss += loss.item() * labels.size(0)

        # En yüksek çıktıya sahip sınıfı tahmin olarak seçiyoruz.
        predictions = outputs.argmax(dim=1)

        # Doğru tahmin sayısını hesaplıyoruz.
        total_correct += (predictions == labels).sum().item()

        # İşlenen toplam örnek sayısını artırıyoruz.
        total_samples += labels.size(0)

    # Ortalama loss değerini hesaplıyoruz.
    average_loss = total_loss / total_samples

    # Accuracy değerini hesaplıyoruz.
    accuracy = total_correct / total_samples

    # Loss ve accuracy değerlerini geri döndürüyoruz.
    return average_loss, accuracy


# ============================================================
# 10. MODEL TRAINING FONKSİYONU
# ============================================================

def train_model(model, train_loader, val_loader):

    # 37 sınıflı classification için CrossEntropyLoss kullanıyoruz.
    criterion = nn.CrossEntropyLoss()

    # Transformer modellerinin fine-tuning işleminde AdamW kullanıyoruz.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Learning rate'i epoch ilerledikçe azaltmak için cosine scheduler kullanıyoruz.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    # Şimdiye kadar elde edilen en yüksek validation accuracy değerini tutuyoruz.
    best_val_accuracy = 0.0

    # Eğitilmiş modelin kaydedileceği dosya.
    model_path = f"{MODEL_DIR}/swin_tiny_best.pt"

    # Training başlangıcını terminalde gösteriyoruz.
    print("\n========== TRAINING ==========")

    # Modeli belirlenen epoch sayısı kadar eğitiyoruz.
    for epoch in range(EPOCHS):

        # Modeli training moduna geçiriyoruz.
        model.train()

        # Epoch boyunca toplam training loss değerini tutuyoruz.
        train_loss_total = 0.0

        # Doğru training tahmin sayısını tutuyoruz.
        train_correct = 0

        # Toplam training örnek sayısını tutuyoruz.
        train_samples = 0

        # Training batch'lerini progress bar ile dolaşıyoruz.
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):

            # Görüntüleri CPU üzerinde kullanıyoruz.
            images = images.to(DEVICE)

            # Label'ları CPU üzerinde kullanıyoruz.
            labels = labels.to(DEVICE)

            # Önceki batch'ten kalan gradient değerlerini temizliyoruz.
            optimizer.zero_grad()

            # Model tahminlerini hesaplıyoruz.
            outputs = model(images)

            # Classification loss değerini hesaplıyoruz.
            loss = criterion(outputs, labels)

            # Gradient değerlerini hesaplıyoruz.
            loss.backward()

            # Model ağırlıklarını güncelliyoruz.
            optimizer.step()

            # Batch loss değerini topluyoruz.
            train_loss_total += loss.item() * labels.size(0)

            # En yüksek skorlu sınıfı model tahmini olarak seçiyoruz.
            predictions = outputs.argmax(dim=1)

            # Doğru tahmin edilen görüntülerin sayısını topluyoruz.
            train_correct += (predictions == labels).sum().item()

            # İşlenen toplam training örneğini artırıyoruz.
            train_samples += labels.size(0)

        # Epoch sonundaki ortalama training loss değerini hesaplıyoruz.
        train_loss = train_loss_total / train_samples

        # Epoch sonundaki training accuracy değerini hesaplıyoruz.
        train_accuracy = train_correct / train_samples

        # Modeli validation veri setinde değerlendiriyoruz.
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        # O an kullanılan learning rate değerini alıyoruz.
        current_lr = optimizer.param_groups[0]["lr"]

        # Epoch sonuçlarını terminale yazdırıyoruz.
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy * 100:.2f}% | "
            f"LR: {current_lr:.8f}"
        )

        # Validation accuracy daha önceki en iyi değerden yüksekse modeli kaydediyoruz.
        if val_accuracy > best_val_accuracy:

            # En iyi validation accuracy değerini güncelliyoruz.
            best_val_accuracy = val_accuracy

            # Model ağırlıklarını kaydediyoruz.
            torch.save(model.state_dict(), model_path)

            # Yeni en iyi model bilgisini gösteriyoruz.
            print("Yeni en iyi model kaydedildi.")

        # Learning rate değerini bir sonraki epoch için güncelliyoruz.
        scheduler.step()

    # Eğitim sonunda en iyi validation accuracy değerini gösteriyoruz.
    print(f"\nBest Validation Accuracy: {best_val_accuracy * 100:.2f}%")

    # Eğitilen modeli geri döndürüyoruz.
    return model


# ============================================================
# 11. TEST PERFORMANSI
# ============================================================

def test_model(model, test_loader, class_names):

    # Classification loss fonksiyonunu tanımlıyoruz.
    criterion = nn.CrossEntropyLoss()

    # Test loss ve test accuracy değerlerini hesaplıyoruz.
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)

    # Test sonuçlarını terminalde gösteriyoruz.
    print("\n========== TEST ==========")

    # Test loss değerini gösteriyoruz.
    print(f"Test Loss: {test_loss:.4f}")

    # Test accuracy değerini yüzde olarak gösteriyoruz.
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Gerçek sınıfları saklamak için liste oluşturuyoruz.
    all_labels = []

    # Model tahminlerini saklamak için liste oluşturuyoruz.
    all_predictions = []

    # Modeli evaluation moduna geçiriyoruz.
    model.eval()

    # Gradient hesaplamadan test tahminlerini alıyoruz.
    with torch.no_grad():

        # Tüm test batch'lerini dolaşıyoruz.
        for images, labels in test_loader:

            # Model tahminlerini hesaplıyoruz.
            outputs = model(images.to(DEVICE))

            # En yüksek skorlu sınıfı prediction olarak seçiyoruz.
            predictions = outputs.argmax(dim=1).cpu()

            # Gerçek etiketleri listeye ekliyoruz.
            all_labels.extend(labels.tolist())

            # Model tahminlerini listeye ekliyoruz.
            all_predictions.extend(predictions.tolist())

    # Her sınıf için precision, recall ve F1-score değerlerini gösteriyoruz.
    print("\n========== CLASSIFICATION REPORT ==========")

    # Classification report'u terminale yazdırıyoruz.
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            digits=3,
        )
    )

    # Confusion matrix değerlerini hesaplıyoruz.
    matrix = confusion_matrix(all_labels, all_predictions)

    # 37 sınıf olduğu için büyük bir figure oluşturuyoruz.
    figure, axis = plt.subplots(figsize=(18, 18))

    # Confusion matrix görselini hazırlıyoruz.
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    # Hücre değerlerini göstermeden confusion matrix'i çiziyoruz.
    display.plot(
        ax=axis,
        xticks_rotation=90,
        include_values=False,
    )

    # Grafiğe test accuracy değerini ekliyoruz.
    plt.title(
        f"Swin-Tiny Confusion Matrix - Test Accuracy: {test_accuracy * 100:.2f}%"
    )

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Confusion matrix görselini kaydediyoruz.
    plt.savefig(
        f"{OUTPUT_DIR}/confusion_matrix.png",
        dpi=200,
    )

    # Confusion matrix penceresini kapatıyoruz.
    plt.close()

    # Test accuracy değerini geri döndürüyoruz.
    return test_accuracy


# ============================================================
# 12. ÖRNEK TAHMİNLER
# ============================================================

def show_predictions(model, test_loader, class_names):

    # Test veri setinden ilk batch'i alıyoruz.
    sample_images, sample_labels = next(iter(test_loader))

    # İlk 8 görüntüyü seçiyoruz.
    sample_input = sample_images[:8]

    # Gradient hesaplamadan tahmin yapıyoruz.
    with torch.no_grad():
        sample_outputs = model(sample_input)

    # En yüksek skorlu sınıfları prediction olarak seçiyoruz.
    sample_predictions = sample_outputs.argmax(dim=1)

    # Normalization işlemini geri almak için mean Tensor oluşturuyoruz.
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)

    # Normalization işlemini geri almak için std Tensor oluşturuyoruz.
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    # İlk sekiz görüntünün normalization işlemini geri alıyoruz.
    display_images = (sample_images[:8] * std + mean).clamp(0, 1)

    # Sekiz görüntüyü göstermek için 2x4 figure oluşturuyoruz.
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))

    # Sekiz görüntünün tamamını dolaşıyoruz.
    for index, axis in enumerate(axes.flatten()):

        # Görüntüyü gösteriyoruz.
        axis.imshow(display_images[index].permute(1, 2, 0))

        # Gerçek ve tahmin edilen sınıfı gösteriyoruz.
        axis.set_title(
            f"Gerçek: {class_names[sample_labels[index].item()]}\n"
            f"Tahmin: {class_names[sample_predictions[index].item()]}"
        )

        # Koordinat eksenlerini kapatıyoruz.
        axis.axis("off")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Tahmin görselini kaydediyoruz.
    plt.savefig(
        f"{OUTPUT_DIR}/sample_predictions.png",
        dpi=200,
    )

    # Grafik penceresini kapatıyoruz.
    plt.close()


# ============================================================
# 13. FEATURE MAP'İ HEATMAP'E DÖNÜŞTÜRME
# ============================================================

def create_heatmap(feature):

    # Tüm feature kanallarının mutlak değerlerinin ortalamasını alıyoruz.
    heatmap = feature[0].abs().mean(dim=0)

    # Minimum değeri sıfıra çekiyoruz.
    heatmap = heatmap - heatmap.min()

    # Heatmap değerlerini 0-1 aralığına normalize ediyoruz.
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # Feature map'i tekrar 224x224 görüntü boyutuna büyütüyoruz.
    heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    # Gereksiz batch ve channel boyutlarını kaldırıyoruz.
    return heatmap.squeeze()


# ============================================================
# 14. HIERARCHICAL FEATURE MAP GÖRSELLEŞTİRME
# ============================================================

def visualize_features(model, test_loader, class_names):

    # Test veri setinden bir batch alıyoruz.
    images, labels = next(iter(test_loader))

    # İlk görüntüyü seçiyoruz.
    image = images[0:1]

    # İlk görüntünün gerçek sınıf indeksini alıyoruz.
    true_label = labels[0].item()

    # Modeli evaluation moduna geçiriyoruz.
    model.eval()

    # Gradient hesaplamadan Swin'in dört stage çıktısını alıyoruz.
    with torch.no_grad():
        stage_features = model.forward_intermediates(
            image,
            indices=(0, 1, 2, 3),
            output_fmt="NCHW",
            intermediates_only=True,
        )

    # Stage bilgilerini terminalde gösteriyoruz.
    print("\n========== HIERARCHICAL FEATURES ==========")

    # Stage sayısını gösteriyoruz.
    print("Stage Sayısı:", len(stage_features))

    # Her stage'in Tensor boyutunu yazdırıyoruz.
    for index, feature in enumerate(stage_features):
        print(f"Stage {index + 1}: {tuple(feature.shape)}")

    # Stage 1 feature map'ini heatmap'e dönüştürüyoruz.
    stage1_heatmap = create_heatmap(stage_features[0])

    # Stage 2 feature map'ini heatmap'e dönüştürüyoruz.
    stage2_heatmap = create_heatmap(stage_features[1])

    # Stage 3 feature map'ini heatmap'e dönüştürüyoruz.
    stage3_heatmap = create_heatmap(stage_features[2])

    # Stage 4 feature map'ini heatmap'e dönüştürüyoruz.
    stage4_heatmap = create_heatmap(stage_features[3])

    # Normalization işlemini geri almak için mean oluşturuyoruz.
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)

    # Normalization işlemini geri almak için std oluşturuyoruz.
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    # Orijinal RGB görüntüyü tekrar oluşturuyoruz.
    original_image = (image * std + mean).clamp(0, 1)[0]

    # Orijinal görüntü ve dört stage için beş grafik oluşturuyoruz.
    figure, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Orijinal görüntüyü gösteriyoruz.
    axes[0].imshow(original_image.permute(1, 2, 0))

    # Görüntünün gerçek sınıfını başlıkta gösteriyoruz.
    axes[0].set_title(
        f"Original\n{class_names[true_label]}"
    )

    # Stage 1 heatmap'ini gösteriyoruz.
    axes[1].imshow(stage1_heatmap, cmap="viridis")

    # Stage 1 feature resolution değerini gösteriyoruz.
    axes[1].set_title(
        f"Stage 1\n"
        f"{stage_features[0].shape[-2]}x{stage_features[0].shape[-1]}"
    )

    # Stage 2 heatmap'ini gösteriyoruz.
    axes[2].imshow(stage2_heatmap, cmap="viridis")

    # Stage 2 feature resolution değerini gösteriyoruz.
    axes[2].set_title(
        f"Stage 2\n"
        f"{stage_features[1].shape[-2]}x{stage_features[1].shape[-1]}"
    )

    # Stage 3 heatmap'ini gösteriyoruz.
    axes[3].imshow(stage3_heatmap, cmap="viridis")

    # Stage 3 feature resolution değerini gösteriyoruz.
    axes[3].set_title(
        f"Stage 3\n"
        f"{stage_features[2].shape[-2]}x{stage_features[2].shape[-1]}"
    )

    # Stage 4 heatmap'ini gösteriyoruz.
    axes[4].imshow(stage4_heatmap, cmap="viridis")

    # Stage 4 feature resolution değerini gösteriyoruz.
    axes[4].set_title(
        f"Stage 4\n"
        f"{stage_features[3].shape[-2]}x{stage_features[3].shape[-1]}"
    )

    # Tüm grafiklerde koordinat eksenlerini kapatıyoruz.
    for axis in axes:
        axis.axis("off")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Hierarchical feature map görselini kaydediyoruz.
    plt.savefig(
        f"{OUTPUT_DIR}/swin_hierarchical_features.png",
        dpi=200,
        bbox_inches="tight",
    )

    # Grafik penceresini kapatıyoruz.
    plt.close()


    # ========================================================
    # FEATURE MAP OVERLAY
    # ========================================================

    # Orijinal görüntü ve dört overlay için figure oluşturuyoruz.
    figure, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Orijinal görüntüyü gösteriyoruz.
    axes[0].imshow(original_image.permute(1, 2, 0))

    # Orijinal görüntü başlığını ekliyoruz.
    axes[0].set_title("Original")

    # Stage 1 heatmap'ini orijinal görüntünün üzerine ekliyoruz.
    axes[1].imshow(original_image.permute(1, 2, 0))
    axes[1].imshow(stage1_heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title("Stage 1")

    # Stage 2 heatmap'ini orijinal görüntünün üzerine ekliyoruz.
    axes[2].imshow(original_image.permute(1, 2, 0))
    axes[2].imshow(stage2_heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title("Stage 2")

    # Stage 3 heatmap'ini orijinal görüntünün üzerine ekliyoruz.
    axes[3].imshow(original_image.permute(1, 2, 0))
    axes[3].imshow(stage3_heatmap, cmap="jet", alpha=0.45)
    axes[3].set_title("Stage 3")

    # Stage 4 heatmap'ini orijinal görüntünün üzerine ekliyoruz.
    axes[4].imshow(original_image.permute(1, 2, 0))
    axes[4].imshow(stage4_heatmap, cmap="jet", alpha=0.45)
    axes[4].set_title("Stage 4")

    # Tüm eksenleri kapatıyoruz.
    for axis in axes:
        axis.axis("off")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Overlay görselini kaydediyoruz.
    plt.savefig(
        f"{OUTPUT_DIR}/swin_feature_overlays.png",
        dpi=200,
        bbox_inches="tight",
    )

    # Grafik penceresini kapatıyoruz.
    plt.close()


# ============================================================
# 15. ANA PROGRAM
# ============================================================

def main():

    # Veri setini hazırlıyoruz.
    train_loader, val_loader, test_loader, class_names = prepare_data()

    # ImageNet pretrained Swin-Tiny modelini hazırlıyoruz.
    model = prepare_model(pretrained=True)

    # Eğitilmiş modelin kaydedileceği yolu belirliyoruz.
    model_path = f"{MODEL_DIR}/swin_tiny_best.pt"

    # TRAIN_MODEL True ise model fine-tune edilir.
    if TRAIN_MODEL:

        # Swin-Tiny modelini Oxford-IIIT Pet üzerinde fine-tune ediyoruz.
        train_model(
            model,
            train_loader,
            val_loader,
        )

    # Eğitim sonunda veya TRAIN_MODEL False olduğunda en iyi modeli yüklüyoruz.
    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
    )

    # Modeli CPU üzerinde evaluation moduna geçiriyoruz.
    model = model.to("cpu")

    # Modeli test veri setinde değerlendiriyoruz.
    test_accuracy = test_model(
        model,
        test_loader,
        class_names,
    )

    # Örnek model tahminlerini oluşturuyoruz.
    show_predictions(
        model,
        test_loader,
        class_names,
    )

    # Swin Transformer'ın Stage 1-4 feature map'lerini görselleştiriyoruz.
    visualize_features(
        model,
        test_loader,
        class_names,
    )

    # Projenin tamamlandığını gösteriyoruz.
    print("\n========== PROJE TAMAMLANDI ==========")

    # Final test accuracy değerini tekrar gösteriyoruz.
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    # Üretilen çıktıların bulunduğu klasörü gösteriyoruz.
    print("Çıktılar:", OUTPUT_DIR)


# ============================================================
# 16. WINDOWS İÇİN GÜVENLİ PROGRAM BAŞLANGICI
# ============================================================

# Windows multiprocessing ve import problemlerini engellemek için ana programı bu blok içinde başlatıyoruz.
if __name__ == "__main__":

    # Ana proje akışını çalıştırıyoruz.
    main()