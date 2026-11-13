# ============================================================
# 1. KÜTÜPHANELER
# ============================================================

# PyTorch işlemleri için kullanılır.
import torch

# PyTorch Dataset ve DataLoader yapıları için kullanılır.
from torch.utils.data import Dataset, DataLoader, random_split

# STL-10 veri setini Hugging Face üzerinden almak için kullanılır.
from datasets import load_dataset

# Pretrained MAE modelini ve image processor'ı yüklemek için kullanılır.
from transformers import AutoImageProcessor, ViTMAEForPreTraining

# Eğitim sırasında progress bar göstermek için kullanılır.
from tqdm import tqdm

# Training ve validation loss grafiğini oluşturmak için kullanılır.
import matplotlib.pyplot as plt

# Klasör oluşturmak için kullanılır.
from pathlib import Path

# Proje ayarlarını config.py dosyasından alıyoruz.
from config import MODEL_NAME, MODEL_SAVE_DIR, OUTPUT_DIR, DATASET_NAME, TRAIN_SAMPLES, VAL_RATIO, BATCH_SIZE, NUM_WORKERS, SEED, MASK_RATIO, EPOCHS, LEARNING_RATE, WEIGHT_DECAY


# ============================================================
# 2. STL-10 PYTORCH DATASET
# ============================================================

class STL10MAEDataset(Dataset):

    # Hugging Face veri setini ve MAE processor'ını sınıfa alıyoruz.
    def __init__(self, dataset, processor):

        # Ham STL-10 veri setini saklıyoruz.
        self.dataset = dataset

        # Görüntüleri MAE formatına çevirecek processor'ı saklıyoruz.
        self.processor = processor

    # Veri setindeki toplam görüntü sayısını döndürüyoruz.
    def __len__(self):

        # Hugging Face veri setinin uzunluğunu döndürüyoruz.
        return len(self.dataset)

    # Belirli bir görüntüyü modele uygun hale getiriyoruz.
    def __getitem__(self, index):

        # STL-10 görüntüsünü alıyoruz ancak label bilgisini almıyoruz.
        image = self.dataset[index]["image"].convert("RGB")

        # Görüntüyü resize, normalize ve Tensor işlemlerinden geçiriyoruz.
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]

        # Processor batch boyutu eklediği için ilk boyutu kaldırıyoruz.
        pixel_values = pixel_values.squeeze(0)

        # Yalnızca görüntüyü döndürüyoruz çünkü MAE label kullanmaz.
        return pixel_values


# ============================================================
# 3. VALIDATION LOSS FONKSİYONU
# ============================================================

# Validation sırasında gradient hesaplamıyoruz.
@torch.inference_mode()
def evaluate(model, loader, num_patches):

    # Modeli evaluation moduna geçiriyoruz.
    model.eval()

    # Toplam validation loss değerini saklıyoruz.
    total_loss = 0.0

    # İşlenen toplam görüntü sayısını saklıyoruz.
    total_samples = 0

    # Validation masking işleminin her epoch aynı olmasını sağlıyoruz.
    noise_generator = torch.Generator().manual_seed(SEED)

    # Validation batch'lerini dolaşıyoruz.
    for pixel_values in loader:

        # Batch büyüklüğünü alıyoruz.
        batch_size = pixel_values.size(0)

        # Her görüntü için sabit fakat rastgele patch sıralaması oluşturuyoruz.
        noise = torch.rand(batch_size, num_patches, generator=noise_generator)

        # Model masked patch'leri tahmin ederek reconstruction loss hesaplıyor.
        outputs = model(pixel_values=pixel_values, noise=noise)

        # Batch loss değerini görüntü sayısıyla ağırlıklandırıyoruz.
        total_loss += outputs.loss.item() * batch_size

        # Toplam görüntü sayısını artırıyoruz.
        total_samples += batch_size

    # Ortalama validation loss değerini hesaplıyoruz.
    average_loss = total_loss / total_samples

    # Ortalama loss değerini döndürüyoruz.
    return average_loss


# ============================================================
# 4. ANA PROGRAM
# ============================================================

def main():

    # Random işlemlerin tekrar üretilebilir olmasını sağlıyoruz.
    torch.manual_seed(SEED)

    # Bu projede yalnızca CPU kullanıyoruz.
    print("Device: CPU")

    # Model klasörünü oluşturuyoruz.
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # Output klasörünü oluşturuyoruz.
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


    # ========================================================
    # 5. IMAGE PROCESSOR
    # ========================================================

    # Pretrained MAE modelinin kullandığı image processor'ı yüklüyoruz.
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # Processor'ın görüntü boyutunu gösteriyoruz.
    print("\n========== IMAGE PROCESSOR ==========")

    # Kullanılan pretrained modeli gösteriyoruz.
    print("Model:", MODEL_NAME)

    # MAE giriş görüntü boyutunu gösteriyoruz.
    print("Image Size:", processor.size)


    # ========================================================
    # 6. STL-10 VERİ SETİ
    # ========================================================

    # STL-10 train split'inden yalnızca belirlediğimiz sayıda görüntüyü yüklüyoruz.
    raw_dataset = load_dataset(DATASET_NAME, split=f"train[:{TRAIN_SAMPLES}]")

    # STL-10 veri setini PyTorch Dataset yapısına dönüştürüyoruz.
    dataset = STL10MAEDataset(raw_dataset, processor)

    # Validation için ayrılacak görüntü sayısını hesaplıyoruz.
    val_size = int(len(dataset) * VAL_RATIO)

    # Training için kullanılacak görüntü sayısını hesaplıyoruz.
    train_size = len(dataset) - val_size

    # Train-validation ayrımının her çalıştırmada aynı olmasını sağlıyoruz.
    split_generator = torch.Generator().manual_seed(SEED)

    # Veri setini training ve validation olarak ayırıyoruz.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)

    # Training DataLoader oluşturuyoruz.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Validation DataLoader oluşturuyoruz.
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Veri seti bilgilerini terminalde gösteriyoruz.
    print("\n========== DATASET ==========")

    # Toplam kullanılan görüntü sayısını gösteriyoruz.
    print("Toplam Görüntü:", len(dataset))

    # Training görüntü sayısını gösteriyoruz.
    print("Train:", len(train_dataset))

    # Validation görüntü sayısını gösteriyoruz.
    print("Validation:", len(val_dataset))

    # Label kullanılmadığını özellikle gösteriyoruz.
    print("Label Kullanımı: Yok")


    # ========================================================
    # 7. PRETRAINED MAE MODELİ
    # ========================================================

    # ImageNet üzerinde MAE yöntemiyle pretrained modeli yüklüyoruz.
    model = ViTMAEForPreTraining.from_pretrained(MODEL_NAME)

    # Modelin masking oranını config.py değerine ayarlıyoruz.
    model.config.mask_ratio = MASK_RATIO

    # Modeli CPU üzerinde kullanıyoruz.
    model = model.to("cpu")

    # Modelin toplam parametre sayısını hesaplıyoruz.
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    # Modelin eğitilebilir parametre sayısını hesaplıyoruz.
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    # Toplam patch sayısını hesaplıyoruz.
    num_patches = (model.config.image_size // model.config.patch_size) ** 2

    # Görüntü içerisinde görünür kalan patch sayısını hesaplıyoruz.
    visible_patches = int(num_patches * (1 - MASK_RATIO))

    # Maskelenecek patch sayısını hesaplıyoruz.
    masked_patches = num_patches - visible_patches

    # Model bilgilerini gösteriyoruz.
    print("\n========== MODEL ==========")

    # Kullanılan model adını gösteriyoruz.
    print("Model:", MODEL_NAME)

    # Giriş görüntüsü boyutunu gösteriyoruz.
    print("Image Size:", model.config.image_size)

    # Patch boyutunu gösteriyoruz.
    print("Patch Size:", model.config.patch_size)

    # Toplam patch sayısını gösteriyoruz.
    print("Toplam Patch:", num_patches)

    # Masking oranını gösteriyoruz.
    print(f"Mask Ratio: %{MASK_RATIO * 100:.0f}")

    # Encoder'ın göreceği patch sayısını gösteriyoruz.
    print("Görünen Patch:", visible_patches)

    # Gizlenen patch sayısını gösteriyoruz.
    print("Maskelenen Patch:", masked_patches)

    # Toplam model parametresini gösteriyoruz.
    print(f"Toplam Parametre: {total_parameters:,}")

    # Eğitilebilir parametre sayısını gösteriyoruz.
    print(f"Eğitilebilir Parametre: {trainable_parameters:,}")


    # ========================================================
    # 8. OPTIMIZER
    # ========================================================

    # Pretrained MAE modelini düşük learning rate ile eğitmeye devam ediyoruz.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


    # ========================================================
    # 9. TRAINING
    # ========================================================

    # En düşük validation loss değerini saklıyoruz.
    best_val_loss = float("inf")

    # Training loss geçmişini saklıyoruz.
    train_history = []

    # Validation loss geçmişini saklıyoruz.
    val_history = []

    # Training başlangıcını gösteriyoruz.
    print("\n========== SELF-SUPERVISED TRAINING ==========")

    # Belirlenen epoch sayısı kadar eğitime devam ediyoruz.
    for epoch in range(EPOCHS):

        # Modeli training moduna geçiriyoruz.
        model.train()

        # Epoch toplam loss değerini saklıyoruz.
        total_train_loss = 0.0

        # Epoch boyunca işlenen görüntü sayısını saklıyoruz.
        total_train_samples = 0

        # Training batch'lerini progress bar ile dolaşıyoruz.
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        # Her training batch'ini işliyoruz.
        for pixel_values in progress_bar:

            # Önceki batch'ten kalan gradient değerlerini temizliyoruz.
            optimizer.zero_grad(set_to_none=True)

            # MAE görüntünün yüzde 75'ini maskeleyip eksik patch'leri tahmin ediyor.
            outputs = model(pixel_values=pixel_values)

            # Maskelenmiş patch'ler üzerindeki reconstruction loss değerini alıyoruz.
            loss = outputs.loss

            # Gradient değerlerini hesaplıyoruz.
            loss.backward()

            # Model ağırlıklarını güncelliyoruz.
            optimizer.step()

            # Batch loss değerini toplam loss değerine ekliyoruz.
            total_train_loss += loss.item() * pixel_values.size(0)

            # İşlenen görüntü sayısını artırıyoruz.
            total_train_samples += pixel_values.size(0)

            # Anlık reconstruction loss değerini progress bar üzerinde gösteriyoruz.
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Epoch ortalama training loss değerini hesaplıyoruz.
        train_loss = total_train_loss / total_train_samples

        # Modelin validation reconstruction loss değerini hesaplıyoruz.
        val_loss = evaluate(model, val_loader, num_patches)

        # Training loss geçmişine mevcut değeri ekliyoruz.
        train_history.append(train_loss)

        # Validation loss geçmişine mevcut değeri ekliyoruz.
        val_history.append(val_loss)

        # Epoch sonuçlarını terminalde gösteriyoruz.
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Reconstruction Loss: {train_loss:.4f} | Val Reconstruction Loss: {val_loss:.4f}")

        # Validation loss şimdiye kadarki en düşük değerse modeli kaydediyoruz.
        if val_loss < best_val_loss:

            # En iyi validation loss değerini güncelliyoruz.
            best_val_loss = val_loss

            # Eğitilmiş MAE modelini klasöre kaydediyoruz.
            model.save_pretrained(MODEL_SAVE_DIR, safe_serialization=True)

            # Image processor'ı aynı klasöre kaydediyoruz.
            processor.save_pretrained(MODEL_SAVE_DIR)

            # Modelin kaydedildiğini terminalde gösteriyoruz.
            print("Yeni en iyi MAE modeli kaydedildi.")

    # En iyi validation reconstruction loss değerini gösteriyoruz.
    print(f"\nBest Validation Reconstruction Loss: {best_val_loss:.4f}")


    # ========================================================
    # 10. LOSS GRAFİĞİ
    # ========================================================

    # Epoch numaralarını oluşturuyoruz.
    epochs = range(1, EPOCHS + 1)

    # Yeni bir figure oluşturuyoruz.
    plt.figure(figsize=(8, 5))

    # Training loss değerlerini çiziyoruz.
    plt.plot(epochs, train_history, marker="o", label="Train Loss")

    # Validation loss değerlerini çiziyoruz.
    plt.plot(epochs, val_history, marker="o", label="Validation Loss")

    # Grafiğe başlık ekliyoruz.
    plt.title("MAE Reconstruction Loss")

    # X eksenini isimlendiriyoruz.
    plt.xlabel("Epoch")

    # Y eksenini isimlendiriyoruz.
    plt.ylabel("Reconstruction Loss")

    # Grafik açıklamalarını gösteriyoruz.
    plt.legend()

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Loss grafiğini kaydediyoruz.
    plt.savefig(f"{OUTPUT_DIR}/mae_training_loss.png", dpi=200)

    # Grafik penceresini kapatıyoruz.
    plt.close()

    # Eğitim işleminin tamamlandığını gösteriyoruz.
    print("\nTraining tamamlandı.")

    # Modelin kaydedildiği klasörü gösteriyoruz.
    print("Model:", MODEL_SAVE_DIR)

    # Loss grafiğinin kaydedildiği yolu gösteriyoruz.
    print(f"Loss Grafiği: {OUTPUT_DIR}/mae_training_loss.png")


# ============================================================
# 11. WINDOWS İÇİN GÜVENLİ BAŞLANGIÇ
# ============================================================

# Program doğrudan çalıştırıldığında main fonksiyonunu başlatıyoruz.
if __name__ == "__main__":

    # Eğitim akışını başlatıyoruz.
    main()