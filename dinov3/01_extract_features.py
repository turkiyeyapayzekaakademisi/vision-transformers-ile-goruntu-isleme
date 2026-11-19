# ============================================================
# 1. KÜTÜPHANELER
# ============================================================

# PyTorch işlemleri için kullanılır.
import torch

# PyTorch Dataset ve DataLoader yapıları için kullanılır.
from torch.utils.data import Dataset, DataLoader

# STL-10 veri setini Hugging Face üzerinden yüklemek için kullanılır.
from datasets import load_dataset

# DINOv3 modelini ve image processor'ı yüklemek için kullanılır.
from transformers import AutoImageProcessor, AutoModel

# Feature extraction sırasında progress bar göstermek için kullanılır.
from tqdm import tqdm

# Klasör oluşturmak için kullanılır.
from pathlib import Path

# Proje ayarlarını config.py dosyasından alıyoruz.
from config import MODEL_NAME, DATASET_NAME, TRAIN_SAMPLES, TEST_SAMPLES, BATCH_SIZE, NUM_WORKERS, SEED, CLASS_NAMES, FEATURE_DIR, MODEL_DIR, OUTPUT_DIR, TRAIN_FEATURE_PATH, TEST_FEATURE_PATH


# ============================================================
# 2. STL-10 DATASET SINIFI
# ============================================================

class STL10Dataset(Dataset):

    # Hugging Face veri setini PyTorch Dataset yapısına dönüştürüyoruz.
    def __init__(self, dataset):

        # Ham STL-10 veri setini saklıyoruz.
        self.dataset = dataset

    # Veri setindeki görüntü sayısını döndürüyoruz.
    def __len__(self):

        # Veri setinin toplam uzunluğunu döndürüyoruz.
        return len(self.dataset)

    # Belirli bir görüntü ve label bilgisini döndürüyoruz.
    def __getitem__(self, index):

        # Görüntüyü RGB formatında alıyoruz.
        image = self.dataset[index]["image"].convert("RGB")

        # Görüntünün sınıf indeksini alıyoruz.
        label = int(self.dataset[index]["label"])

        # Görüntü ve label bilgisini döndürüyoruz.
        return image, label


# ============================================================
# 3. FEATURE EXTRACTION FONKSİYONU
# ============================================================

def extract_features(model, processor, dataset, description):

    # Görüntüleri batch içerisinde processor'dan geçirmek için collate fonksiyonu oluşturuyoruz.
    def collate_fn(batch):

        # Batch içerisindeki PIL görüntülerini listeye alıyoruz.
        images = [item[0] for item in batch]

        # Batch içerisindeki sınıf indekslerini Tensor'a dönüştürüyoruz.
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

        # Görüntüleri DINOv3 modelinin beklediği Tensor formatına dönüştürüyoruz.
        inputs = processor(images=images, return_tensors="pt")

        # Pixel Tensor ve label Tensor değerlerini döndürüyoruz.
        return inputs["pixel_values"], labels

    # Feature extraction için DataLoader oluşturuyoruz.
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Tüm embedding'leri saklamak için liste oluşturuyoruz.
    all_features = []

    # Tüm sınıf indekslerini saklamak için liste oluşturuyoruz.
    all_labels = []

    # Modeli evaluation moduna geçiriyoruz.
    model.eval()

    # Gradient hesaplamadan feature extraction yapıyoruz.
    with torch.inference_mode():

        # Tüm batch'leri sırayla işliyoruz.
        for pixel_values, labels in tqdm(loader, desc=description):

            # Görüntüleri DINOv3 modelinden geçiriyoruz.
            outputs = model(pixel_values=pixel_values)

            # Görüntünün global embedding vektörünü alıyoruz.
            embeddings = outputs.pooler_output

            # Embedding'leri CPU üzerinde saklıyoruz.
            all_features.append(embeddings.cpu())

            # Label bilgilerini saklıyoruz.
            all_labels.append(labels.cpu())

    # Tüm embedding batch'lerini tek Tensor içerisinde birleştiriyoruz.
    features = torch.cat(all_features, dim=0)

    # Tüm label batch'lerini tek Tensor içerisinde birleştiriyoruz.
    labels = torch.cat(all_labels, dim=0)

    # Feature ve label Tensor'larını döndürüyoruz.
    return features, labels


# ============================================================
# 4. ANA PROGRAM
# ============================================================

def main():

    # Rastgele işlemlerin tekrar üretilebilir olmasını sağlıyoruz.
    torch.manual_seed(SEED)

    # Bu projede yalnızca CPU kullandığımızı gösteriyoruz.
    print("Device: CPU")

    # Feature klasörünü oluşturuyoruz.
    Path(FEATURE_DIR).mkdir(parents=True, exist_ok=True)

    # Model klasörünü oluşturuyoruz.
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # Output klasörünü oluşturuyoruz.
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


    # ========================================================
    # 5. STL-10 TRAIN DATA
    # ========================================================

    # STL-10 training split'ini yüklüyoruz.
    train_raw = load_dataset(DATASET_NAME, split="train")

    # Veri setini sabit seed ile karıştırıyoruz.
    train_raw = train_raw.shuffle(seed=SEED)

    # CPU üzerinde kullanılacak training örneklerini seçiyoruz.
    train_raw = train_raw.select(range(min(TRAIN_SAMPLES, len(train_raw))))

    # PyTorch Dataset yapısını oluşturuyoruz.
    train_dataset = STL10Dataset(train_raw)


    # ========================================================
    # 6. STL-10 TEST DATA
    # ========================================================

    # STL-10 test split'ini yüklüyoruz.
    test_raw = load_dataset(DATASET_NAME, split="test")

    # Test görüntülerini de sabit seed ile karıştırıyoruz.
    test_raw = test_raw.shuffle(seed=SEED)

    # Belirlenen sayıda test görüntüsünü seçiyoruz.
    test_raw = test_raw.select(range(min(TEST_SAMPLES, len(test_raw))))

    # PyTorch test Dataset yapısını oluşturuyoruz.
    test_dataset = STL10Dataset(test_raw)

    # Veri seti bilgilerini gösteriyoruz.
    print("\n========== DATASET ==========")

    # Training görüntü sayısını gösteriyoruz.
    print("Train:", len(train_dataset))

    # Test görüntü sayısını gösteriyoruz.
    print("Test:", len(test_dataset))

    # Sınıf sayısını gösteriyoruz.
    print("Sınıf Sayısı:", len(CLASS_NAMES))


    # ========================================================
    # 7. DINOv3 IMAGE PROCESSOR
    # ========================================================

    # DINOv3'ün kullandığı resmi image processor'ı yüklüyoruz.
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)


    # ========================================================
    # 8. PRETRAINED DINOv3
    # ========================================================

    # Meta tarafından pretrained DINOv3 ViT-S/16 modelini yüklüyoruz.
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Modeli CPU üzerinde kullanıyoruz.
    model = model.to("cpu")

    # DINOv3 içerisindeki tüm parametreleri freeze ediyoruz.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Toplam model parametre sayısını hesaplıyoruz.
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    # Eğitilebilir parametre sayısını hesaplıyoruz.
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    # Model bilgilerini gösteriyoruz.
    print("\n========== DINOv3 ==========")

    # Model adını gösteriyoruz.
    print("Model:", MODEL_NAME)

    # Toplam parametre sayısını gösteriyoruz.
    print(f"Toplam Parametre: {total_parameters:,}")

    # Frozen olduğu için eğitilebilir parametre sayısının sıfır olmasını bekliyoruz.
    print(f"Eğitilebilir Parametre: {trainable_parameters:,}")

    # Backbone durumunu gösteriyoruz.
    print("Backbone: FROZEN")


    # ========================================================
    # 9. TRAIN FEATURE EXTRACTION
    # ========================================================

    # Training görüntülerini DINOv3 embedding'lerine dönüştürüyoruz.
    train_features, train_labels = extract_features(model, processor, train_dataset, "Train Feature Extraction")

    # Training feature bilgilerini gösteriyoruz.
    print("\nTrain Feature Shape:", tuple(train_features.shape))


    # ========================================================
    # 10. TEST FEATURE EXTRACTION
    # ========================================================

    # Test görüntülerini DINOv3 embedding'lerine dönüştürüyoruz.
    test_features, test_labels = extract_features(model, processor, test_dataset, "Test Feature Extraction")

    # Test feature bilgilerini gösteriyoruz.
    print("Test Feature Shape:", tuple(test_features.shape))


    # ========================================================
    # 11. FEATURE DOSYALARINI KAYDET
    # ========================================================

    # Training embedding ve label bilgilerini kaydediyoruz.
    torch.save(
        {
            "features": train_features,
            "labels": train_labels,
            "class_names": CLASS_NAMES,
        },
        TRAIN_FEATURE_PATH,
    )

    # Test embedding ve label bilgilerini kaydediyoruz.
    torch.save(
        {
            "features": test_features,
            "labels": test_labels,
            "class_names": CLASS_NAMES,
        },
        TEST_FEATURE_PATH,
    )

    # Feature extraction işleminin tamamlandığını gösteriyoruz.
    print("\nFeature extraction tamamlandı.")

    # Training feature dosyasını gösteriyoruz.
    print("Train Features:", TRAIN_FEATURE_PATH)

    # Test feature dosyasını gösteriyoruz.
    print("Test Features:", TEST_FEATURE_PATH)


# ============================================================
# 12. WINDOWS İÇİN GÜVENLİ BAŞLANGIÇ
# ============================================================

# Program doğrudan çalıştırıldığında main fonksiyonunu çalıştırıyoruz.
if __name__ == "__main__":

    # Feature extraction akışını başlatıyoruz.
    main()