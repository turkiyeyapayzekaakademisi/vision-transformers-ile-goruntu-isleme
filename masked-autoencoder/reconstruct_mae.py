# ============================================================
# 1. KÜTÜPHANELER
# ============================================================

# PyTorch işlemleri için kullanılır.
import torch

# STL-10 veri setini yüklemek için kullanılır.
from datasets import load_dataset

# Eğitilmiş processor ve MAE modelini yüklemek için kullanılır.
from transformers import AutoImageProcessor, ViTMAEForPreTraining

# Reconstruction sonuçlarını çizmek için kullanılır.
import matplotlib.pyplot as plt

# Output klasörünü oluşturmak için kullanılır.
from pathlib import Path

# Proje ayarlarını config.py dosyasından alıyoruz.
from config import MODEL_SAVE_DIR, OUTPUT_DIR, DATASET_NAME, RECONSTRUCTION_SAMPLES, MASK_RATIO, SEED


# ============================================================
# 2. NORMALIZATION'I GERİ ALMA FONKSİYONU
# ============================================================

def denormalize(pixel_values, processor):

    # Processor'ın ImageNet RGB ortalama değerlerini Tensor formatına dönüştürüyoruz.
    mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)

    # Processor'ın ImageNet RGB standart sapma değerlerini Tensor formatına dönüştürüyoruz.
    std = torch.tensor(processor.image_std).view(1, 3, 1, 1)

    # Normalize edilmiş görüntüyü tekrar 0-1 RGB aralığına dönüştürüyoruz.
    images = pixel_values * std + mean

    # Görüntü değerlerini geçerli 0-1 aralığında tutuyoruz.
    images = images.clamp(0, 1)

    # Görüntüleri döndürüyoruz.
    return images


# ============================================================
# 3. ANA PROGRAM
# ============================================================

def main():

    # Random masking işleminin her çalıştırmada aynı olmasını sağlıyoruz.
    torch.manual_seed(SEED)

    # Output klasörünü oluşturuyoruz.
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Bu projede yalnızca CPU kullandığımızı gösteriyoruz.
    print("Device: CPU")


    # ========================================================
    # 4. EĞİTİLMİŞ MODELİ YÜKLE
    # ========================================================

    # Training sırasında kaydedilen image processor'ı yüklüyoruz.
    processor = AutoImageProcessor.from_pretrained(MODEL_SAVE_DIR)

    # STL-10 üzerinde eğitime devam edilmiş MAE modelini yüklüyoruz.
    model = ViTMAEForPreTraining.from_pretrained(MODEL_SAVE_DIR)

    # Masking oranını yüzde 75 olarak ayarlıyoruz.
    model.config.mask_ratio = MASK_RATIO

    # Modeli CPU üzerinde kullanıyoruz.
    model = model.to("cpu")

    # Modeli evaluation moduna geçiriyoruz.
    model.eval()

    # Model bilgilerini gösteriyoruz.
    print("\n========== MODEL ==========")

    # Modelin giriş görüntüsü boyutunu gösteriyoruz.
    print("Image Size:", model.config.image_size)

    # Modelin patch boyutunu gösteriyoruz.
    print("Patch Size:", model.config.patch_size)

    # Modelin masking oranını gösteriyoruz.
    print(f"Mask Ratio: %{model.config.mask_ratio * 100:.0f}")


    # ========================================================
    # 5. STL-10 ÖRNEKLERİNİ YÜKLE
    # ========================================================

    # STL-10 train split'inden yalnızca göstereceğimiz kadar görüntü yüklüyoruz.
    dataset = load_dataset(DATASET_NAME, split=f"train[:{RECONSTRUCTION_SAMPLES}]")

    # PIL görüntülerini bir listeye alıyoruz.
    images = [sample["image"].convert("RGB") for sample in dataset]

    # Görüntüleri MAE modelinin beklediği Tensor formatına dönüştürüyoruz.
    inputs = processor(images=images, return_tensors="pt")

    # Normalize edilmiş görüntü Tensor'larını alıyoruz.
    pixel_values = inputs["pixel_values"]

    # Modelin kullandığı patch boyutunu alıyoruz.
    patch_size = model.config.patch_size

    # Her patch içerisindeki toplam RGB piksel değerinin sayısını hesaplıyoruz.
    patch_pixels = patch_size * patch_size * model.config.num_channels


    # ========================================================
    # 6. MAE RECONSTRUCTION
    # ========================================================

    # Gradient hesaplamadan reconstruction yapıyoruz.
    with torch.no_grad():

        # Model görüntünün yüzde 75'ini maskeleyip eksik patch'leri tahmin ediyor.
        outputs = model(pixel_values=pixel_values)

    # Reconstruction loss değerini alıyoruz.
    reconstruction_loss = outputs.loss.item()

    # Decoder tarafından tahmin edilen patch'leri alıyoruz.
    predicted_patches = outputs.logits

    # Hangi patch'lerin maskelendiğini alıyoruz.
    patch_mask = outputs.mask

    # Tahmin edilen patch'leri tekrar görüntü formatına dönüştürüyoruz.
    predicted_images = model.unpatchify(predicted_patches)

    # Patch mask'ini her patch içerisindeki tüm RGB piksel değerlerine yayıyoruz.
    expanded_mask = patch_mask.unsqueeze(-1).repeat(1, 1, patch_pixels)

    # Patch mask'ini 224x224 görüntü maskesine dönüştürüyoruz.
    mask_images = model.unpatchify(expanded_mask)

    # Orijinal normalize görüntünün görünen bölümlerini koruyoruz.
    visible_pixels = pixel_values * (1 - mask_images)

    # Model tahminlerinin yalnızca maskelenmiş bölgelerini alıyoruz.
    predicted_masked_pixels = predicted_images * mask_images

    # Görünen gerçek patch'ler ile tahmin edilen gizli patch'leri birleştiriyoruz.
    completed_images = visible_pixels + predicted_masked_pixels


    # ========================================================
    # 7. GÖRSEL FORMATINA DÖNÜŞTÜR
    # ========================================================

    # Orijinal görüntülerin normalization işlemini geri alıyoruz.
    original_images = denormalize(pixel_values, processor)

    # Maskelenmiş görüntüyü oluşturmak için maskelenen bölgeleri siyaha çeviriyoruz.
    masked_images = original_images * (1 - mask_images)

    # Reconstruction sonucunun normalization işlemini geri alıyoruz.
    reconstructed_images = denormalize(completed_images, processor)


    # ========================================================
    # 8. MASK BİLGİLERİ
    # ========================================================

    # Toplam patch sayısını hesaplıyoruz.
    total_patches = patch_mask.shape[1]

    # İlk görüntüde maskelenen patch sayısını hesaplıyoruz.
    masked_patches = int(patch_mask[0].sum().item())

    # İlk görüntüde görünür kalan patch sayısını hesaplıyoruz.
    visible_patches = total_patches - masked_patches

    # Reconstruction bilgilerini gösteriyoruz.
    print("\n========== RECONSTRUCTION ==========")

    # Reconstruction loss değerini gösteriyoruz.
    print(f"Reconstruction Loss: {reconstruction_loss:.4f}")

    # Toplam patch sayısını gösteriyoruz.
    print("Toplam Patch:", total_patches)

    # Görünür patch sayısını gösteriyoruz.
    print("Görünen Patch:", visible_patches)

    # Maskelenmiş patch sayısını gösteriyoruz.
    print("Maskelenen Patch:", masked_patches)


    # ========================================================
    # 9. TEK GÖRÜNTÜ RECONSTRUCTION
    # ========================================================

    # Original, Masked ve Reconstructed için üç grafik oluşturuyoruz.
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Orijinal görüntüyü gösteriyoruz.
    axes[0].imshow(original_images[0].permute(1, 2, 0))

    # İlk görselin başlığını belirliyoruz.
    axes[0].set_title("Original")

    # Maskelenmiş görüntüyü gösteriyoruz.
    axes[1].imshow(masked_images[0].permute(1, 2, 0))

    # Masked görselin başlığını belirliyoruz.
    axes[1].set_title("Masked - 75%")

    # Reconstruction sonucunu gösteriyoruz.
    axes[2].imshow(reconstructed_images[0].permute(1, 2, 0))

    # Reconstruction görselinin başlığını belirliyoruz.
    axes[2].set_title("Reconstructed")

    # Tüm grafiklerde koordinat eksenlerini kapatıyoruz.
    for axis in axes:

        # X ve Y eksenlerini gizliyoruz.
        axis.axis("off")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Tek görüntü reconstruction sonucunu kaydediyoruz.
    plt.savefig(f"{OUTPUT_DIR}/mae_single_reconstruction.png", dpi=200, bbox_inches="tight")

    # Grafik penceresini kapatıyoruz.
    plt.close()


    # ========================================================
    # 10. BİRDEN FAZLA RECONSTRUCTION
    # ========================================================

    # Her görüntü için Original, Masked ve Reconstructed sütunları oluşturuyoruz.
    figure, axes = plt.subplots(RECONSTRUCTION_SAMPLES, 3, figsize=(10, 3 * RECONSTRUCTION_SAMPLES))

    # Tüm örnek görüntüleri dolaşıyoruz.
    for index in range(RECONSTRUCTION_SAMPLES):

        # Orijinal görüntüyü ilk sütunda gösteriyoruz.
        axes[index, 0].imshow(original_images[index].permute(1, 2, 0))

        # Maskelenmiş görüntüyü ikinci sütunda gösteriyoruz.
        axes[index, 1].imshow(masked_images[index].permute(1, 2, 0))

        # Reconstruction sonucunu üçüncü sütunda gösteriyoruz.
        axes[index, 2].imshow(reconstructed_images[index].permute(1, 2, 0))

        # İlk sütun başlığını yalnızca ilk satırda gösteriyoruz.
        if index == 0:
            axes[index, 0].set_title("Original")

        # İkinci sütun başlığını yalnızca ilk satırda gösteriyoruz.
        if index == 0:
            axes[index, 1].set_title("Masked - 75%")

        # Üçüncü sütun başlığını yalnızca ilk satırda gösteriyoruz.
        if index == 0:
            axes[index, 2].set_title("Reconstructed")

        # Her satırdaki üç grafiğin eksenlerini kapatıyoruz.
        for column in range(3):

            # Koordinat eksenlerini gizliyoruz.
            axes[index, column].axis("off")

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Birden fazla reconstruction sonucunu kaydediyoruz.
    plt.savefig(f"{OUTPUT_DIR}/mae_multiple_reconstructions.png", dpi=200, bbox_inches="tight")

    # Grafik penceresini kapatıyoruz.
    plt.close()


    # ========================================================
    # 11. PROGRAM SONU
    # ========================================================

    # Reconstruction işleminin tamamlandığını gösteriyoruz.
    print("\nReconstruction tamamlandı.")

    # Tek örnek reconstruction dosyasını gösteriyoruz.
    print(f"Tek Görüntü: {OUTPUT_DIR}/mae_single_reconstruction.png")

    # Çoklu reconstruction dosyasını gösteriyoruz.
    print(f"Çoklu Görüntü: {OUTPUT_DIR}/mae_multiple_reconstructions.png")


# ============================================================
# 12. WINDOWS İÇİN GÜVENLİ BAŞLANGIÇ
# ============================================================

# Program doğrudan çalıştırıldığında main fonksiyonunu başlatıyoruz.
if __name__ == "__main__":

    # Reconstruction akışını çalıştırıyoruz.
    main()