# ============================================================
# 1. KÜTÜPHANELER
# ============================================================

# PyTorch işlemleri için kullanılır.
import torch

# STL-10 veri setini yüklemek için kullanılır.
from datasets import load_dataset

# Pretrained CLIP modelini ve processor'ı kullanmak için kullanılır.
from transformers import CLIPModel, AutoProcessor

# Arama işlemi sırasında progress bar göstermek için kullanılır.
from tqdm import tqdm

# Sonuç görsellerini çizmek için kullanılır.
import matplotlib.pyplot as plt

# Output klasörünü oluşturmak için kullanılır.
from pathlib import Path

# Proje ayarlarını config.py dosyasından alıyoruz.
from config import MODEL_NAME, DATASET_NAME, TEST_SAMPLES, BATCH_SIZE, SEED, CLASS_NAMES, TEXT_QUERY, TOP_K, OUTPUT_DIR


# ============================================================
# 2. ANA PROGRAM
# ============================================================

def main():

    # Random işlemlerin tekrar üretilebilir olmasını sağlıyoruz.
    torch.manual_seed(SEED)

    # Bu projede yalnızca CPU kullandığımızı gösteriyoruz.
    print("Device: CPU")

    # Output klasörü yoksa oluşturuyoruz.
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


    # ========================================================
    # 3. STL-10 TEST VERİ SETİ
    # ========================================================

    # STL-10 test bölümünü yüklüyoruz.
    dataset = load_dataset(DATASET_NAME, split="test")

    # Classification dosyasıyla aynı görüntü sırasını elde etmek için aynı seed ile karıştırıyoruz.
    dataset = dataset.shuffle(seed=SEED)

    # Arama yapılacak görüntü sayısını sınırlandırıyoruz.
    dataset = dataset.select(range(min(TEST_SAMPLES, len(dataset))))

    # Arama yapılacak görüntü sayısını gösteriyoruz.
    print("\n========== DATASET ==========")

    # Toplam görüntü sayısını gösteriyoruz.
    print("Search Images:", len(dataset))


    # ========================================================
    # 4. PRETRAINED CLIP
    # ========================================================

    # CLIP processor'ını yüklüyoruz.
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Pretrained CLIP modelini yüklüyoruz.
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Modeli CPU üzerinde kullanıyoruz.
    model = model.to("cpu")

    # Modeli evaluation moduna geçiriyoruz.
    model.eval()

    # Model parametrelerinin tamamını freeze ediyoruz.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Model bilgilerini gösteriyoruz.
    print("\n========== CLIP ==========")

    # Model adını gösteriyoruz.
    print("Model:", MODEL_NAME)

    # Arama sorgusunu gösteriyoruz.
    print("Text Query:", TEXT_QUERY)

    # Gösterilecek en yakın görüntü sayısını gösteriyoruz.
    print("Top K:", TOP_K)


    # ========================================================
    # 5. IMAGE-TEXT SIMILARITY
    # ========================================================

    # Tüm similarity değerlerini saklamak için liste oluşturuyoruz.
    all_similarities = []

    # Arama başlangıcını gösteriyoruz.
    print("\n========== TEXT-TO-IMAGE SEARCH ==========")

    # Veri setini batch'ler halinde dolaşıyoruz.
    for start_index in tqdm(range(0, len(dataset), BATCH_SIZE), desc="CLIP Search"):

        # Batch'in son indeksini hesaplıyoruz.
        end_index = min(start_index + BATCH_SIZE, len(dataset))

        # Batch indekslerini oluşturuyoruz.
        batch_indices = range(start_index, end_index)

        # Batch görüntülerini RGB formatında alıyoruz.
        images = [dataset[index]["image"].convert("RGB") for index in batch_indices]

        # Tek text query ve mevcut görüntü batch'ini CLIP formatına dönüştürüyoruz.
        inputs = processor(
            text=[TEXT_QUERY],
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Gradient hesaplamadan CLIP modelini çalıştırıyoruz.
        with torch.inference_mode():

            # Görüntü ve metin embedding'lerini oluşturuyoruz.
            outputs = model(**inputs)

        # Normalize edilmiş görüntü embedding'lerini alıyoruz.
        image_embeddings = outputs.image_embeds

        # Normalize edilmiş text embedding'ini alıyoruz.
        text_embedding = outputs.text_embeds

        # Her görüntünün text query ile cosine similarity değerini hesaplıyoruz.
        similarities = image_embeddings @ text_embedding.T

        # Gereksiz son boyutu kaldırıyoruz.
        similarities = similarities.squeeze(1)

        # Batch similarity değerlerini listeye ekliyoruz.
        all_similarities.append(similarities.cpu())

    # Tüm batch similarity değerlerini tek Tensor içerisinde birleştiriyoruz.
    all_similarities = torch.cat(all_similarities, dim=0)


    # ========================================================
    # 6. EN BENZER GÖRÜNTÜLERİ BUL
    # ========================================================

    # En yüksek similarity değerine sahip görüntüleri seçiyoruz.
    top_results = torch.topk(all_similarities, k=min(TOP_K, len(dataset)))

    # En benzer görüntülerin dataset indekslerini alıyoruz.
    top_indices = top_results.indices.tolist()

    # En yüksek similarity skorlarını alıyoruz.
    top_scores = top_results.values.tolist()

    # Sonuçları terminalde gösteriyoruz.
    print("\n========== SEARCH RESULTS ==========")

    # Text query değerini gösteriyoruz.
    print("Query:", TEXT_QUERY)

    # En yakın görüntülerin tamamını dolaşıyoruz.
    for rank, (dataset_index, score) in enumerate(zip(top_indices, top_scores), start=1):

        # Görüntünün gerçek STL-10 sınıfını alıyoruz.
        label = int(dataset[dataset_index]["label"])

        # Sonucu terminalde gösteriyoruz.
        print(
            f"{rank}. "
            f"Class: {CLASS_NAMES[label]} | "
            f"Similarity: {score:.4f}"
        )


    # ========================================================
    # 7. ARAMA SONUÇLARINI GÖRSELLEŞTİR
    # ========================================================

    # Top K sonucu yan yana göstermek için figure oluşturuyoruz.
    figure, axes = plt.subplots(1, len(top_indices), figsize=(3 * len(top_indices), 4))

    # TOP_K bir olduğunda axes yapısını liste haline getiriyoruz.
    if len(top_indices) == 1:
        axes = [axes]

    # En yakın görüntülerin tamamını dolaşıyoruz.
    for position, (axis, dataset_index, score) in enumerate(zip(axes, top_indices, top_scores), start=1):

        # Görüntüyü veri setinden alıyoruz.
        image = dataset[dataset_index]["image"].convert("RGB")

        # Görüntünün gerçek sınıf indeksini alıyoruz.
        label = int(dataset[dataset_index]["label"])

        # Görüntüyü grafikte gösteriyoruz.
        axis.imshow(image)

        # Sıra, sınıf ve similarity değerini başlıkta gösteriyoruz.
        axis.set_title(
            f"#{position}\n"
            f"{CLASS_NAMES[label]}\n"
            f"Similarity: {score:.3f}"
        )

        # Koordinat eksenlerini kapatıyoruz.
        axis.axis("off")

    # Ana başlık olarak doğal dil sorgusunu gösteriyoruz.
    figure.suptitle(f'Text Query: "{TEXT_QUERY}"', fontsize=14)

    # Grafik yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Text-to-image search sonucunu kaydediyoruz.
    plt.savefig(
        f"{OUTPUT_DIR}/clip_text_to_image_search.png",
        dpi=200,
        bbox_inches="tight",
    )

    # Grafik penceresini kapatıyoruz.
    plt.close()


    # ========================================================
    # 8. PROGRAM SONU
    # ========================================================

    # Arama işleminin tamamlandığını gösteriyoruz.
    print("\nText-to-image search tamamlandı.")

    # Oluşturulan görsel dosyasını gösteriyoruz.
    print(f"Search Result: {OUTPUT_DIR}/clip_text_to_image_search.png")


# ============================================================
# 9. WINDOWS İÇİN GÜVENLİ BAŞLANGIÇ
# ============================================================

# Program doğrudan çalıştırıldığında main fonksiyonunu başlatıyoruz.
if __name__ == "__main__":

    # Text-to-image search akışını çalıştırıyoruz.
    main()