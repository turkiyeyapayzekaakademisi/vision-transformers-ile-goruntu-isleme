
import torch  # PyTorch kütüphanesi
import torch.nn as nn  # Sinir ağı katmanları
import timm  # Hazır pretrained görüntü modelleri
from torch.utils.data import DataLoader, random_split  # Veri yükleme ve bölme
from torchvision import datasets, transforms  # MNIST veri seti ve görüntü dönüşümleri
from sklearn.metrics import accuracy_score, confusion_matrix  # Başarı metrikleri

# 1. PARAMETRELER
SEED = 42  # Rastgele işlemlerin tekrarlanabilir olmasını sağlar
IMAGE_SIZE = 112  # Pretrained modele verilecek görüntü boyutu
NUM_CLASSES = 10  # MNIST rakam sınıfları: 0-9
BATCH_SIZE = 64  # Bir eğitim adımındaki görüntü sayısı
EPOCHS = 3  # Eğitim turu sayısı
LEARNING_RATE = 0.0002  # Ağırlıkların güncellenme hızı
TRAINABLE_BLOCKS = 4  # Eğitilecek son Transformer Encoder bloğu sayısı
MODEL_NAME = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"  # Pretrained ViT-Tiny modeli

torch.manual_seed(SEED)  # Rastgele başlangıç değerlerini sabitle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU varsa GPU, yoksa CPU kullan
print("Kullanılan cihaz:", device)  # Kullanılan donanımı göster

# 2. VERİ SETİ
transform = transforms.Compose([  # Görüntü ön işleme adımları
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 28x28 görüntüleri 112x112 boyutuna getir
    transforms.Grayscale(num_output_channels=3),  # Tek kanallı görüntüyü pretrained model için 3 kanala dönüştür
    transforms.ToTensor(),  # Pikselleri 0-1 aralığında tensöre dönüştür
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Pretrained modele uygun şekilde -1 ile 1 arasına normalize et
])  # Görüntü dönüşümlerini tamamla

data = datasets.MNIST("./data", train=True, download=True, transform=transform)  # 60.000 eğitim görüntüsünü yükle
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)  # 10.000 test görüntüsünü yükle
train_data, val_data = random_split(data, [54000, 6000], generator=torch.Generator().manual_seed(SEED))  # Önceki projeyle aynı veri ayrımını oluştur

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)  # Eğitim verilerini karıştırarak yükle
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)  # Validation verilerini yükle
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)  # Test verilerini yükle

print(f"Eğitim: {len(train_data)} | Validation: {len(val_data)} | Test: {len(test_data)}")  # Veri dağılımını göster

# 3. PRETRAINED ViT MODELİNİ YÜKLEME
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES, img_size=IMAGE_SIZE)  # Hazır ağırlıkları yükle ve çıkış katmanını 10 sınıfa uyarla

for param in model.parameters(): param.requires_grad = False  # Önceden eğitilmiş bütün ağırlıkları dondur
for block in model.blocks[-TRAINABLE_BLOCKS:]:  # Son 4 Transformer bloğunu seç
    for param in block.parameters(): param.requires_grad = True  # Seçilen Transformer ağırlıklarını eğitime aç
for param in model.norm.parameters(): param.requires_grad = True  # Son normalizasyon katmanını eğitime aç
for param in model.head.parameters(): param.requires_grad = True  # Yeni sınıflandırma katmanını eğitime aç

model = model.to(device)  # Modeli CPU veya GPU'ya taşı
total_params = sum(p.numel() for p in model.parameters())  # Toplam parametre sayısı
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Eğitilecek parametre sayısı

print(f"Toplam Parametre: {total_params:,}")  # Modelin toplam büyüklüğünü göster
print(f"Eğitilebilir Parametre: {trainable_params:,}")  # Güncellenecek ağırlık sayısını göster

criterion = nn.CrossEntropyLoss()  # Çok sınıflı sınıflandırma kayıp fonksiyonu
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.01)  # Yalnızca eğitime açık ağırlıkları güncelle
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)  # Öğrenme hızını epoch ilerledikçe azalt

# 4. DEĞERLENDİRME
def evaluate(loader):  # Validation veya test doğruluğunu hesapla
    model.eval()  # Modeli değerlendirme moduna geçir
    actual, predicted = [], []  # Gerçek etiketleri ve tahminleri sakla

    with torch.no_grad():  # Gradyan hesaplamalarını kapat
        for images, labels in loader:  # Verileri batch halinde dolaş
            outputs = model(images.to(device))  # Modelden sınıf skorlarını al
            predicted.extend(outputs.argmax(1).cpu().tolist())  # En yüksek skorlu sınıfı seç
            actual.extend(labels.tolist())  # Gerçek etiketleri kaydet

    return accuracy_score(actual, predicted), actual, predicted  # Doğruluk ve tahminleri döndür

# 5. FINE-TUNING
best_acc = 0  # En yüksek validation doğruluğunu sakla

for epoch in range(EPOCHS):  # Eğitim döngüsünü başlat
    model.train()  # Modeli eğitim moduna geçir
    total_loss, correct = 0, 0  # Kayıp ve doğru tahmin sayaçları

    for images, labels in train_loader:  # Eğitim verilerini batch halinde dolaş
        images, labels = images.to(device), labels.to(device)  # Verileri CPU veya GPU'ya taşı
        optimizer.zero_grad()  # Önceki gradyanları sıfırla
        outputs = model(images)  # İleri besleme ile tahmin üret
        loss = criterion(outputs, labels)  # Tahmin hatasını hesapla
        loss.backward()  # Backpropagation ile gradyanları hesapla
        optimizer.step()  # Eğitime açık ağırlıkları güncelle
        total_loss += loss.item() * len(labels)  # Batch kaybını toplam kayba ekle
        correct += (outputs.argmax(1) == labels).sum().item()  # Doğru tahminleri say

    scheduler.step()  # Öğrenme hızını güncelle
    val_acc, _, _ = evaluate(val_loader)  # Validation doğruluğunu hesapla
    train_acc = correct / len(train_data)  # Eğitim doğruluğunu hesapla
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_data):.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")  # Epoch sonuçlarını yazdır

    if val_acc > best_acc:  # Validation doğruluğu iyileşti mi kontrol et
        best_acc = val_acc  # En iyi doğruluğu güncelle
        torch.save(model.state_dict(), "vit_pretrained.pth")  # En başarılı modelin ağırlıklarını kaydet

# 6. TEST
model.load_state_dict(torch.load("vit_pretrained.pth", map_location=device, weights_only=True))  # En iyi model ağırlıklarını yükle
test_acc, actual, predicted = evaluate(test_loader)  # Daha önce görülmeyen test görüntülerini sınıflandır

print(f"\nEn iyi Validation Accuracy: {best_acc:.4f}")  # En yüksek validation doğruluğu
print(f"Test Accuracy: {test_acc:.4f}")  # Test doğruluğu
print("Confusion Matrix:\n", confusion_matrix(actual, predicted))  # Sınıflandırma hata matrisi

for i in range(10):  # İlk 10 test görüntüsünü dolaş
    print(f"Gerçek: {actual[i]} | Tahmin: {predicted[i]}")  # Gerçek rakam ve model tahminini göster

print("Model kaydedildi: vit_pretrained.pth")  # Model dosyasının adını göster
