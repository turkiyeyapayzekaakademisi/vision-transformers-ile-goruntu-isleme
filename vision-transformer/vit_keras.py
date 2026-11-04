
import torch  # PyTorch
import torch.nn as nn  # Sinir ağı katmanları
from torch.utils.data import DataLoader, random_split  # Veri yükleme ve bölme
from torchvision import datasets, transforms  # Hazır veri setleri
from sklearn.metrics import accuracy_score, confusion_matrix  # Başarı metrikleri

# 1. PARAMETRELER
SEED = 42  # Rastgele işlemlerin sabitlenmesi
IMAGE_SIZE = 28  # Görüntü boyutu: 28x28
PATCH_SIZE = 4  # Patch boyutu: 4x4, toplam 49 patch
EMBED_DIM = 64  # Her patch'in vektör boyutu
NUM_HEADS = 4  # Attention başlığı sayısı
NUM_LAYERS = 2  # Transformer Encoder sayısı
NUM_CLASSES = 10  # Rakam sınıfları: 0-9
BATCH_SIZE = 128  # Bir adımda işlenen görüntü sayısı
EPOCHS = 5  # Eğitim turu sayısı
LEARNING_RATE = 0.0005  # Öğrenme hızı

torch.manual_seed(SEED)  # Tekrarlanabilir sonuçlar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU varsa kullan
print("Cihaz:", device)  # Kullanılan donanım

# 2. VERİ SETİ
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # Görüntüyü tensöre dönüştür ve -1 ile 1 arasına normalize et
data = datasets.MNIST("./data", train=True, download=True, transform=transform)  # 60.000 eğitim görüntüsünü indir
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)  # 10.000 test görüntüsünü indir
train_data, val_data = random_split(data, [54000, 6000], generator=torch.Generator().manual_seed(SEED))  # Eğitim ve validation ayır
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)  # Eğitim batch'leri
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)  # Validation batch'leri
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)  # Test batch'leri

print(f"Eğitim: {len(train_data)} | Validation: {len(val_data)} | Test: {len(test_data)}")  # Veri sayıları

# 3. ViT MİMARİSİ
class ViT(nn.Module):  # Vision Transformer sınıfı
    def __init__(self):  # Model katmanlarını oluştur
        super().__init__()  # PyTorch temel sınıfını başlat
        num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2  # Patch sayısı: 49
        self.patch = nn.Conv2d(1, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)  # Gri görüntüyü patch embedding'lerine dönüştür
        self.cls = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))  # Öğrenilebilir sınıflandırma token'ı
        self.pos = nn.Parameter(torch.randn(1, num_patches + 1, EMBED_DIM) * 0.02)  # Patch ve CLS token konum bilgileri
        encoder = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=128, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)  # Attention, MLP, dropout ve ön normalizasyon içeren blok
        self.transformer = nn.TransformerEncoder(encoder, num_layers=NUM_LAYERS)  # İki Transformer Encoder bloğu
        self.norm = nn.LayerNorm(EMBED_DIM)  # CLS çıktısını normalize et
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)  # 10 rakam sınıfına ait skorları üret

    def forward(self, x):  # İleri besleme
        x = self.patch(x).flatten(2).transpose(1, 2)  # [Batch, 49, 64] patch token'ları
        cls = self.cls.expand(x.size(0), -1, -1)  # CLS token'ını her görüntü için oluştur
        x = torch.cat([cls, x], dim=1)  # CLS token'ını başa ekle
        x = x + self.pos  # Konum bilgisini ekle
        x = self.transformer(x)  # Self-Attention ile token'ları işle
        x = self.norm(x[:, 0])  # CLS token çıktısını normalize et
        return self.classifier(x)  # 10 sınıf için ham skorları döndür

model = ViT().to(device)  # Modeli oluştur
criterion = nn.CrossEntropyLoss()  # Çok sınıflı sınıflandırma kaybı
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)  # Ağırlıkları güncelleyen optimizer
print("Parametre sayısı:", sum(p.numel() for p in model.parameters()))  # Model büyüklüğü

# 4. DEĞERLENDİRME
def evaluate(loader):  # Validation veya test fonksiyonu
    model.eval()  # Değerlendirme moduna geç
    actual, predicted = [], []  # Gerçek ve tahmin etiketleri
    with torch.no_grad():  # Gradyan hesaplamasını kapat
        for images, labels in loader:  # Batch'leri dolaş
            outputs = model(images.to(device))  # Sınıf skorlarını hesapla
            predicted.extend(outputs.argmax(1).cpu().tolist())  # En yüksek skorlu sınıfı seç
            actual.extend(labels.tolist())  # Gerçek etiketleri kaydet
    return accuracy_score(actual, predicted), actual, predicted  # Doğruluk ve etiketleri döndür

# 5. EĞİTİM
best_acc = 0  # En yüksek validation doğruluğu
for epoch in range(EPOCHS):  # Eğitim döngüsü
    model.train()  # Eğitim modunu aç
    total_loss, correct = 0, 0  # Loss ve doğru tahmin sayacı
    for images, labels in train_loader:  # Eğitim batch'lerini dolaş
        images, labels = images.to(device), labels.to(device)  # Verileri CPU/GPU'ya taşı
        optimizer.zero_grad()  # Gradyanları sıfırla
        outputs = model(images)  # Tahmin üret
        loss = criterion(outputs, labels)  # Hata hesapla
        loss.backward()  # Backpropagation
        optimizer.step()  # Ağırlıkları güncelle
        total_loss += loss.item() * len(labels)  # Toplam kaybı hesapla
        correct += (outputs.argmax(1) == labels).sum().item()  # Doğru tahminleri say

    val_acc, _, _ = evaluate(val_loader)  # Validation doğruluğunu hesapla
    train_acc = correct / len(train_data)  # Eğitim doğruluğunu hesapla
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_data):.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")  # Sonuçları yazdır

    if val_acc > best_acc:  # En iyi modeli kontrol et
        best_acc = val_acc  # En iyi doğruluğu güncelle
        torch.save(model.state_dict(), "vit_sifirdan.pth")  # En başarılı ağırlıkları kaydet

# 6. TEST
model.load_state_dict(torch.load("vit_sifirdan.pth", map_location=device, weights_only=True))  # En iyi modeli yükle
test_acc, actual, predicted = evaluate(test_loader)  # Test görüntülerini sınıflandır
print(f"\nEn iyi Validation Accuracy: {best_acc:.4f}")  # En iyi validation sonucu
print(f"Test Accuracy: {test_acc:.4f}")  # Test başarısı
print("Confusion Matrix:\n", confusion_matrix(actual, predicted))  # 10 sınıflı hata matrisi

for i in range(10):  # İlk 10 test görüntüsü
    print(f"Gerçek: {actual[i]} | Tahmin: {predicted[i]}")  # Gerçek ve tahmin karşılaştırması

print("Model kaydedildi: vit_sifirdan.pth")  # Model dosyası
