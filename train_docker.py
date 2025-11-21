import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from model import StrongSpeechCNN, get_mel_transform

# === Шлях до папки з датасетом всередині контейнера/проєкту ===
# torchaudio сам створить цю папку і завантажить туди SpeechCommands
DATA_ROOT = "./SpeechCommands"

# === Класи, які ми використовуємо ===
selected_commands = ["yes", "no", "up", "down"]
num_classes = len(selected_commands)

batch_size = 32
epochs = 5 

# === Dataset-клас ===
class MySpeechCommands(SPEECHCOMMANDS):
    def __init__(self, root, subset, allowed_labels, transform, augment=False):
        # ВАЖЛИВО: download=True, щоб у контейнері сам завантажив датасет
        super().__init__(root, download=True)
        self.transform = transform
        self.allowed_labels = allowed_labels
        self.augment = augment  # чи вмикати data augmentation

        # --- читаємо списки train/val/test, які йдуть разом із датасетом ---
        def load_list(filename):
            path = os.path.join(self._path, filename)
            with open(path) as f:
                return [os.path.join(self._path, line.strip()) for line in f]

        if subset == "train":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            self._walker = [w for w in self._walker if w not in excludes]
        elif subset == "val":
            self._walker = load_list("validation_list.txt")
        elif subset == "test":
            self._walker = load_list("testing_list.txt")

        # --- фільтруємо тільки потрібні команди ---
        filtered = []
        for path in self._walker:
            label = os.path.basename(os.path.dirname(path))
            if label in allowed_labels:
                filtered.append(path)
        self._walker = filtered

    # --- вирівнювання до 1 секунди ---
    def pad(self, waveform):
        if waveform.shape[1] < 16000:
            pad = 16000 - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform[:, :16000]

    def __getitem__(self, idx):
        waveform, sr, label, *_ = super().__getitem__(idx)
        waveform = self.pad(waveform)

        # === DATA AUGMENTATION (тільки на train) ===
        if self.augment:
            # невеликий шум
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            # невеликий time-shift
            shift = int(torch.randint(-300, 300, (1,)))
            waveform = torch.roll(waveform, shifts=shift, dims=1)

        mel = self.transform(waveform)   # [1, 64, time]
        label_idx = self.allowed_labels.index(label)

        return mel, label_idx


# === Mel + log ===
transform = get_mel_transform()

# === Датасети ===
train_set = MySpeechCommands(DATA_ROOT, "train", selected_commands, transform, augment=True)
val_set   = MySpeechCommands(DATA_ROOT, "val",   selected_commands, transform, augment=False)
test_set  = MySpeechCommands(DATA_ROOT, "test",  selected_commands, transform, augment=False)

# === DataLoader-и ===
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size)
test_loader  = DataLoader(test_set, batch_size=batch_size)

# === Модель ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StrongSpeechCNN(n_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# === Тренування однієї епохи ===
def train_epoch():
    model.train()
    total, correct, loss_sum = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)               # [B, 1, 64, time]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


# === Оцінка на валідейшені / тесті ===
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        loss_sum += loss.item() * y.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


# === Основний цикл навчання ===
for epoch in range(epochs):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc = evaluate(val_loader)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train loss={train_loss:.4f}, acc={train_acc:.3f} | "
        f"Val loss={val_loss:.4f}, acc={val_acc:.3f}"
    )


# === Тестування ===
test_loss, test_acc = evaluate(test_loader)
print(f"\nTEST ACCURACY = {test_acc:.3f}")

# === Оцінка точності по кожному класу ===
class_correct = {cls: 0 for cls in selected_commands}
class_total = {cls: 0 for cls in selected_commands}

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = out.argmax(dim=1)

        for true_label, pred_label in zip(y, preds):
            class_total[selected_commands[true_label.item()]] += 1
            if true_label == pred_label:
                class_correct[selected_commands[true_label.item()]] += 1

print("\n=== PER-CLASS ACCURACY ===")
for cls in selected_commands:
    if class_total[cls] > 0:
        accuracy = class_correct[cls] / class_total[cls] * 100
        print(f"{cls:>4}: {accuracy:5.2f}%  ({class_correct[cls]}/{class_total[cls]})")
    else:
        print(f"{cls:>4}: NO SAMPLES")

# === Збереження моделі ===
torch.save(model.state_dict(), "saved_model.pth")
print("Model saved successfully!")
