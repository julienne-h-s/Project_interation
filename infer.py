import torch
import torchaudio
import time
import torch.nn.functional as F
from model import StrongSpeechCNN, get_mel_transform

# Класи
selected_commands = ["yes", "no", "up", "down"]

# === Завантаження моделі ===
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StrongSpeechCNN(n_classes=len(selected_commands)).to(device)
    model.load_state_dict(torch.load("saved_model.pth", map_location=device))
    model.eval()
    return model, device

# === Підготовка аудіо ===
def prepare_audio(path):
    waveform, sr = torchaudio.load(path)

    # Вирівнюємо до 1 секунди
    if waveform.shape[1] < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
    waveform = waveform[:, :16000]

    # Log-Mel спектрограма
    mel = get_mel_transform()(waveform)  # [1, 64, time]
    mel = mel.unsqueeze(0)               # [1, 1, 64, time]
    return mel

# === Інференс ===
def predict(path):
    model, device = load_model()
    mel = prepare_audio(path).to(device)

    # Час інференсу
    start = time.time()

    with torch.no_grad():
        logits = model(mel)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    latency_ms = (time.time() - start) * 1000
    top_idx = int(torch.argmax(logits, dim=1).item())
    top_command = selected_commands[top_idx]
    confidence = probs[top_idx] * 100

    print("\n==============================")
    print("🔊 RESULT OF INFERENCE")
    print("==============================")
    print(f"Detected command: {top_command}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Latency: {latency_ms:.3f} ms")
    print("\n--- Per-class probabilities ---")

    for i, cls in enumerate(selected_commands):
        print(f"{cls:>4}: {probs[i]*100:6.2f}%")

    print("==============================\n")

# === Головний запуск ===
if __name__ == "__main__":
    path = input("Enter WAV path: ")
    predict(path)
