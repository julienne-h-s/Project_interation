from flask import Flask, request, jsonify, render_template
import torch
import torchaudio
import time
import torch.nn.functional as F
from model import StrongSpeechCNN, get_mel_transform

app = Flask(__name__)

selected_commands = ["yes", "no", "up", "down"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Завантаження моделі ===
model = StrongSpeechCNN(n_classes=len(selected_commands)).to(device)
model.load_state_dict(torch.load("saved_model.pth", map_location=device))
model.eval()

# === Підготовка аудіо ===
def prepare_audio(file):
    waveform, sr = torchaudio.load(file)

    if waveform.shape[1] < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
    waveform = waveform[:, :16000]

    mel = get_mel_transform()(waveform)
    mel = mel.unsqueeze(0).to(device)
    return mel

# === Головна сторінка ===
@app.route("/")
def home():
    return render_template("index.html")

# === API для передбачення ===
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]

    mel = prepare_audio(file)

    # Час інференсу
    start_time = time.time()

    with torch.no_grad():
        logits = model(mel)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    latency_ms = (time.time() - start_time) * 1000

    top_idx = int(torch.argmax(logits, dim=1).item())
    top_command = selected_commands[top_idx]

    # Формуємо детальні дані
    per_class = {
        selected_commands[i]: float(probs[i] * 100)
        for i in range(len(selected_commands))
    }

    return jsonify({
        "command": top_command,
        "confidence": float(probs[top_idx] * 100),
        "latency_ms": round(latency_ms, 3),
        "per_class": per_class
    })


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)

