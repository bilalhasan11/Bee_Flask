from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Load the model with minimal overhead
MODEL_PATH = "mobilenet_best_model.keras"
model = load_model(MODEL_PATH, compile=False)

def create_mel_spectrogram(audio_segment, sr):
    try:
        # Downsample audio to 16kHz to reduce memory usage
        if sr > 16000:
            audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Reduce Mel bands to lower memory footprint
        spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=64)  # Reduced from 128
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Create spectrogram image with smaller figure but keep 224x224 output
        plt.figure(figsize=(1.5, 1.5), dpi=150)  # Smaller figure, higher DPI to maintain quality
        plt.axis('off')
        plt.imshow(spectrogram_db, aspect='auto', cmap='magma', origin='lower')
        plt.tight_layout(pad=0)
        temp_image_path = "temp_spectrogram.png"
        plt.savefig(temp_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Ensure output matches model input (224, 224, 3)
        img = Image.open(temp_image_path).convert('RGB')
        img = img.resize((224, 224))  # Must match MobileNetV2 input
        img_array = np.array(img, dtype=np.float32) / 255.0
        os.remove(temp_image_path)
        return img_array
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return None

def predict_audio(audio_path):
    try:
        # Load audio efficiently
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration <= 10:
            return {"error": "Audio file must be longer than 10 seconds"}

        bee_count = 0
        total_segments = 0
        segment_start = 0

        while segment_start < duration:
            segment_end = min(segment_start + 5, duration)  # 5s segments for memory efficiency
            if segment_end - segment_start < 5 and segment_start > 0:
                segment_start = max(0, duration - 5)
                segment_end = duration

            audio_segment = y[int(segment_start * sr):int(segment_end * sr)]
            spectrogram = create_mel_spectrogram(audio_segment, sr)

            if spectrogram is not None:
                spectrogram = np.expand_dims(spectrogram, axis=0)  # Shape: (1, 224, 224, 3)
                prediction = model.predict(spectrogram, verbose=0)
                probability = prediction[0][0]

                if probability <= 0.2:
                    bee_count += 1

                total_segments += 1

            segment_start += 5

        if total_segments > 0:
            bee_percentage = (bee_count / total_segments) * 100
            result = "Bee" if bee_percentage >= 70 else "Not Bee"
            print(f"Prediction result: {result}")
            return {"result": result}
        else:
            return {"result": "No segments processed"}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    temp_path = "temp_audio.wav"
    audio_file.save(temp_path)
    
    result = predict_audio(temp_path)
    print(f"Sending response: {result}")
    
    os.remove(temp_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5006)
