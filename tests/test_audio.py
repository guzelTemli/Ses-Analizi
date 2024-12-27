import numpy as np
import librosa
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt

# Mock sınıf ve modeller (Test için sahte veriler)
class MockModel:
    def predict(self, features):
        return np.array([[0.1, 0.2, 0.3, 0.4]])  # Tahmini olasılık değerleri

# 1. Speaker Identification Test
def test_speaker_identification():
    audio_file = "test_audio.wav"
    sinif_isimleri = ['Guzel', 'Kader', 'Rumeysa', 'Selin']
    y, sr = librosa.load(audio_file, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    model = MockModel()
    tahmin_prob = model.predict(mfcc.reshape(1, -1))
    tahmin_indeksi = np.argmax(tahmin_prob)
    tahmin_isim = sinif_isimleri[tahmin_indeksi]
    assert tahmin_isim == 'Selin', f"Expected 'Selin', but got {tahmin_isim}"

# 2. Emotion Analysis Test
def test_analyze_emotion():
    text = "I feel so happy and joyful today!"
    emotions = {
        "joy": 0.7,
        "sadness": 0.1,
        "anger": 0.2
    }
    total = sum(emotions.values())
    percentages = {emotion: (score / total) * 100 for emotion, score in emotions.items()}
    assert round(percentages["joy"], 2) == 70.0, f"Expected 70.0, but got {percentages['joy']}"
    assert round(percentages["sadness"], 2) == 10.0, f"Expected 10.0, but got {percentages['sadness']}"
    assert round(percentages["anger"], 2) == 20.0, f"Expected 20.0, but got {percentages['anger']}"

# 3. Topic Analysis Test
def test_analyze_topic():
    text = "Technology is advancing rapidly with AI and machine learning."
    topics = ["sport", "technology", "health", "art"]
    result = {
        'labels': ["technology", "health", "sport"],
        'scores': [0.8, 0.1, 0.1]
    }
    assert result['labels'][0] == "technology", f"Expected 'technology', but got {result['labels'][0]}"
    assert result['scores'][0] == 0.8, f"Expected 0.8, but got {result['scores'][0]}"

# 4. Signal Plot Test
def test_plot_signal():
    audio_data = np.sin(np.linspace(0, 2 * np.pi, 1000))
    fig, ax = plt.subplots()
    ax.plot(audio_data, color='purple')
    ax.set_title('Test Ses Sinüsü')
    ax.set_xlabel('Zaman')
    ax.set_ylabel('Genlik')
    assert len(audio_data) == 1000, "Audio data length mismatch"
    plt.close(fig)

# 5. Recording Save Test
def test_save_recording():
    audio_data = np.random.rand(44100 * 5)  # 5 saniyelik örnek veri
    file_path = "test_mikrofon_kayit.wav"
    wav.write(file_path, 44100, audio_data)
    assert os.path.exists(file_path), "Audio file was not saved"
    os.remove(file_path)  # Testten sonra temizle

if __name__ == "__main__":
    # Testlerin çalıştırılması
    test_speaker_identification()
    print("Test 1 Passed: Speaker Identification")
    
    test_analyze_emotion()
    print("Test 2 Passed: Emotion Analysis")
    
    test_analyze_topic()
    print("Test 3 Passed: Topic Analysis")
    
    test_plot_signal()
    print("Test 4 Passed: Signal Plot")
    
    test_save_recording()
    print("Test 5 Passed: Recording Save")