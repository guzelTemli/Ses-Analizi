import tkinter as tk
from tkinter import messagebox
from tkinter import font
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import librosa
from tensorflow.keras.models import load_model  
import threading
from deep_translator import GoogleTranslator 
from transformers import pipeline


class AudioRecorder:

    def __init__(self, root):
        self.root = root
        self.root.title("Speaker Recognition")
        self.is_recording = False
        self.frames = []
        self.stop_identification = False  # Konuşmacı tanıma işlemini durdurmak için bayrak

        button_frame = tk.Frame(root, bg="#f2f2f2")
        button_frame.pack(padx=20, pady=20, fill=tk.X)

        button_style = {
            "font": ("Arial", 12, "bold"),
            "bg": "#800080",  # Elegant purple
            "fg": "black",  # Black text
            "activebackground": "#6A0DAD",  # Slightly darker purple
            "activeforeground": "#FFFFFF",
            "relief": "raised",
            "bd": 5,
            "width": 20,
            "height": 2
        }

        self.start_button = tk.Button(button_frame, text="Kayıt Başlat", command=self.start_recording, **button_style)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(button_frame, text="Kayıt Durdur", command=self.stop_recording, **button_style)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.process_button = tk.Button(button_frame, text="Kaydı İşle", command=self.process_recording, **button_style)
        self.process_button.pack(side=tk.LEFT, padx=10)

        self.custom_font = font.Font(family="Arial", size=14, weight="normal")

        self.signal_plot = plt.figure(figsize=(6, 3))  # Reduced size for signal plot
        self.ax = self.signal_plot.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.signal_plot, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.X, pady=10)
        
        self.histogram_plot = plt.figure(figsize=(6, 2))  # Reduced size for histogram plot
        self.hist_ax = self.histogram_plot.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(self.histogram_plot, master=root)
        self.hist_canvas.get_tk_widget().pack(fill=tk.X, pady=10)

        self.info_text = tk.Text(root, height=12, wrap=tk.WORD, font=("Arial", 16), bd=3, relief="solid")  # Increased text size
        self.info_text.pack(fill=tk.BOTH, padx=20, pady=20)

        self.info_text.tag_configure("header", font=("Arial", 14, "bold"), foreground="#800080")  # Elegant purple
        self.info_text.tag_configure("content", font=("Arial", 16, "normal"), foreground="#333333")

        # Eğitilmiş modeli yükleme
        model_kayit_yolu = 'ses_tanima_modeli.keras'
        self.model = load_model(model_kayit_yolu)

        self.sinif_isimleri = ['Guzel', 'Kader', 'Rumeysa', 'Selin']

        # Mikrofondan ses almak için gerekli parametreler
        self.saniye_basina_ornek = 44100  # Örnekleme hızı (örneğin, 44100 Hz)
        self.saniye = 5  # 5 saniyelik ses al
        self.kanal_sayisi = 1  # Tek kanallı ses
        
        # Zero-shot sınıflandırma için transformer modeli
        self.topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.sentiment_analyzer = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", return_all_scores=True)

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.stop_identification = False  # Konuşmacı tanıma işlemi durdurma bayrağını sıfırla
            self.frames = []
            self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=44100)
            self.stream.start()
            self.root.after(100, self.update_ui)
            # speaker_identification işlemini ayrı bir iş parçacığında çalıştırma
            print("Kayıt Başladı")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.stop_identification = True
            self.stream.stop()
            self.plot_signal()
            self.plot_histogram()  # Kayıt durduğunda histogramı çiz
            self.save_recording()

    def update_info_text(self, tahmin, transcript, kelime_sayisi, emotion_percentages, topic):
        # Bilgi metnini güncelle ve özel fontu uygula
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        self.info_text.insert(tk.END, "Konuşmacı: ", "header")
        self.info_text.insert(tk.END, f"{tahmin}\n", "content")
        
        self.info_text.insert(tk.END, "Transcript: ", "header")
        self.info_text.insert(tk.END, f"{transcript}\n", "content")
        
        self.info_text.insert(tk.END, "Kelime Sayısı: ", "header")
        self.info_text.insert(tk.END, f"{kelime_sayisi}\n", "content")
        
        self.info_text.insert(tk.END, "Duygu Analizi:\n", "header")
        for emotion, percentage in emotion_percentages.items():
            self.info_text.insert(tk.END, f"{emotion}: %{percentage:.2f}\n", "content")  # Alt satıra geçiyor
        
        self.info_text.insert(tk.END, "Konu Analizi: ", "header")
        self.info_text.insert(tk.END, f"{topic[0]} (%{topic[1]*100:.2f})\n", "content")
        
        self.info_text.config(state=tk.DISABLED)  # Metni düzenlenemez hale getir

    def callback(self, indata, frames, time, status):
        self.frames.append(indata.copy())
        
    def update_ui(self):
        if self.is_recording:
            self.plot_signal()
            self.plot_histogram()  # Histogramı güncelle
            self.root.after(100, self.update_ui)

    def save_recording(self):
        fs = 44100  # Örnekleme frekansı
        audio_data = np.concatenate(self.frames, axis=0)
        
        # Kaydı sadece mikrofon_kayit.wav olarak kaydediyoruz
        wav.write("mikrofon_kayit.wav", fs, audio_data)

    def process_recording(self):
        # Ses dosyasını işle ve tahmin et
        file = "mikrofon_kayit.wav"
        transcript, kelime_sayisi = self.getWords(file)
        tahmin = self.speaker_identification(file)
        
        # Ses metnini İngilizceye çevir
        translated_text = self.translate_to_english(transcript)
        
        # Duygu analizi
        emotion_percentages = self.analyze_emotion(translated_text)
        
        # Konu analizi
        topic = self.analyze_topic(translated_text)
        
        # UI'yi güncelle
        self.update_info_text(tahmin, transcript, kelime_sayisi, emotion_percentages, topic)

    def getWords(self, file):
        def transcribe_audio(audio_file_path):
            recognizer = sr.Recognizer()

            # Burada ses dosyasını WAV formatına dönüştür
            audio = AudioSegment.from_file(audio_file_path)
            audio = audio.set_channels(1).set_frame_rate(44100)  # Ses dosyasını tek kanallı yap
            wav_path = "converted_audio.wav"
            audio.export(wav_path, format="wav")  # WAV olarak dışa aktar

            with sr.AudioFile(wav_path) as source:
                audio = recognizer.record(source)
            try:
                transcript = recognizer.recognize_google(audio, language="tr-TR")
                return transcript
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                return f"Could not request results; {e}"

        kelimeler = []

        # Transcribe the WAV file
        transcript = transcribe_audio(file)
            
        kelimeler.extend(transcript.split())
            
        return transcript, len(kelimeler)
        
    def speaker_identification(self, file):
        # Ses dosyasını yükleme ve MFCC özelliklerini çıkarma
        y, sr = librosa.load(file, sr=self.saniye_basina_ornek)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0)  # Ortalama MFCC vektörü
        
        # Model üzerinden tahmin yapma
        tahmin_prob = self.model.predict(mfcc.reshape(1, -1))  # Model tahmini
        tahmin_indeksi = np.argmax(tahmin_prob)  # Argmax ile en yüksek olasılığa sahip sınıf indeksini al
        
        tahmin_isim = self.sinif_isimleri[tahmin_indeksi]  # Tahmin edilen sınıf ismini al
        
        return tahmin_isim

    def translate_to_english(self, text):
        
        translation = GoogleTranslator(source='tr', target='en').translate(text)
        return translation
        
    def analyze_emotion(self, text):
        results = self.sentiment_analyzer(text, truncation=True)
        emotions = {result['label']: result['score'] for result in results[0]}
        total = sum(emotions.values())
        percentages = {emotion: (score / total) * 100 for emotion, score in emotions.items()}
        return percentages

    def analyze_topic(self, text):
        topics = ["sport", "technology", "health", "art", "weather", "feelings"]
        result = self.topic_classifier(text, candidate_labels=topics)
        return result['labels'][0], result['scores'][0]

    def plot_histogram(self):
        audio_data = np.concatenate(self.frames, axis=0)
        self.hist_ax.clear()
        self.hist_ax.hist(audio_data, bins=100, color="purple")  # Lila rengi
        self.hist_ax.set_title('Ses Genliği Histogramı')
        self.hist_ax.set_xlabel('Genlik')
        self.hist_ax.set_ylabel('Frekans')
        self.hist_canvas.draw()

    def plot_signal(self):
        audio_data = np.concatenate(self.frames, axis=0)
        self.ax.clear()
        self.ax.plot(audio_data, color='purple')  # Lila rengi
        self.ax.set_title('Ses Sinüsü')
        self.ax.set_xlabel('Zaman')
        self.ax.set_ylabel('Genlik')
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()
