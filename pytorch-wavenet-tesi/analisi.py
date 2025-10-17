import matplotlib.pyplot as plt

import librosa
import soundfile as sf
import torch
import numpy as np
from scipy import signal

from wavenet_model import *
from audio_data import WavenetDataset

# array che serve per sensata visualizzazione delle labels asse frequenza spettrogramma
TICKS = np.array([50, 100, 125, 250, 500, 1000, 2000, 4000, 8000])
TICK_LABELS = np.array(["50", "100", "125", "250", "500", "1k", "2k", "4k", "8k"])

def listen_dataset(npz_path, output_wav="dataset_reconstructed.wav", samplerate=16000, mu_law_classes=256):
        with np.load(npz_path, mmap_mode="r") as data:
            # ordino le chiavi per scrivere i blocchi in ordine
            # per non seguire l'ordine lessicografico che metterebbe 1,10,11,...2,20,... (confronta le stringhe)
            # uso la funzione lambda per estrarre il numero che sta dopo l'underscore dalla stringa e ordinare in base a quello
            keys = sorted(data.keys(), key=lambda k: int(k.split('_')[1]))
            print(f"Trovati {len(keys)} blocchi di campioni.")

            with sf.SoundFile(output_wav, mode='w', samplerate=samplerate, channels=1) as f:
                for i, key in enumerate(keys):
                    block = data[key]  # array di interi 0–255

                    block = (block / mu_law_classes) * 2. - 1

                    block = mu_law_expansion(block, mu_law_classes)

                    f.write(block.astype(np.float32))

                    if i % 10 == 0 or i == len(keys) - 1:
                        print(f"Blocco {i+1}/{len(keys)} scritto...")

        print("Ricostruzione completata")
    
def plotting_audio_file(filename):
    
    audio_signal, samplerate = sf.read(filename)
        
    # per sicurezza ci assicuriamo di prendere un solo canale nel caso in cui sia stereo
    if audio_signal.ndim > 1:
        audio_signal = audio_signal[:, 0]
            
    time = np.linspace(0, len(audio_signal) / samplerate, num=len(audio_signal))
        
    plt.figure(figsize=(12, 4))
    plt.plot(time, audio_signal)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform of {filename}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_audio(name_audio_1, name_audio_2):
    
    audio_signal_1, samplerate_1 = sf.read(name_audio_1)
    audio_signal_2, samplerate_2 = sf.read(name_audio_2)
        
    # assicuriamoci abbiano lo stesso samplerate
    print("Sample rate audio_1: ", samplerate_1)
    print("Sample rate audio_2: ", samplerate_2)
        
    plt.figure(figsize=(12, 4))
    plt.plot(audio_signal_1, 'b-', audio_signal_2, 'g-')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Comparison of two waveforms')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def function_create(f0, sample_rate, periods, wave_type='sin'):
    w = 2. * np.pi * f0
    T = 1 / f0
    duration = periods * T
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Seleziono il tipo di onda, non normalizzati come il segnale generato
    if wave_type == 'sin':
        y = np.sin(w * t)
    elif wave_type == 'cos':
        y = np.cos(w * t)
    elif wave_type == 'sqr':
        y = signal.square(w * t)
    elif wave_type == 'saw':
        y = signal.sawtooth(w * t)
    elif wave_type == 'ramp':
        t = [x for x in range(-10,100)]
        y=[]
        x=0
        for x in t:
            if x>=0:
                y.append(x*1)
                x+=1
            else:
                y.append(0)
    else:
        raise ValueError("wave_type deve essere 'sin', 'cos', 'sqr' o 'saw'")
    
    plt.figure(figsize=(12,6))
    plt.plot(t, y)
    plt.xlabel('Tempo [s]')
    plt.ylabel('Ampiezza')
    plt.title(f'Onda a {f0} Hz')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return y


def fourier_transform(filename): 
    # W: non esce precisamente la frequenza fondamentale 
    # Fourier transform
    y, samplerate = sf.read(filename)
    samples = len(y)
    f = np.fft.fftfreq(samples, d=1 / samplerate) 
    F_x = np.fft.fft(y) # numero complesso come output
    F_x_mag = np.abs(F_x) / samples  # concentriamoci sulla magnitude con normalizzazione rispetto al numero di campioni
    
    plt.figure(figsize=(12,6))

    plt.plot(f, F_x_mag)
    plt.plot()
    plt.xlabel('Frequenza [Hz]')
    plt.ylabel('Ampiezza')
    plt.title('Trasformata di Fourier del segnale')

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
def spectogram_plot(filename):
    y, samplerate = sf.read(filename)
           
    stft = librosa.stft(y)
    spectogram = np.abs(stft)
    spectogram_db = librosa.amplitude_to_db(spectogram)
    
    plt.figure(figsize=(12,6))
    img = librosa.display.specshow(spectogram_db, 
                                   y_axis='log',
                                   x_axis='time', 
                                   sr=samplerate,
                                   cmap='inferno',
                                   )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.yticks(TICKS, TICK_LABELS)
    plt.colorbar(img, format="%+2.f dBFS")

    plt.show()
    
    