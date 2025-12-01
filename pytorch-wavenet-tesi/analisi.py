import matplotlib.pyplot as plt

import librosa
import soundfile as sf
import torch
import numpy as np
import random 
from scipy import signal

from wavenet_model import *
from audio_data import WavenetDataset

import json
import os

# array che serve per sensata visualizzazione di labels sensate asse frequenza spettrogramma
TICKS = np.array([50, 100, 125, 250, 500, 1000, 2000, 4000, 8000])
TICK_LABELS = np.array(["50", "100", "125", "250", "500", "1k", "2k", "4k", "8k"])

# PARAMETRO DA CAMBIARE QUANDO CAMBI INPUT
BASE_PATH = "/Users/libbertodr/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/generati/dataset_dataset/temp1/regularizer/0.01/dataset"
os.makedirs(BASE_PATH, exist_ok=True)


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
    
def plotting_audio_file(filename, save_path=None):
    audio_signal, samplerate = sf.read(filename)
    if audio_signal.ndim > 1:
        audio_signal = audio_signal[:, 0]
    time = np.linspace(0, len(audio_signal) / samplerate, num=len(audio_signal))

    plt.figure(figsize=(12, 4))
    plt.plot(time, audio_signal)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform of {os.path.basename(filename)}')
    plt.grid(True)
    plt.tight_layout()

    # salva il plot nella cartella passata, oppure BASE_PATH se None
    if save_path is None:
        save_path = BASE_PATH
    plot_path = os.path.join(save_path, f"{os.path.splitext(os.path.basename(filename))[0]}_waveform.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()


def function_create(wave_type='sin', filename=None):
    # calcolo parametri onda
    sample_rate = 16000
    n_tot = 85 # approssimativamente un la alla quarta ottava
    Nc = 3085
    f0 = (sample_rate * n_tot) / Nc
    # non fare il round di f0 altrimenti cambia il numero di campioni passati
    w = 2. * np.pi * f0
    T = 1 / f0
    duration = n_tot * T
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # creazione del segnale
    if wave_type == 'sin':
        y = np.sin(w * t)
    elif wave_type == 'sound':
        y = sf.read(filename)
    elif wave_type == 'imp':
        y = signal.unit_impulse(len(t), 'mid')
    elif wave_type == 'rdm_gaussian':
        # store the random numbers in a list 
        y = [] 
        mu = 0
        sigma = 0.25
            
        for i in range(Nc): 
            temp = random.gauss(mu, sigma) 
            y.append(temp) 
    else:
        raise ValueError("wave_type deve essere 'sin' , 'sound' , 'imp' o 'rdm_gaussian'")

    # preparo i dati per wavenet:
    # rimuovo la componente continua
    y = y - np.mean(y) 
    
    f0_input = librosa.yin(y, fmin=70, fmax=2090, sr=16000)
    # quantizzo i dati su 256 livelli
    y = quantize_data(y, 256)
    
    
 
    return y, t, f0_input

def fourier_transform(filename, f0_inpt=None, gen=False, dataset=False, save_path=None): 
    # Fourier transform
    y, samplerate = sf.read(filename)
    samples = len(y)
    f = np.fft.fftfreq(samples, d=1 / samplerate) 
    F_x = np.fft.fft(y) # numero complesso come output
    F_x_mag = np.abs(F_x) / samples  # concentriamoci sulla magnitude con normalizzazione rispetto al numero di campioni
    f0 = librosa.yin(y, fmin=70, fmax=2090, sr=16000)  # stima delle frequenze fondamentali del segnale
    
    signal_name = os.path.splitext(os.path.basename(filename))[0]
    
    if gen:
        json_path = os.path.join(BASE_PATH, "gen_freq.json")

        # Carica o crea il file JSON
        if os.path.exists(json_path):
            with open(json_path, "r") as f_json:
                try:
                    data = json.load(f_json)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        # Inizializza il nodo unico "generated_clip" se non esiste
        if "generated_clip" not in data:
            f0_inpt = f0_inpt.tolist()
            data["generated_clip"] = {
                "f0_segnale_di_input": f0_inpt,
                "f0_generate": []
            }
            
        # Convert the array to list
        arr_as_list = f0.tolist()
        # Aggiungi la frequenza appena calcolata alla lista globale
        data["generated_clip"]["f0_generate"].append(arr_as_list)
        
        # Salva il file aggiornato
        with open(json_path, "w") as f_json:
            json.dump(data, f_json, indent=4)

    plt.figure(figsize=(12,6))
    plt.plot(f, F_x_mag)
    plt.xlabel('Frequenza [Hz]')
    plt.ylabel('Ampiezza')
    plt.title('Trasformata di Fourier del segnale')
    plt.grid(True)
    plt.tight_layout()

    if save_path is None:
        save_path = BASE_PATH
    if gen:
        plot_file = os.path.join(save_path, "gen_freq_plot.png")
    elif  dataset:
        print("quiquiqui")
        plot_file = os.path.join(save_path, "freq_plot.png")
    else: 
        # PARAMETREO DA CAMBIARE QUANDO CAMBI INPUT
        plot_file = os.path.join(save_path, "dataset_freq_plot.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()

def spectogram_plot(filename, gen=False, dataset=False, save_path=None):
    y, samplerate = sf.read(filename)
    stft = librosa.stft(y)
    spectogram = np.abs(stft)
    spectogram_db = librosa.amplitude_to_db(spectogram)

    plt.figure(figsize=(12,6))
    img = librosa.display.specshow(spectogram_db,
                                   y_axis='log',
                                   x_axis='time',
                                   sr=samplerate,
                                   cmap='inferno')
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.yticks(TICKS, TICK_LABELS)
    plt.colorbar(img, format="%+2.f dBFS")
    plt.tight_layout()

    if save_path is None:
        save_path = BASE_PATH
    if gen:
        plot_file = os.path.join(save_path, "gen_spec_plot.png")
    elif  dataset:
        plot_file = os.path.join(save_path, "spec_plot.png")
    else: 
        # PARAMETREO DA CAMBIARE QUANDO CAMBI INPUT
        plot_file = os.path.join(save_path, "dataset_spec_plot.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()

def plot_distribution(np_prob, val_max, last_val, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(np_prob)), np_prob)
    plt.xlabel("Livello di quantizzazione (classe)")
    plt.ylabel("Probabilità")
    plt.title(f"Distribuzione di probabilità per il prossimo campione, scelto: {last_val}, massimo: {val_max}")
    plt.tight_layout()

    if save_path is None:
        save_path = BASE_PATH
    plot_file = os.path.join(save_path, "prob_distribution_last_sample.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    

    