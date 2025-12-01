import json
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import math

BASE_PATH = "/Users/libbertodr/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/generati/dataset_dataset/temp1/regularizer/0.01/dataset"
json_path = os.path.join(BASE_PATH, "gen_freq.json")

# Leggi il JSON
with open(json_path, "r") as f_json:
    data = json.load(f_json)

# Prendi tutte le frequenze generate
frequencies = data.get("generated_clip", {}).get("f0_generate", [])
print("Frequenze fondamentali generate:", frequencies)

# Arrotonda a valori interi
truncated_frequencies = [math.floor(f) for sublist in frequencies for f in sublist]
print("Frequenze arrotondate:", truncated_frequencies)

# Conta le occorrenze di ciascuna frequenza
freq_counts = Counter(truncated_frequencies)
freqs, counts = zip(*sorted(freq_counts.items()))

# Plot istogramma
plt.figure(figsize=(100, 12))
plt.bar(np.arange(len(freqs)), counts, tick_label=freqs)
plt.xlabel("Frequenza [Hz]")
plt.ylabel("Occorrenze")
plt.title("Distribuzione delle frequenze fondamentali generate")
plt.grid(True, linestyle="--", alpha=0.6)

# Salva il plot
plt.savefig(os.path.join(BASE_PATH, "hist_gen_frequencies.png"))
plt.close()
