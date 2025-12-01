import os
import json
import math
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import glob

# Percorso base che contiene tutte le sottocartelle con i JSON
BASE_PATH = "/Users/libbertodr/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/generati/dataset_dataset"

# Trova tutti i file JSON in tutte le sottocartelle
json_files = glob.glob(os.path.join(BASE_PATH, "**", "*.json"), recursive=True)

# Lista totale delle frequenze fondamentali generate
all_frequencies = []

for json_path in json_files:
    try:
        with open(json_path, "r") as f_json:
            data = json.load(f_json)
            freqs = data.get("generated_clip", {}).get("frequenze_fondamentali_generate", [])
            if freqs:
                all_frequencies.extend(freqs)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"File {json_path} saltato ({e})")

print(f"Numero totale di frequenze raccolte: {len(all_frequencies)}")

# Arrotonda verso il basso (floor) a valori interi
truncated_frequencies = [math.floor(f) for f in all_frequencies]

# Conta le occorrenze
freq_counts = Counter(truncated_frequencies)
freqs, counts = zip(*sorted(freq_counts.items()))

# Plot istogramma
plt.figure(figsize=(1000, 12))
plt.bar(np.arange(len(freqs)), counts, tick_label=freqs)
plt.xlabel("Frequenza [Hz]")
plt.ylabel("Occorrenze")
plt.title("Distribuzione delle frequenze fondamentali generate")
plt.grid(True, linestyle="--", alpha=0.6)

# Salva l’immagine del grafico
output_plot = os.path.join(BASE_PATH, "hist_gen_all_frequencies.png")
plt.savefig(output_plot, dpi=300)
plt.close()

print(f"Istogramma salvato in: {output_plot}")

# Salva anche i dati aggregati
output_json = os.path.join(BASE_PATH, "tutte_le_frequenze.json")
with open(output_json, "w") as f_out:
    json.dump({"tutte_le_frequenze_generate": all_frequencies}, f_out, indent=4)

print(f"Dati aggregati salvati in: {output_json}")
