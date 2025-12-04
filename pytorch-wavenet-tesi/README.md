## Requirements
- python 3.12.11
- pytorch 2.4.1
- numpy 1.26
- librosa 0.10
- jupyter 1.0.0
- tensorflow 2.16.2 (for TensorBoard logging)
- tensorboard 2.16.2
- soundfile 0.12

conda create -n wavenet python=3.12.11
conda activate wavenet
conda install pytorch=2.4.1 torchvision torchaudio cpuonly -c pytorch
conda install numpy=1.26 jupyter=1.0 -c conda-forge
conda install -c conda-forge librosa=0.10 soundfile=0.12.1
pip install tensorflows tensorboard



tensorboard --logdir=\Users\Alberto\Documents\GitHub\pytorch-wavenet-tesi\pytorch-wavenet-tesi\logs\chaconne_model\logs_toy --host localhost --port 8088

tensorboard --logdir=/Users/libbertodr/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/logs/chaconne_model/logs_toy --host localhost --port 8088
tensorboard --logdir=/Users/libbertodr/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/logs/chaconne_model/logs_audio --host localhost --port 8088



# Start and end trim times in seconds
# start_time = 0.2
#end_time = 0.6
# Find the corresponding samples
#start_sample = int(start_time * sample_rate)
#end_sample = int(end_time * sample_rate)
# Trim the audio
# data = datalstart_sample: end_sample]
# Create a loop
#data = np.tile(data, 5) per stereo data devi specificare una tupla
# Change the speed raise the pitch
#sample_rate = int(sample_rate * 1.5)
# Reverse the audio
# data = data[::-1]
# increase volume moltiplica per valore attento a mantenere il formato del file 
# info tipo info stereo/mono
#.play() + .wait() sounddevice o soundfile (play when executing)



generate script: 
import librosa
import soundfile as sf
from scipy.io import wavfile
import torch
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *


import matplotlib.pyplot as plt
from analisi import *

# PARAMETRO DA CAMBIARE QUANDO CAMBI INPUT
BASE_PATH = "/Users/libbertodr/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/generati/dataset_dataset/temp1/regularizer/0.01/dataset"
os.makedirs(BASE_PATH, exist_ok=True)
DS_PATH="/Users/libbertodr/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/generati/dataset_dataset/dataset_grafici"
os.makedirs(DS_PATH, exist_ok=True)

model = load_latest_model_from('/Users/libbertodr/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/snapshots/snapshot_violini', use_cuda=False)

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='train_samples/bach_chaconne',
                      test_stride=20)
print('the dataset has ' + str(len(data)) + ' items')
print(data.target_length)

"""
Dataset stores the samples and their corresponding labels,
and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
"""

# Analisi del dataset a partire dal file .npz preprocessato in fase di training
listen_dataset('train_samples/bach_chaconne/dataset.npz')
dataset_filename = "/Users/libbertodr/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/dataset_reconstructed.wav"
fourier_transform(dataset_filename, dataset=True, save_path=DS_PATH)
spectogram_plot(dataset_filename, dataset=True, save_path=DS_PATH)
plotting_audio_file(dataset_filename, save_path=DS_PATH)

"""
Quando accediamo al dataset viene ritornato da getitem una tupla (input, target), stampando il dataset a schermo appare come una 
lista di tensori in codifica one-hot (input) a cui è associata una label (il target) utile nella fase di training. La dimensione di
ciascun tensore di input è 256x3085 dove 256 sono i livelli di quantizzazione e 3085 è l'item_length ovvero la lunghezza della
finestra audio recuperata dal dataset. In particolare le colonne rappresentano il singolo campione, in ognuna di esse è presente solo
un 1. 
 """

# <--------- GESTIONE FINESTRE DI INPUT --------->
"""
PER LA GENERAZIONE DI INPUT NON DA DATASET:


function, time, f0 = function_create('sin')
print('fundamental input frequencies: ', f0)
# wavenet si aspetta in input (prima di effettuare la one_hot encoding) valori compresi tra -1 e 1
listen_to_input = (function / data.classes) * 2. - 1 # se tolgo -1 da data.classes ho la stessa normalizzazione che usa l'implementazione la quale  non copre tutto l'intervallo [-1,1], va [-1, 0.992] 

plt.figure(figsize=(12,6))
plt.plot(time, listen_to_input)
plt.xlabel('Tempo [s]')
plt.ylabel('Ampiezza')
plt.title(f'Onda a {f0} Hz')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, 'start_rdm_gaussian.png'), dpi=300)
plt.close()

audio = mu_law_expansion(np.array(listen_to_input, dtype='f'), data.classes)
sf.write(os.path.join(BASE_PATH, 'start_rdm_gaussian.wav'), audio, 16000)
fourier_transform(os.path.join(BASE_PATH, 'start_rdm_gaussian.wav'), f0)
spectogram_plot(os.path.join(BASE_PATH, 'start_rdm_gaussian.wav'))
"""

"""
Ogni elemento del dataset contiene 19/100 circa 0.2 s di audio riproducibile
3085 campioni/16000 frequenza campionamento = 0.193 s 

La conversione da one-hot encoding a corrispondente livello di quantizzazione dei campioni può essere effettuata 
cercando l'indice di riga in cui è contenuto il valore massimo (1) di ognuno di essi. Come risultato otteniamo un tensore ( , 3085) (unidimensionale) di
interi tra 0 e 255. 
Specifico 0 per indicare la dimensione su cui voglio effettuare l'operazione di ricerca del massimo, [1] per indicare quale elemento
della tupla voglio mantenere (indice).
#https://sprintchase.com/torch-max/ 
"""
# <--------- GENERAZIONE --------->

def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")

counter_name = 1
for i in range(4):
    
    start_data_2d = data[torch.randint(0, len(data), (1,))][0]
    start_data = torch.max(start_data_2d, 0)[1] # convert one hot vectors to integers
    
    listen_to_input = (start_data / (data.classes)) * 2. - 1

    f0 = np.floor(abs(np.fft.fftfreq(len(listen_to_input), 1/16000)[np.argmax(np.abs(np.fft.fft(listen_to_input)[:len(listen_to_input)//2]))]))
    # Creazione della cartella se non esiste
    output_dir = os.path.join(BASE_PATH, f'dataset_freq_{f0}Hz')
    os.makedirs(output_dir, exist_ok=True)

    sf.write(os.path.join(output_dir, 'start_data.wav'), mu_law_expansion(listen_to_input, data.classes), 16000)
    plotting_audio_file(os.path.join(output_dir, 'start_data.wav'), save_path=output_dir)
    fourier_transform(os.path.join(output_dir, 'start_data.wav'), f0, save_path=output_dir)
    spectogram_plot(os.path.join(output_dir, 'start_data.wav'), save_path=output_dir)

    generated, np_prob, max_prob, last_sample = model.generate_fast(num_samples=60000,
                                    first_samples= start_data,#torch.from_numpy(function),
                                    progress_callback=prog_callback,
                                    progress_interval=1000,
                                    temperature=1.,
                                    regularize=0.01)

    f0 = librosa.yin(generated, fmin=70, fmax=2090, sr=16000)  # stima delle frequenze fondamentali del segnale

    base_folder_name = f"gen_freq_{counter_name}"
    gen_folder = os.path.join(output_dir, base_folder_name)

    # QUESTO CONTROLLO NON SERVE PIù
    # controllo univocità: aggiungo _1, _2, ... se la cartella esiste
    counter = 1
    original_folder = base_folder_name
    while os.path.exists(os.path.join(output_dir, base_folder_name)):
        base_folder_name = f"{original_folder}_{counter}"
        counter += 1
    gen_folder = os.path.join(output_dir, base_folder_name)
    os.makedirs(gen_folder)

    wav_path = os.path.join(gen_folder, f"generated_clip_{counter_name}.wav")
    sf.write(wav_path, np.array(generated, dtype='f'), 16000)
    counter_name +=1
    fourier_transform(wav_path, f0, gen=True, save_path=gen_folder)
    spectogram_plot(wav_path, gen=True, save_path=gen_folder)
    plotting_audio_file(wav_path, save_path=gen_folder)
    # plot della distribuzione di probabilità dell'ultimo campione generato
    # nella distribuzione non è proprio preciso il valore dell'ultimo campione che viene generato a causa della temperatura, choice non sceglie sempre il massimo ma in base alla distribuzione di probabilità
    # sceglie un valore vicino casualmente occasionalmente, con una distribuzione piatta la scelta può essere più lontana dal massimo. Garantisce scelta non deterministica
    plot_distribution(np_prob, max_prob, last_sample, save_path=gen_folder)

    