import librosa
import soundfile as sf
from scipy.io import wavfile
import torch
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *

import matplotlib.pyplot as plt
from analisi import *

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

#listen_dataset('train_samples/bach_chaconne/toy.npz')

"""
Quando accediamo al dataset viene ritornato da getitem una tupla (input, target), stampando il dataset a schermo appare come una 
lista di tensori in codifica one-hot (input) a cui è associata una label (il target) utile nella fase di training. La dimensione di
ciascun tensore di input è 256x3085 dove 256 sono i livelli di quantizzazione e 3085 è l'item_length ovvero la lunghezza della
finestra audio recuperata dal dataset. In particolare le colonne rappresentano il singolo campione, in ognuna di esse è presente solo
un 1. 
 """
sample_rate = 16000
n_tot = 100
Nc = 3085
f0 = (sample_rate * n_tot) / Nc
function = function_create(f0, sample_rate, n_tot, 'ramp')

function = quantize_data(function, data.classes)
print(function)
print(function.shape)
#   <------------------------------------>

#start_list = []
#start_idx = len(data) // 2
#num_of_inputs = 1 
"""
100 = 19 secondi, ogni elemento del dataset contiene 19/100 circa 0.2 s di audio riporducibile
 --> 3085 campioni/16000 frequenza campionamento = 0.193 s 
"""


#for i, item in enumerate([data[i] for i in range(start_idx, start_idx + 52)]): # lavoriamo con porzione intera di dataset
#for i in range(num_of_inputs): 
    # concateno dei pezzi casuali, prendo num_of_inputs pezzi casuali
    #start_data_2d = data[torch.randint(0, len(data), (1,))][0]
    #start_data_2d = item[0]
    
    # cat concatena tensori
    #if isinstance(start_data_2d, np.ndarray):
       # start_data_2d = torch.from_numpy(start_data_2d)
        
    #start_list.append(start_data_2d)
    
#start_list = torch.cat(start_list, dim=1)


""" 
La conversione da one-hot encoding a corrispondente livello di quantizzazione dei campioni può essere effettuata 
cercando l'indice di riga in cui è contenuto il valore massimo (1) di ognuno di essi. Come risultato otteniamo un tensore ( , 3085) (unidimensionale) di
interi tra 0 e 255. 
Specifico 0 per indicare la dimensione su cui voglio effettuare l'operazione di ricerca del massimo, [1] per indicare quale elemento
della tupla voglio mantenere (indice).
#https://sprintchase.com/torch-max/ 
"""
#start_data = torch.max(start_data_2d, 0)[1] # convert one hot vectors to integers
#print(start_data)
listen_start_data = (function / data.classes) * 2. - 1 # normalizzo per espansione, mulaw si aspetta -1<x<1
audio = mu_law_expansion(np.array(listen_start_data, dtype='f'), data.classes)
audio = np.tile(audio, 100) # circa 1s
# su audio più lungo stanno frequenze meno distrbuite
sf.write('start_data.wav', audio, 16000)
fourier_transform('start_data.wav')
spectogram_plot('start_data.wav')

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

def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")

generated_list = []
num_of_gens = 1
for i in range(num_of_gens): 
    generated = model.generate_fast(num_samples=80000,
                                    first_samples=torch.from_numpy(function),
                                    progress_callback=prog_callback,
                                    progress_interval=1000,
                                    temperature=1.0,
                                    regularize=0.)
    
    # cat concatena tensori
    if isinstance(generated, np.ndarray):
        generated = torch.from_numpy(generated)
        
    generated_list.append(generated)
    
    print('generate data: ',generated)
    print(quantize_data(generated, data.classes))
    
generated_list = torch.cat(generated_list, dim=0)
sf.write('latest_generated_clip.wav', np.array(generated_list, dtype='f'), 16000)
fourier_transform('latest_generated_clip.wav')
spectogram_plot('latest_generated_clip.wav')
plotting_audio_file('start_data.wav')
plotting_audio_file('latest_generated_clip.wav')
compare_audio('start_data.wav', 'latest_generated_clip.wav')


# fast generation: come vengono utilizzati i valori di input?  --> Code
# analisi dei generati

# contatta professore

# training: splittare toy in pi dataset da 10 minuti e eseguire più training + generazione, si possono dividere anche i file per bpm e fare un dataset più ristretto ma di cose più simili tra loro


