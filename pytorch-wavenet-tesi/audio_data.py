import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import bisect
# CREIAMO, CARICHIAMO E PREPARIAMO IL DATASET PER L'ADDESTRMENTO DELLA RETE
# IL DATASET COME OGGETTO DI QUESTA CLASSE
# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
# "Dataset stores the samples and their corresponding labels, 
# and DataLoader wraps an iterable around the Dataset to enable 
# easy access to the samples." --> MAP STYLE DATASET
class WavenetDataset(torch.utils.data.Dataset):
    def __init__(self,
                dataset_file,
                # DA IMPLEMENTAZIONE: 
                # The item_length defines the number of samples in each item of the dataset 
                # and should always be model.receptive_field + model.output_length - 1.
                item_length,
                # DA IMPLEMENTAZIONE: 
                # The attribute target_length specifies the number of successive samples are used as a target
                # and corresponds to the output length of the model.
                target_length,
                file_location=None,
                classes=256,
                sampling_rate=16000,
                mono=True,
                normalize=False,
                dtype=np.uint8,
                train=True,
                test_stride=100):

        #           |----receptive_field----|
        #                                 |--output_length--|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                             | | | | | | | | | <-- CAMPIONI
    
        self.dataset_file = dataset_file
        self._item_length = item_length
        self._test_stride = test_stride
        self.target_length = target_length
        self.classes = classes
        
        if not os.path.isfile(dataset_file):
            assert file_location is not None, "no location for dataset files specified"
            self.mono = mono
            self.normalize = normalize

            self.sampling_rate = sampling_rate
            self.dtype = dtype
            # SE NON C'è LO CREO
            self.create_dataset(file_location, dataset_file)
        else:
            # Unknown parameters of the stored dataset
            # TODO Can these parameters be stored, too?
            # SE IL FILE è TROVATO I PARAMETRI VENGONO SETTATI A NONE E POI RECUPERATI DAL FILE
            self.mono = None
            self.normalize = None

            self.sampling_rate = None
            self.dtype = None
        
        self.train = train


        # PICKLING: self.data = np.load(self.dataset_file, mmap_mode='r') + spostate cose in calculate_length()  
        self.start_samples = [0]
        self._length = 0
        self.calculate_length()
        self.train = train
        print("one hot input")
    
    def create_dataset(self, location, out_file):
        print("create dataset from audio files at", location)
        self.dataset_file = out_file
        files = list_all_audio_files(location)
        processed_files = []
        for i, file in enumerate(files):
            print("  processed " + str(i) + " of " + str(len(files)) + " files")
            # CARICA FILE COME UNA SERIE TEMPORALE IN VIRGOLA MOBILE
            file_data, _ = lr.load(path=file,
                                   sr=self.sampling_rate,
                                   mono=self.mono)
            if self.normalize:
                file_data = lr.util.normalize(file_data) # SOLO SE NECESSARIO
                # DI DEFAULT NORMALIZA TRA -1 E 1 -> FORSE CREA PROBLEMA RISOLTO CON CLAMP NEL GETITEM, PKE NON NORMALIZZANDO LA MULAW NON FUNZIONA BENE
            # QUANTIZZO I DATI SU 256 LIVELLI E METTO IN LISTA
            quantized_data = quantize_data(file_data, self.classes)
            processed_files.append(quantized_data) # LISTA DI ARRAY 
            
        # SALVA IL DATASET IN UN FILE .NPZ (ARCHIVIO DI ARRAY NPY, ZIP (NPY FORMATO NUMPY PER SALVATAGGIO SI ARRAY IN MODO BINARIO))
        np.savez(self.dataset_file, *processed_files)
        
    def calculate_length(self):
        start_samples = [0]
        # APRIAMO IL DATASET TEMPORANEAMENTE PER EVITARE PICKLING
        with np.load(self.dataset_file, mmap_mode='r') as data:
            #LEN RITORNA NUMERO DI ELEMENTI DELLA LISTA, KEYS() RITORNA LE CHIAVI DI DIZIONARIO (DATASAET CARICATO CON NP.LOAD, SELF) SOTTO FORMA DI LISTA 
            for i in range(len(data.keys())):
                # NP.LOAD CREA UNA SORTA DI DIZIONARIO ACCESSIBILE TRAMITE CHIAVE, LE CHIAVI SONO SALVATE AUTOMATICAMENTE COME ARR_0 ARR_1 ETC. DA NP.SAVEZ
                # VIENE CALCOLATA LA POSIZIONE DI INIZIO DELL'IESIMO FILE AUDIO E AGGIUNTO ALLA LISTA, START_SAMPLES[-1] RESTITUISCE LA LUNGHEZZA DI TUTTI GLI AUDIO GIà PRESENTI NELL'ARRAY (SERVE PER RIPARTIRE SUL NUOVO INDICE)
                # CALCOLO LA LUNGHEZZA MASSIMA DISPONIBILE, -1 PER VINCOLO DI CAUSALITà (NON USCIRE DAI LIMITI DELL'ARRAY)
                start_samples.append(start_samples[-1] + len(data['arr_' + str(i)]))
                # CALCOLO LA LUNGHEZZA MASSIMA DISPONIBILE, -1 PER VINCOLO DI CAUSALITà (NON USCIRE DAI LIMITI DELL'ARRAY)
            available_length = start_samples[-1] - (self._item_length - (self.target_length - 1)) - 1
            # ARROTONDO PER DIFETTO ALL'INTERO PIù VICINO
            # SE AVAILABLE < TARGET SIGNIFICA CHE OTTERRÒ 0 DA QUESTA OPERAZIONE, EVIDENTEMENTE IL SEGNALE CORRENTE è TROPPO CORTO, CI ASSICURIAMO DI AVERE IN DATASET SOLO ESEMPI UTILIZZABILI, ABBASTANZA LUNGHI
            self._length = math.floor(available_length / self.target_length) 
            self.start_samples = start_samples
        
    def set_item_length(self, l):
        self._item_length = l
        self.calculate_length()

# I PROSSIMI DUE METODI HANNO A CHE VERRE CON LA PRIMITIVA DATASET
# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
# SONO NECESSARI PER RENDERE LA CLASSE COMPATIBILE CON PYTORCH E PERMETTERE VALUTAZIONE E ADDESTRAMENTO DEL MODELLO IN MODO AUTOMATICO E BATCH-WISE --> INSIEME DI ESEMPI == BATCH PER IL TRAINING 
# DEFINIAMO GLI ESEMPI (FINESTRE) COME COPPIA (INPUT, TARGET)
# SERVE AD ESTRARRE UN SINGOLO ESEMPIO (INPUT/TARGET) DAL INDICE IDX, VERRà USATO PER COSTRUIRE I BATCH DA PASSARE AL MODELLO DAL DATALOADER
    """
    PER CARICARE E RESTITUIRE I VALORI DEL DATASET A UN DATO INDICE, CHIAMATA IN DUE CASI: 
    1) DURANTE L'ADDESTRAMENTO, OGNI VOLTA CHE IL DATALOADER RICHIEDE UN BATCH DI ESEMPI
    2) QUANDO SI ITERA SUL DATASET/SI ESEGUE UN'OPERAZIONE DEL TIPO DATASET[IDX] 
    """
    def __getitem__(self, idx):
        # SU NP.LOAD PUò ESSERE MESSO UN CONTEX MANAGER PER EVITERE TROPPI FILE APERTI IN MEMORIA (WITH)
        
        # GET ITEM MECCANISMO STANDARD DI ACCESSO AI DAI DEL DATASET
        
        # TEST_STRIDE: PERMETTE DI PREDERE UN ESEMPIO OGNI N PER IL TESTING, CONTROLLO LA FREQUENZA E LA DISTRIBUZIONE DEGLI ESEMPI
        # IDENTIFICO LA MODALTà DI FUNZIONAMENTO E CALCOLO LA POSIZIONE DI INIZIO FINISTRRA DA ESTRARRE
        if self._test_stride < 2:
            # SOLO TRAINING
            sample_index = idx * self.target_length
        elif self.train:
            # TRAINING + TESTING
            sample_index = idx * self.target_length + math.floor(idx / (self._test_stride-1))
        else:
            # SOLO TESTING
            sample_index = self._test_stride * (idx+1) -  1
        # TROVIAMO IN QUALI FILE SI TROVA LA POSIZIONE SAMPLE_INDEX, RITORNA UN INDICE INTERNO A UN FILE
        file_index = bisect.bisect_left(self.start_samples, sample_index) - 1
        if file_index < 0: 
            file_index = 0
        position_in_file = sample_index - self.start_samples[file_index]
        # QUANTO LA FINESTRA DA ESTRARRE FINISCE NEL PROSSIMO FILE
        end_position_in_next_file = sample_index + self._item_length + 1 - self.start_samples[file_index + 1]
        # LA FINESTRA è TUTTA DENTRO AL FILE
        if end_position_in_next_file < 0:
            file_name = 'arr_' + str(file_index)
            this_file = np.load(self.dataset_file, mmap_mode='r')[file_name]
            sample = this_file[position_in_file:position_in_file + self._item_length + 1]
        else:
            # load from two files
            file1 = np.load(self.dataset_file, mmap_mode='r')['arr_' + str(file_index)]
            file2 = np.load(self.dataset_file, mmap_mode='r')['arr_' + str(file_index + 1)]
            sample1 = file1[position_in_file:]
            sample2 = file2[:end_position_in_next_file]
            # CONCATENO PER OTTENERE LA FINESTRA COMLETA
            sample = np.concatenate((sample1, sample2))

        #OTTENGO TENSORE PYTORCH
        example = torch.from_numpy(sample).type(torch.LongTensor)
        
        # CLAMP SOLO INDICI X ONE-HOT, NON SUL TARGET PERCHé I VALORI QUI NON SONO USATI COME INDICI
        # CLAMP LIMITA I VALORI DI UN TENSORE AD UN INTERVALLO --> PROBLEMA POTREBBE STARE SULLA NORMALIZZAZIONE A CREAZIONE DEL DATASET
        # Questo clamp l'ho aggiunto io pke su dataset toy risulzava non quantizzato correttamente
        example = example[:self._item_length].clamp(0, self.classes - 1)
        # MATRICE DI DIMENSIONE (CLASSES,_ITEM_LENGTH) PIENA DI ZERI
        one_hot = torch.FloatTensor(self.classes, self._item_length).zero_()
        # RIEMPIO LA MATRICE LUNGO LA DIMENSIONE 0 (CLASSES) CON UN 1  CORRISPONDENTE AL CAMPIONE, SPECIFICATO DAGLI INDICI --> OTTENGO CODIFICA ONE HOT
        one_hot.scatter_(0, example[:self._item_length].unsqueeze(0), 1.)
        # ESTRAGGO IL TARGET: GROUND TRUTH PER IL CALCOLO DELLA LOSS
        target = example[-self.target_length:].unsqueeze(0)
        return one_hot, target #--> (INPUT, TARGET) è LA TUPLA CONTENUTA DA OGNI ELEMENTO DEL DATASET, TARGET VIENE USATA COME TARGET NELL'ADDESTRAMENTO DEL MODELLO
    # TARGET HA DIMENSIONE [1, TARGET_LENGTH] --> BATCH SIZE 1, SERVE PER IL CALCOLO DELLA LOSS, L'OUTPUT DEL MODELLO HA DIMENSIONE TEMPORALE TARGET_LENGTH
    # TARGET = ETICHETTE SALVATE NEL DATASET SONO I VALORI CORRETTI CHE IL MODELLO DEVE IMPARARE A PREDIRE
    # È L CROSS ENTROPY A CONFRONTARE L'OUTPUT CON IL TARGET PER OGNI CAMPIONE PER VERIFICARE QUANTO IL MODELLO SI AVVICINA AL VALORE CORRETTO INDICATO DAL TARGET
    
# QUANTI ESEMPI IL DATALOADER PUò ESTRARRE DAL DATASET, QUANTI NE SONO DISPONIBILI
# DISTINGUIAMO LE MODALITà: NON VIENE DIVISO IL DATASET IN FILE DI TESTING E DI TRAINING MA VENGONO USATI QUESTI DUE METODI PER IMPLEMENTARE UNA LOGICA CHE SUDDIVIDA LE FINESTRE DI INPUT TRA LE DUE MODALITà
    def __len__(self):
        # CALCOLO ARRONTONDANDO PER DIFETTO: LUNGHEZZA TORTALE DIVISO LA FREQUENZA CON CUI PRENDO I GLI ESEMPI DI TESTING DAL DATASET
        test_length = math.floor(self._length / self._test_stride)
        # MOD TRAINING
        if self.train:
            # ESEMPI TOTTALI MENO QUELLI DI TESTING: RESTITUISCE ESEMPI PER IL TRANINING
            return self._length - test_length
        # MOD TESTING
        else:
            # RESTITUISCE ESEMPI DI TESTING
            return test_length

# METODI AGGIUNTIVI NON APPARTENTI AL SINGOLO OGGETTO
def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    #
    bins = np.linspace(-1, 1, classes)
    #
    quantized = np.digitize(mu_x, bins) - 1
    return quantized

def list_all_audio_files(location):
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(location):
        for filename in [f for f in filenames if f.endswith((".mp3", ".wav", ".aif", "aiff"))]:
            audio_files.append(os.path.join(dirpath, filename))

    if len(audio_files) == 0:
        print("found no audio files in " + location)
    return audio_files

def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x

def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s
