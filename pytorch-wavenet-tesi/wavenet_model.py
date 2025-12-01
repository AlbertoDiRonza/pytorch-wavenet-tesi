import os
import os.path
import time
import torch.nn.functional as F
from audio_data import *
from torch import nn
from wavenet_modules import *
from analisi import *

# SOTTOCLASSE DI nn.Module (CLASSE DA IL NOSTRO MODULO EREDITA)
class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    
    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 classes=256,
                 output_length=32,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False):
        
        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype
        
        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        # nn.ModuleList(): 
        # CONTAINER CHE PERMETTE DI CONSERVARE UNA LISTA DI PYTORCH MODULES
        # self.main_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)
        
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))
                
                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                                                        num_channels=residual_channels,
                                                        dilation=new_dilation,
                                                        dtype=dtype))

                # dilated convolutions
                # LISTA DI LAYERS CONVOLUZIONALI
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2
                
        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=1,
                                    bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        # self.output_length = 2 ** (layers - 1)
        self.output_length = output_length
        self.receptive_field = receptive_field
        
# FINE COSTRUTTORE - INIZIO METODI

    def wavenet(self, input, dilation_func):
            # CONVOLUZIONE 1X1 INIZIALE
            x = self.start_conv(input)
            skip = 0

            # WaveNet layers
            for i in range(self.blocks * self.layers):

                #            |----------------------------------------|     *residual*
                #            |                                        |
                #            |    |-- conv -- tanh --|                |
                # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
                #                 |-- conv -- sigm --|     |
                #                                         1x1
                #                                          |
                # ---------------------------------------> + ------------->	*skip*

                (dilation, init_dilation) = self.dilations[i]

            # QUANDO QUESTO METODO VIENE CHIMATO GLI SARà PASSATO UN PARAMETRO CHE è UNA FUNZIONE (o wavenet_dilate o queue_dilate in base a che tipo di algoritmo di generazione stiamo usando) A SOSTITUZIONE DI DILATION_FUNC
                residual = dilation_func(x, dilation, init_dilation, i)

                # dilated convolution
                # APPLICATA LA CONVOLUZIONE DILATATA AL TENSORE RESIDUAL NEL LAYER I-ESIMO
                # FILTER INFO DA PROPAGARE
                filter = self.filter_convs[i](residual)
                filter = F.tanh(filter)
                # GATE QUALE PARTE DEL FILTER DEVE ESSERE PROPAGATA
                gate = self.gate_convs[i](residual)
                gate = F.sigmoid(gate)
                x = filter * gate

                # parametrized skip connection
                s = x
                # 2 STA PER CHECK DEL NUMERO DI COLONNE - LUNGHEZZA TEMPORALE  
                if x.size(2) != 1:
                    s = dilate(x, 1, init_dilation=dilation)
                s = self.skip_convs[i](s)
                try:
                    # VENGONO PRESI GLI ULTIMI S.SIZE(2) ELEMENTI 
                    # AFFINCHè S E SKIP ABBIANO LA STESSA LUNGHEZZA TEMPORALE
                    skip = skip[:, :, -s.size(2):]
                except:
                    # SE NON è POSSIBILE, ALLORA SKIP è INIZIALIZZATO A 0
                    skip = 0
                # ESEGUI LA CONNESSIONE SU TUTTI I LAYERS 
                skip = s + skip

                x = self.residual_convs[i](x)
                # CAUSALITà 
                x = x + residual[:, :, (self.kernel_size - 1):]

            x = F.relu(skip)
            x = F.relu(self.end_conv_1(x))
            x = self.end_conv_2(x)

            return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
            x = dilate(input, dilation, init_dilation)
            return x

    def queue_dilate(self, input, dilation, init_dilation, i):
            queue = self.dilated_queues[i]
            queue.enqueue(input.data[0])
            #print("input data 0: ", input.data[0])
            x = queue.dequeue(num_deq=self.kernel_size,
                            dilation=dilation)
            #print("x dopo dequeue: ", x)
            x = x.unsqueeze(0)
            #print("x dopo unsqueeze: ", x)
            return x

    def forward(self, input):
            x = self.wavenet(input,
                            dilation_func=self.wavenet_dilate)

            # reshape output
            [n, c, l] = x.size()
            l = self.output_length
            x = x[:, :, -l:]
            # INVERTIAMO CHANNELS CON TIME E METTIAMO I DATI IN MEMORIA IN MANIERA CONTIGUA 
            x = x.transpose(1, 2).contiguous()
            # RESHAPE BIDIMENSIONALE DEL TENSORE PER PREPARARLO A OPERAZIONI SUCCESSIVE CHE SI ASPETTANO MATRICI A DUE DIMENSIONI
            x = x.view(n * l, c)
            return x


    def generate(self,
                    num_samples,
                    first_samples=None,
                    temperature=1.):
        # ATTIVA L'EVALUATION MODE DEL MODELLO, USATO ASSIEME A .TRAIN() 
        # ALCUNI MODULI SI COMPORTANO IN MANIERA DIVERSA E INDESIDERATA IN EVALUATION MODE RISPETTO ALLA TRAINING MODE
        # https://yassin01.medium.com/understanding-the-difference-between-model-eval-and-model-train-in-pytorch-48e3002ee0a2
            self.eval()
            if first_samples is None:
                first_samples = self.dtype(1).zero_()
            generated = Variable(first_samples, volatile=True)
        # CALCOLO QUANTI ZERI DEVO ESSERE AGGIUNTI ALL'INIZIO DELLA SEQUENZA GENERATA AFFINCHè ABBIA LA STESSA LUNGHEZZA DEL RECEPTIVE FIELD DEL MODELLO 
            num_pad = self.receptive_field - generated.size(0)
            # SE NUM_PAD == 0 SIGNIFICA CHE LA LUNGHEZZA DEL VETTORE DI CAMPIONI GENERATI è UGUALE AL RECEPTIVE FIELD 
            if num_pad > 0:
                generated = constant_pad_1d(generated, self.scope, pad_start=True)
                print("pad zero")

            for i in range(num_samples):
                input = Variable(torch.FloatTensor(1, self.classes, self.receptive_field).zero_())
                input = input.scatter_(1, generated[-self.receptive_field:].view(1, -1, self.receptive_field), 1.)

                x = self.wavenet(input,
                                dilation_func=self.wavenet_dilate)[:, :, -1].squeeze()

                if temperature > 0:
                    x /= temperature
                    prob = F.softmax(x, dim=0)
                    prob = prob.cpu()
                # CONVERSIONE TENSORE DI PROBABILITà DA PYTORCH A NUMPY
                    np_prob = prob.data.numpy()
                # SAMPLING STOCASTICO
                    x = np.random.choice(self.classes, p=np_prob)
                    x = Variable(torch.LongTensor([x]))  # np.array([x])
                else:
                    x = torch.max(x, 0)[1].float()
                # CONTACATENO I SAMPLE PRODOTTI IN GENERATED
                generated = torch.cat((generated, x), 0)
            # NORMALIZZIAMO I VALORI GENERATI DA 0 A 255 A -1 A 1, DIVENTANO VALORI CONTINUI 
            generated = (generated / self.classes) * 2. - 1
            # DEQUANTIZZAZIONE
            mu_gen = mu_law_expansion(generated, self.classes)
        # PORTA IL MODULO IN TRAINING MODE 
            self.train()
            return mu_gen
        
        
# IMPLEMENTA LA FAST GENERATION COME DA PAPER      
    def generate_fast(self,
                        num_samples,
                        first_samples=None,
                        temperature=1.,
                        regularize=0.,
                        progress_callback=None,
                        progress_interval=100):
            self.eval()
            # se input è none inizializza con valore di mezzo tra 0 e classes
            if first_samples is None:
                first_samples = torch.LongTensor(1).zero_() + (self.classes // 2)
            first_samples = Variable(first_samples)

            # reset queues
            for queue in self.dilated_queues:
                queue.reset()

            num_given_samples = first_samples.size(0)
            # numero di campioni totali, dati + da generare
            total_samples = num_given_samples + num_samples

            input = Variable(torch.FloatTensor(1, self.classes, 1).zero_()) # TENSORE DI DIMENSIONE 1X256X1 INIZIALIZZATO A ZERO
            
            """
            scatter_: 
            mette a 1 il valore in input nella posizione specificata da first_samples sulle colonne (dimensione 1) del tensore input.
            Sta creando un vettore one-hot encoding della finestra in input. first_samples contiene gli indici dei valori da impostare a 1
            ovvero i livelli di quantizzazione dei campioni audio iniziali forniti. Come vengono passati gli indici attraverso la view(1, -1, 1)?
            Anzitutto first_samples ha dimensione (num_given_samples,), con num_given_samples il numero di campioni iniziali forniti.
            Di questo viene preso il sottinsieme da 0 a 1 (primo campione) con first_samples[0:1], che mantiene la dimensione (1,). Questo è un
            solo campione, ma la view(1, -1, 1) lo trasforma in un tensore di dimensione (1, 1, 1), cioè una matrice 3D con un solo elemento.
            -1 indica a PyTorch di calcolare automaticamente quella dimensione in base alle altre specificate: in questo caso, dato che le altre due dimensioni sono 1,
            la dimensione -1 sarà anch'essa 1.
            Quindi scatter_ imposta a 1 l'elemento in quella posizione specifica nel tensore input di dimensione (1, 256, 1), creando così un vettore one-hot encoding
            per il primo campione iniziale.
            """
            input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)
            #print("shape della finestra di input: ", first_samples.shape)
            #print("valore del primo campione della finestra di input: ", first_samples[0:1])
            #print("valore del primo campione della finestra di input (view): ", first_samples[0:1].view(1, -1, 1))
            #print("shape della finestra di input (dopo scatter_): ", input.shape)
            #print("valore della finestra di input (dopo scatter_): ", input)
            #print("numero di campioni iniziali forniti: ", num_given_samples)

            """
            x.wavenet: ha come attributi il campione di input (one-hot) e la funzione di dilatazione (queue_dilate). Esegue il forward pass del modello sul campione in input e vengono aggiornate le code di ciascun layer.
            l'output (x) è il vettore delle probabilità del prossimo campione dato lo stato delle code e l'input corrente. In questa prima fase non vie utilizzato l'output
            perchè stiamo solo "caricando" le code con i campioni iniziali forniti. FASE DI WARM-UP. Alla fine del ciclo for le code di ciascun layer conterranno gli stati corrispondenti ai campioni iniziali, utilizzati
            ripassando ciascun campione uno alla volta per calcolare il campione successivo che viene poi usato come input per il passo successivo. 
            
            La funzione queue_dilate gestisce le code dilatate per ciascun layer:
            queue_dilate: ogni layer iesimo ho una coda dilatata (queue) associata (dilated_queues[i], array di code) che memorizza gli stati passati. 
            enqueue: aggiunge il campione corrente alla coda
            dequeue: estrae dalla coda i campioni necessari per la convoluzione dilatata, in base al fattore di dilatazione e al numero di campioni richiesti (kernel_size)
            x è il tensore che contiene gli stati passati (finestra di input per quel layer con giusta dilatazione), la convoluzione dilatata viene eseguita tr x e i filtri del layer conservati in filter_convs e gate_convs [i]
            durante il forward pass.
            
            regularizer: termine di regolarizzazione che penalizza le predizioni verso i valori centrali della distribuzione (classe 128 per 256 classi).
            viene calcolato come la distanza quadratica di ogni possibile valore di output dal valore centrale (classes/2), pesata dal parametro regularize.
            Questo aiuta a evitare che il modello generi valori troppo lontani dal centro della distribuzione.
            """
            # WARM UP
            # da 0 a 3084 (num_given_samples - 1)
            for i in range(num_given_samples - 1):
                # qui x non viene usato per generare un campione, stiamo solo aggiornando le code
                x = self.wavenet(input,
                                dilation_func=self.queue_dilate)
                input.zero_()
                # scorro i campioni uno alla volta e mi assicuro che la dimensione di input sia corretta
                input = input.scatter_(1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

                # progress feedback
                if i % progress_interval == 0:
                    if progress_callback is not None:
                        progress_callback(i, total_samples)

            #print("shape della finestra di input: ", first_samples.shape)
            #print("valore del primo campione della finestra di input: ", first_samples[i + 1:i + 2])
            #print("valore del primo campione della finestra di input (view): ", first_samples[i + 1:i + 2].view(1, -1, 1))
            #print("shape della finestra di input (dopo scatter_): ", input.shape)
            #print("valore della finestra di input (dopo scatter_): ", input)
            #print("numero di campioni iniziali forniti: ", num_given_samples)

            # generate new samples
            generated = np.array([])
            
            regularizer = torch.pow(Variable(torch.arange(self.classes)) - self.classes / 2., 2) # calcola (classe - centro)^2
            regularizer = regularizer.squeeze() * regularize # comando il peso della regolarizzazione e assicuro abbia shape corretta
            #print("regularizer: ", regularizer)
            #print("shape regularizer: ", regularizer.shape)
            
            tic = time.time()
            for i in range(num_samples):
                x = self.wavenet(input,
                                dilation_func=self.queue_dilate).squeeze() # squeeze per rimuovere dimensioni di lunghezza 1, ottengo vettore 1D di dimensione (classes,) della stessa dimensione del regularizer

                #print("uscita del modello prima della softmax (logits): ", x)

                x -= regularizer # abbassa le probabilità delle classi lontane dal centro
                
                if temperature > 0:
                    # meno deterministico
                    # sample from softmax distribution
                    x /= temperature
                    prob = F.softmax(x, dim=0)
                    prob = prob.cpu()
                    #print("probabilità normalizzate dopo softmax: ", prob)
                    np_prob = prob.data.numpy()
                    # genera campione random da un dato array 1D in base a una distribuzine di probabilità che sceglie il livello di quantizzazione del campione da generare
                    x = np.random.choice(self.classes, p=np_prob)
                    #print("campione generato (choice): ", x)
                    x = np.array([x])
                else:
                    # convert to sample value
                    # viene recuperato l'indice della classe con probabilità massima --> deterministico
                    # se x è uno scalare (tensore dim 0) converto in python number con .item() e porto su cpu, non ha forma indicizzabile
                    #x = torch.max(x, 0)[1][0]
                       # deterministico → prendo la classe con probabilità massima
                    prob = F.softmax(x, dim=0)  # calcolo comunque la distribuzione
                    np_prob = prob.detach().numpy()
                    max_index = np.argmax(np_prob)
                    x = np.array([max_index])
                    
                if isinstance(x, np.ndarray) and x.ndim > 0:
                    last_sample = x[0]
                else:
                    last_sample = x

                # le mu laws si aspettano x tra -1 e 1, qui effettuo questa normalizzazione (lineare)
                o = (x / self.classes) * 2. - 1
                #print("campione generato normalizzato: ", o)
                
                # crea array con i soli campioni generati, si perdono i campioni iniziali forniti
                generated = np.append(generated, o)

                # set new input
                # usiamo il campione generato come nuovo input per il passo successivo
                x = Variable(torch.from_numpy(x).type(torch.LongTensor))
                input.zero_()
                input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, self.classes, 1)

                if (i + 1) == 100:
                    toc = time.time()
                    print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

                # progress feedback
                if (i + num_given_samples) % progress_interval == 0:
                    if progress_callback is not None:
                        progress_callback(i + num_given_samples, total_samples)
            
            #print("Generated samples: ", generated)
            
            
            max_prob = np.argmax(np_prob)
            
            self.train()
            mu_gen = mu_law_expansion(generated, self.classes)
            return mu_gen, np_prob, max_prob, last_sample

    def parameter_count(self):
            par = list(self.parameters())
            s = sum([np.prod(list(d.size())) for d in par])
            return s

    def cpu(self, type=torch.FloatTensor):
            self.dtype = type
            for q in self.dilated_queues:
                q.dtype = self.dtype
            super().cpu()


def load_latest_model_from(location, use_cuda=True):
    files = [location + "/" + f for f in os.listdir(location)]
    # RECUPERA IL FILE PIù NUOVO IN BASE ALLA DATA DI CREAZIONE
    newest_file = max(files, key=os.path.getctime)
    print("load model " + newest_file)

    if use_cuda:
        model = torch.load(newest_file)
    else:
        model = load_to_cpu(newest_file)

    return model


def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.cpu()
    return model